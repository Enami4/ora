"""
Error recovery and resilience module for regulatory document processing.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from functools import wraps
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from enum import Enum


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecoveryStrategy(Enum):
    """Recovery strategy options."""
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"
    SKIP = "SKIP"
    FAIL = "FAIL"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: datetime
    error_type: str
    error_message: str
    function_name: str
    file_path: Optional[str]
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    recovery_successful: bool
    traceback_info: str
    context: Dict[str, Any]


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and maintains error history.
    
    Features:
    - Configurable retry logic with exponential backoff
    - Fallback method execution
    - Error pattern detection
    - Recovery strategy recommendation
    - Error logging and reporting
    """
    
    def __init__(self, log_dir: str = ".regulatory_logs"):
        """
        Initialize error recovery manager.
        
        Args:
            log_dir: Directory for error logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.error_history: List[ErrorRecord] = []
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        self.recovery_stats: Dict[str, Dict[str, int]] = {}
        
        # Recovery configuration
        self.max_retries = 3
        self.base_delay = 1.0  # Base delay for exponential backoff
        self.max_delay = 30.0  # Maximum delay between retries
        
        # Error categorization rules
        self.severity_rules = {
            'FileNotFoundError': ErrorSeverity.HIGH,
            'PermissionError': ErrorSeverity.HIGH,
            'MemoryError': ErrorSeverity.CRITICAL,
            'ConnectionError': ErrorSeverity.MEDIUM,
            'TimeoutError': ErrorSeverity.MEDIUM,
            'JSONDecodeError': ErrorSeverity.LOW,
            'UnicodeDecodeError': ErrorSeverity.MEDIUM,
            'AttributeError': ErrorSeverity.MEDIUM,
            'KeyError': ErrorSeverity.LOW,
            'ValueError': ErrorSeverity.LOW,
            'TypeError': ErrorSeverity.MEDIUM,
        }
        
        logger.info(f"ErrorRecoveryManager initialized with log_dir: {log_dir}")
    
    def _classify_error(self, error: Exception) -> Tuple[ErrorSeverity, RecoveryStrategy]:
        """
        Classify error and determine recovery strategy.
        
        Args:
            error: Exception object
            
        Returns:
            Tuple of (severity, recovery_strategy)
        """
        error_type = type(error).__name__
        
        # Get severity from rules
        severity = self.severity_rules.get(error_type, ErrorSeverity.MEDIUM)
        
        # Determine recovery strategy based on error type and severity
        if error_type in ['FileNotFoundError', 'PermissionError']:
            strategy = RecoveryStrategy.FALLBACK
        elif error_type in ['ConnectionError', 'TimeoutError']:
            strategy = RecoveryStrategy.RETRY
        elif error_type in ['MemoryError']:
            strategy = RecoveryStrategy.FAIL
        elif error_type in ['JSONDecodeError', 'UnicodeDecodeError']:
            strategy = RecoveryStrategy.FALLBACK
        elif error_type in ['AttributeError', 'KeyError', 'ValueError']:
            strategy = RecoveryStrategy.FALLBACK
        else:
            strategy = RecoveryStrategy.RETRY
        
        return severity, strategy
    
    def _record_error(self, error: Exception, function_name: str, 
                     file_path: Optional[str] = None, context: Optional[Dict[str, Any]] = None,
                     recovery_successful: bool = False):
        """Record error occurrence."""
        severity, strategy = self._classify_error(error)
        
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            function_name=function_name,
            file_path=file_path,
            severity=severity,
            recovery_strategy=strategy,
            recovery_successful=recovery_successful,
            traceback_info=traceback.format_exc(),
            context=context or {}
        )
        
        self.error_history.append(error_record)
        
        # Update statistics
        error_type = error_record.error_type
        if error_type not in self.recovery_stats:
            self.recovery_stats[error_type] = {
                'total_occurrences': 0,
                'successful_recoveries': 0,
                'failed_recoveries': 0
            }
        
        self.recovery_stats[error_type]['total_occurrences'] += 1
        if recovery_successful:
            self.recovery_stats[error_type]['successful_recoveries'] += 1
        else:
            self.recovery_stats[error_type]['failed_recoveries'] += 1
        
        # Log error
        log_level = {
            ErrorSeverity.LOW: logging.WARNING,
            ErrorSeverity.MEDIUM: logging.ERROR,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        logger.log(log_level, f"Error in {function_name}: {error} (Strategy: {strategy.value})")
        
        # Save to file for severe errors
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._save_error_report(error_record)
    
    def _save_error_report(self, error_record: ErrorRecord):
        """Save detailed error report to file."""
        try:
            timestamp_str = error_record.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"error_report_{timestamp_str}_{error_record.error_type}.json"
            report_path = self.log_dir / filename
            
            report_data = {
                'timestamp': error_record.timestamp.isoformat(),
                'error_type': error_record.error_type,
                'error_message': error_record.error_message,
                'function_name': error_record.function_name,
                'file_path': error_record.file_path,
                'severity': error_record.severity.value,
                'recovery_strategy': error_record.recovery_strategy.value,
                'recovery_successful': error_record.recovery_successful,
                'traceback': error_record.traceback_info,
                'context': error_record.context
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Error report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")
    
    def with_retry(self, max_retries: Optional[int] = None, 
                   base_delay: Optional[float] = None,
                   max_delay: Optional[float] = None,
                   retry_exceptions: Optional[Tuple] = None):
        """
        Decorator for retry logic with exponential backoff.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
            retry_exceptions: Tuple of exception types to retry on
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                _max_retries = max_retries or self.max_retries
                _base_delay = base_delay or self.base_delay
                _max_delay = max_delay or self.max_delay
                _retry_exceptions = retry_exceptions or (ConnectionError, TimeoutError, OSError)
                
                last_exception = None
                
                for attempt in range(_max_retries + 1):
                    try:
                        result = func(*args, **kwargs)
                        
                        # Record successful recovery if this wasn't the first attempt
                        if attempt > 0 and last_exception:
                            self._record_error(
                                last_exception, func.__name__,
                                file_path=str(args[0]) if args else None,
                                context={'attempt': attempt + 1, 'max_retries': _max_retries},
                                recovery_successful=True
                            )
                        
                        return result
                        
                    except Exception as e:
                        last_exception = e
                        
                        # Check if this exception type should be retried
                        if not isinstance(e, _retry_exceptions):
                            # Not a retryable exception, record and re-raise
                            self._record_error(
                                e, func.__name__,
                                file_path=str(args[0]) if args else None,
                                context={'attempt': attempt + 1, 'final_attempt': True}
                            )
                            raise
                        
                        if attempt < _max_retries:
                            # Calculate delay with exponential backoff
                            delay = min(_base_delay * (2 ** attempt), _max_delay)
                            logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                         f"Retrying in {delay:.1f}s...")
                            time.sleep(delay)
                        else:
                            # Final attempt failed, record and re-raise
                            self._record_error(
                                e, func.__name__,
                                file_path=str(args[0]) if args else None,
                                context={'attempt': attempt + 1, 'max_retries_exceeded': True},
                                recovery_successful=False
                            )
                            raise
                
                # This should never be reached, but just in case
                raise last_exception
            
            return wrapper
        return decorator
    
    def with_fallback(self, fallback_func: Callable, 
                      fallback_exceptions: Optional[Tuple] = None):
        """
        Decorator for fallback method execution.
        
        Args:
            fallback_func: Function to call if primary function fails
            fallback_exceptions: Tuple of exception types to trigger fallback
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                _fallback_exceptions = fallback_exceptions or (Exception,)
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if isinstance(e, _fallback_exceptions):
                        logger.warning(f"Primary method {func.__name__} failed: {e}. "
                                     f"Attempting fallback...")
                        
                        try:
                            result = fallback_func(*args, **kwargs)
                            self._record_error(
                                e, func.__name__,
                                file_path=str(args[0]) if args else None,
                                context={'fallback_used': True, 'fallback_successful': True},
                                recovery_successful=True
                            )
                            return result
                        except Exception as fallback_error:
                            self._record_error(
                                e, func.__name__,
                                file_path=str(args[0]) if args else None,
                                context={'fallback_used': True, 'fallback_error': str(fallback_error)},
                                recovery_successful=False
                            )
                            # Re-raise original exception
                            raise e
                    else:
                        # Not a fallback-triggering exception
                        self._record_error(e, func.__name__)
                        raise
            
            return wrapper
        return decorator
    
    def safe_execute(self, func: Callable, *args, 
                    default_return=None, 
                    log_errors: bool = True,
                    **kwargs) -> Any:
        """
        Safely execute a function with error handling.
        
        Args:
            func: Function to execute
            *args: Function arguments
            default_return: Default value to return on error
            log_errors: Whether to log errors
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or default_return on error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                self._record_error(
                    e, func.__name__,
                    context={'safe_execute': True, 'default_return': default_return}
                )
            return default_return
    
    def get_error_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get summary of errors from the last N hours.
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            Error summary dictionary
        """
        cutoff_time = datetime.now().timestamp() - (hours_back * 3600)
        recent_errors = [
            err for err in self.error_history 
            if err.timestamp.timestamp() > cutoff_time
        ]
        
        # Analyze error patterns
        error_types = {}
        severity_counts = {s.value: 0 for s in ErrorSeverity}
        recovery_success_rate = {}
        
        for error in recent_errors:
            # Count by type
            error_type = error.error_type
            if error_type not in error_types:
                error_types[error_type] = {'count': 0, 'files': set()}
            error_types[error_type]['count'] += 1
            if error.file_path:
                error_types[error_type]['files'].add(error.file_path)
            
            # Count by severity
            severity_counts[error.severity.value] += 1
            
            # Recovery success rate
            if error_type not in recovery_success_rate:
                recovery_success_rate[error_type] = {'total': 0, 'successful': 0}
            recovery_success_rate[error_type]['total'] += 1
            if error.recovery_successful:
                recovery_success_rate[error_type]['successful'] += 1
        
        # Convert sets to lists for JSON serialization
        for error_type in error_types:
            error_types[error_type]['files'] = list(error_types[error_type]['files'])
        
        return {
            'time_period_hours': hours_back,
            'total_errors': len(recent_errors),
            'error_by_type': error_types,
            'error_by_severity': severity_counts,
            'recovery_success_rate': {
                k: v['successful'] / v['total'] if v['total'] > 0 else 0
                for k, v in recovery_success_rate.items()
            },
            'most_problematic_files': self._get_most_problematic_files(recent_errors),
            'recommendations': self._generate_recommendations(recent_errors)
        }
    
    def _get_most_problematic_files(self, errors: List[ErrorRecord], 
                                   top_n: int = 5) -> List[Dict[str, Any]]:
        """Get files with the most errors."""
        file_errors = {}
        for error in errors:
            if error.file_path:
                if error.file_path not in file_errors:
                    file_errors[error.file_path] = {
                        'error_count': 0,
                        'error_types': set(),
                        'critical_errors': 0
                    }
                file_errors[error.file_path]['error_count'] += 1
                file_errors[error.file_path]['error_types'].add(error.error_type)
                if error.severity == ErrorSeverity.CRITICAL:
                    file_errors[error.file_path]['critical_errors'] += 1
        
        # Sort by error count and convert sets to lists
        sorted_files = sorted(
            file_errors.items(),
            key=lambda x: (x[1]['critical_errors'], x[1]['error_count']),
            reverse=True
        )[:top_n]
        
        return [
            {
                'file_path': file_path,
                'error_count': data['error_count'],
                'error_types': list(data['error_types']),
                'critical_errors': data['critical_errors']
            }
            for file_path, data in sorted_files
        ]
    
    def _generate_recommendations(self, errors: List[ErrorRecord]) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []
        
        # Analyze error patterns
        error_type_counts = {}
        for error in errors:
            error_type_counts[error.error_type] = error_type_counts.get(error.error_type, 0) + 1
        
        # Generate recommendations based on common errors
        if error_type_counts.get('FileNotFoundError', 0) > 2:
            recommendations.append("Multiple file not found errors detected. Check file paths and permissions.")
        
        if error_type_counts.get('MemoryError', 0) > 0:
            recommendations.append("Memory errors detected. Consider processing smaller chunks or increasing available memory.")
        
        if error_type_counts.get('ConnectionError', 0) > 3:
            recommendations.append("Frequent connection errors. Check network connectivity and API limits.")
        
        if error_type_counts.get('UnicodeDecodeError', 0) > 1:
            recommendations.append("Text encoding issues detected. Consider using OCR for scanned documents.")
        
        if error_type_counts.get('TimeoutError', 0) > 2:
            recommendations.append("Timeout errors detected. Consider increasing timeout values or processing smaller batches.")
        
        # Check recovery success rates
        total_errors = len(errors)
        successful_recoveries = sum(1 for e in errors if e.recovery_successful)
        recovery_rate = successful_recoveries / total_errors if total_errors > 0 else 0
        
        if recovery_rate < 0.5:
            recommendations.append(f"Low recovery success rate ({recovery_rate:.1%}). Review error handling strategies.")
        
        return recommendations


# Global error recovery manager instance
_global_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_error_manager(log_dir: str = ".regulatory_logs") -> ErrorRecoveryManager:
    """Get or create global error recovery manager instance."""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = ErrorRecoveryManager(log_dir)
    return _global_recovery_manager


# Convenience decorators using global manager
def with_retry(max_retries: int = 3, base_delay: float = 1.0, 
               retry_exceptions: Optional[Tuple] = None):
    """Convenience decorator for retry logic."""
    manager = get_error_manager()
    return manager.with_retry(max_retries, base_delay, retry_exceptions=retry_exceptions)


def with_fallback(fallback_func: Callable, fallback_exceptions: Optional[Tuple] = None):
    """Convenience decorator for fallback execution."""
    manager = get_error_manager()
    return manager.with_fallback(fallback_func, fallback_exceptions)


def safe_execute(func: Callable, *args, default_return=None, **kwargs):
    """Convenience function for safe execution."""
    manager = get_error_manager()
    return manager.safe_execute(func, *args, default_return=default_return, **kwargs)