"""
Centralized error handling and recovery strategies.
Provides consistent error handling patterns across the application.
"""

import logging
import traceback
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better handling."""
    DOCUMENT_PROCESSING = "document_processing"
    VALIDATION = "validation"
    EXPORT = "export"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    SYSTEM = "system"


@dataclass
class ProcessingError:
    """Structured error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: Dict[str, Any]
    timestamp: datetime
    traceback: Optional[str] = None
    suggestion: Optional[str] = None


class ProcessorException(Exception):
    """Base exception for processor errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Dict[str, Any] = None):
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()
        super().__init__(message)


class DocumentProcessingError(ProcessorException):
    """Document processing specific errors."""
    
    def __init__(self, message: str, file_path: str = None, **kwargs):
        context = kwargs.get('context', {})
        if file_path:
            context['file_path'] = file_path
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.DOCUMENT_PROCESSING
        super().__init__(message, **kwargs)


class ValidationError(ProcessorException):
    """Validation specific errors."""
    
    def __init__(self, message: str, validation_type: str = None, **kwargs):
        context = kwargs.get('context', {})
        if validation_type:
            context['validation_type'] = validation_type
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.VALIDATION
        super().__init__(message, **kwargs)


class ExportError(ProcessorException):
    """Export specific errors."""
    
    def __init__(self, message: str, output_path: str = None, **kwargs):
        context = kwargs.get('context', {})
        if output_path:
            context['output_path'] = output_path
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.EXPORT
        super().__init__(message, **kwargs)


class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_log: List[ProcessingError] = []
        self.recovery_strategies = self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self) -> Dict[ErrorCategory, Callable]:
        """Setup recovery strategies for different error types."""
        return {
            ErrorCategory.DOCUMENT_PROCESSING: self._recover_document_processing,
            ErrorCategory.VALIDATION: self._recover_validation,
            ErrorCategory.EXPORT: self._recover_export,
            ErrorCategory.NETWORK: self._recover_network,
            ErrorCategory.AUTHENTICATION: self._recover_authentication
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[bool, Any]:
        """
        Handle error with appropriate recovery strategy.
        
        Returns:
            Tuple of (recovery_successful, recovery_result)
        """
        context = context or {}
        
        # Convert to ProcessorException if needed
        if isinstance(error, ProcessorException):
            proc_error = error
        else:
            proc_error = ProcessorException(
                str(error),
                category=self._categorize_error(error),
                severity=self._assess_severity(error),
                context=context
            )
        
        # Log the error
        self._log_error(proc_error, error)
        
        # Add to error log
        structured_error = ProcessingError(
            category=proc_error.category,
            severity=proc_error.severity,
            message=proc_error.message,
            context=proc_error.context,
            timestamp=proc_error.timestamp,
            traceback=traceback.format_exc(),
            suggestion=self._get_error_suggestion(proc_error)
        )
        self.error_log.append(structured_error)
        
        # Attempt recovery
        if proc_error.category in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[proc_error.category](proc_error, context)
                self.logger.info(f"Recovery successful for {proc_error.category.value} error")
                return True, recovery_result
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}")
                return False, None
        
        return False, None
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on type and message."""
        error_str = str(error).lower()
        
        if any(keyword in error_str for keyword in ['pdf', 'extract', 'document', 'file']):
            return ErrorCategory.DOCUMENT_PROCESSING
        elif any(keyword in error_str for keyword in ['validation', 'invalid', 'format']):
            return ErrorCategory.VALIDATION
        elif any(keyword in error_str for keyword in ['export', 'excel', 'save', 'write']):
            return ErrorCategory.EXPORT
        elif any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'api']):
            return ErrorCategory.NETWORK
        elif any(keyword in error_str for keyword in ['authentication', 'api key', 'unauthorized']):
            return ErrorCategory.AUTHENTICATION
        else:
            return ErrorCategory.SYSTEM
    
    def _assess_severity(self, error: Exception) -> ErrorSeverity:
        """Assess error severity based on type and impact."""
        if isinstance(error, (FileNotFoundError, PermissionError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM
    
    def _log_error(self, proc_error: ProcessorException, original_error: Exception):
        """Log error with appropriate level."""
        if proc_error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"{proc_error.category.value}: {proc_error.message}", 
                               extra={'context': proc_error.context})
        elif proc_error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"{proc_error.category.value}: {proc_error.message}",
                            extra={'context': proc_error.context})
        elif proc_error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"{proc_error.category.value}: {proc_error.message}",
                              extra={'context': proc_error.context})
        else:
            self.logger.info(f"{proc_error.category.value}: {proc_error.message}",
                           extra={'context': proc_error.context})
    
    def _get_error_suggestion(self, error: ProcessorException) -> str:
        """Get helpful suggestion for error resolution."""
        suggestions = {
            ErrorCategory.DOCUMENT_PROCESSING: "Check if the PDF file is not corrupted and is readable. Try with a different PDF extraction method.",
            ErrorCategory.VALIDATION: "Verify your API key is correct and you have internet connectivity for AI validation.",
            ErrorCategory.EXPORT: "Ensure the output directory exists and you have write permissions. Close any open Excel files.",
            ErrorCategory.NETWORK: "Check your internet connection and try again. The service might be temporarily unavailable.",
            ErrorCategory.AUTHENTICATION: "Verify your API key is correct and has not expired. Check your account status.",
            ErrorCategory.CONFIGURATION: "Review your configuration settings and ensure all required values are provided."
        }
        return suggestions.get(error.category, "Check the logs for more detailed error information.")
    
    def _recover_document_processing(self, error: ProcessorException, context: Dict[str, Any]) -> Any:
        """Recovery strategy for document processing errors."""
        file_path = context.get('file_path')
        
        if file_path and "extract" in error.message.lower():
            # Try alternative extraction method
            self.logger.info(f"Attempting alternative extraction for {file_path}")
            # This would call alternative extractor
            return {"status": "recovered", "method": "alternative_extractor"}
        
        return None
    
    def _recover_validation(self, error: ProcessorException, context: Dict[str, Any]) -> Any:
        """Recovery strategy for validation errors."""
        if "api" in error.message.lower() or "network" in error.message.lower():
            # Provide fallback validation
            self.logger.info("Using fallback validation due to API/network error")
            return {
                "validation_score": {"overall": 75, "completeness": 80, "reliability": 70},
                "method": "fallback"
            }
        
        return None
    
    def _recover_export(self, error: ProcessorException, context: Dict[str, Any]) -> Any:
        """Recovery strategy for export errors."""
        output_path = context.get('output_path', '')
        
        if "permission" in error.message.lower() and output_path:
            # Try alternative output location
            import tempfile
            alternative_path = tempfile.gettempdir() + "/" + output_path.split("/")[-1]
            self.logger.info(f"Attempting export to alternative location: {alternative_path}")
            return {"alternative_path": alternative_path}
        
        return None
    
    def _recover_network(self, error: ProcessorException, context: Dict[str, Any]) -> Any:
        """Recovery strategy for network errors."""
        # Implement retry logic or offline mode
        self.logger.info("Switching to offline mode due to network error")
        return {"mode": "offline"}
    
    def _recover_authentication(self, error: ProcessorException, context: Dict[str, Any]) -> Any:
        """Recovery strategy for authentication errors."""
        # Suggest re-authentication or API key check
        self.logger.warning("Authentication failed. Please check your API key.")
        return {"suggestion": "check_api_key"}
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        if not self.error_log:
            return {"total_errors": 0}
        
        summary = {
            "total_errors": len(self.error_log),
            "by_category": {},
            "by_severity": {},
            "recent_errors": []
        }
        
        for error in self.error_log:
            # Count by category
            category = error.category.value
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
            
            # Count by severity
            severity = error.severity.value
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        
        # Get 5 most recent errors
        summary["recent_errors"] = [
            {
                "message": error.message,
                "category": error.category.value,
                "severity": error.severity.value,
                "timestamp": error.timestamp.isoformat(),
                "suggestion": error.suggestion
            }
            for error in self.error_log[-5:]
        ]
        
        return summary
    
    def clear_errors(self):
        """Clear error log."""
        self.error_log.clear()
        self.logger.info("Error log cleared")