"""
Metrics collection and feedback system for continuous improvement.
This module implements performance tracking, feedback loops, and system monitoring.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from collections import defaultdict
import threading
from .improvements import QualityAssurance

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Metrics for a single document processing session."""
    session_id: str
    file_name: str
    file_size: int
    page_count: int
    processing_time: float
    extraction_method: str
    ocr_confidence: float
    text_completeness: float
    article_count: int
    quality_score: float
    status: str
    error_count: int
    warnings_count: int
    user_feedback: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    total_documents: int
    successful_processing: int
    failed_processing: int
    average_processing_time: float
    average_quality_score: float
    average_ocr_confidence: float
    average_article_count: float
    common_errors: List[Dict[str, Any]]
    quality_trends: Dict[str, float]
    performance_trends: Dict[str, float]
    user_satisfaction: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MetricsCollector:
    """Collects and manages performance metrics and feedback."""
    
    def __init__(self, db_path: str = ".regulatory_cache/metrics.db"):
        """
        Initialize metrics collector.
        
        Args:
            db_path: Path to SQLite database for storing metrics
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self.qa_framework = QualityAssurance()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS processing_metrics (
                        session_id TEXT PRIMARY KEY,
                        file_name TEXT,
                        file_size INTEGER,
                        page_count INTEGER,
                        processing_time REAL,
                        extraction_method TEXT,
                        ocr_confidence REAL,
                        text_completeness REAL,
                        article_count INTEGER,
                        quality_score REAL,
                        status TEXT,
                        error_count INTEGER,
                        warnings_count INTEGER,
                        user_feedback TEXT,
                        timestamp TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_documents INTEGER,
                        successful_processing INTEGER,
                        failed_processing INTEGER,
                        average_processing_time REAL,
                        average_quality_score REAL,
                        average_ocr_confidence REAL,
                        average_article_count REAL,
                        common_errors TEXT,
                        quality_trends TEXT,
                        performance_trends TEXT,
                        user_satisfaction REAL,
                        timestamp TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        feedback_type TEXT,
                        rating INTEGER,
                        comments TEXT,
                        suggestions TEXT,
                        timestamp TEXT,
                        FOREIGN KEY (session_id) REFERENCES processing_metrics (session_id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS error_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        error_type TEXT,
                        error_message TEXT,
                        function_name TEXT,
                        stack_trace TEXT,
                        timestamp TEXT
                    )
                """)
                
                conn.commit()
                logger.info("Metrics database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")
    
    def record_processing_metrics(self, metrics: ProcessingMetrics):
        """
        Record metrics for a document processing session.
        
        Args:
            metrics: ProcessingMetrics object with session data
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO processing_metrics 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metrics.session_id,
                        metrics.file_name,
                        metrics.file_size,
                        metrics.page_count,
                        metrics.processing_time,
                        metrics.extraction_method,
                        metrics.ocr_confidence,
                        metrics.text_completeness,
                        metrics.article_count,
                        metrics.quality_score,
                        metrics.status,
                        metrics.error_count,
                        metrics.warnings_count,
                        json.dumps(metrics.user_feedback) if metrics.user_feedback else None,
                        metrics.timestamp.isoformat()
                    ))
                    conn.commit()
                    
            logger.debug(f"Recorded processing metrics for session {metrics.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to record processing metrics: {e}")
    
    def record_user_feedback(self, session_id: str, feedback_type: str, 
                           rating: int, comments: str = "", suggestions: str = ""):
        """
        Record user feedback for a processing session.
        
        Args:
            session_id: Processing session identifier
            feedback_type: Type of feedback (accuracy, speed, usability, etc.)
            rating: Rating from 1-5
            comments: User comments
            suggestions: User suggestions for improvement
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO user_feedback 
                        (session_id, feedback_type, rating, comments, suggestions, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        feedback_type,
                        rating,
                        comments,
                        suggestions,
                        datetime.now().isoformat()
                    ))
                    conn.commit()
                    
            logger.info(f"Recorded user feedback for session {session_id}: {rating}/5")
            
        except Exception as e:
            logger.error(f"Failed to record user feedback: {e}")
    
    def record_error(self, session_id: str, error_type: str, error_message: str,
                    function_name: str = "", stack_trace: str = ""):
        """
        Record error information for analysis.
        
        Args:
            session_id: Processing session identifier
            error_type: Type of error
            error_message: Error message
            function_name: Function where error occurred
            stack_trace: Stack trace if available
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO error_log 
                        (session_id, error_type, error_message, function_name, stack_trace, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        error_type,
                        error_message,
                        function_name,
                        stack_trace,
                        datetime.now().isoformat()
                    ))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to record error: {e}")
    
    def get_system_metrics(self, days_back: int = 30) -> SystemMetrics:
        """
        Calculate system-wide metrics for the specified period.
        
        Args:
            days_back: Number of days to look back for metrics
            
        Returns:
            SystemMetrics object with aggregated data
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            with sqlite3.connect(self.db_path) as conn:
                # Basic counts and averages
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                        AVG(processing_time) as avg_time,
                        AVG(quality_score) as avg_quality,
                        AVG(ocr_confidence) as avg_ocr,
                        AVG(article_count) as avg_articles
                    FROM processing_metrics 
                    WHERE timestamp > ?
                """, (cutoff_date.isoformat(),))
                
                row = cursor.fetchone()
                if not row or row[0] == 0:
                    return self._create_empty_system_metrics()
                
                (total, successful, failed, avg_time, avg_quality, 
                 avg_ocr, avg_articles) = row
                
                # Get common errors
                cursor = conn.execute("""
                    SELECT error_type, COUNT(*) as count, error_message
                    FROM error_log 
                    WHERE timestamp > ?
                    GROUP BY error_type, error_message
                    ORDER BY count DESC
                    LIMIT 10
                """, (cutoff_date.isoformat(),))
                
                common_errors = [
                    {
                        'error_type': row[0],
                        'count': row[1],
                        'message': row[2]
                    }
                    for row in cursor.fetchall()
                ]
                
                # Calculate trends (compare to previous period)
                quality_trends = self._calculate_quality_trends(days_back)
                performance_trends = self._calculate_performance_trends(days_back)
                
                # Get user satisfaction
                cursor = conn.execute("""
                    SELECT AVG(rating) 
                    FROM user_feedback 
                    WHERE timestamp > ?
                """, (cutoff_date.isoformat(),))
                
                user_satisfaction = cursor.fetchone()[0] or 0.0
                
                return SystemMetrics(
                    total_documents=total or 0,
                    successful_processing=successful or 0,
                    failed_processing=failed or 0,
                    average_processing_time=avg_time or 0.0,
                    average_quality_score=avg_quality or 0.0,
                    average_ocr_confidence=avg_ocr or 0.0,
                    average_article_count=avg_articles or 0.0,
                    common_errors=common_errors,
                    quality_trends=quality_trends,
                    performance_trends=performance_trends,
                    user_satisfaction=user_satisfaction
                )
                
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return self._create_empty_system_metrics()
    
    def get_performance_report(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            days_back: Number of days to include in report
            
        Returns:
            Dictionary with detailed performance analysis
        """
        try:
            system_metrics = self.get_system_metrics(days_back)
            
            # Calculate success rate
            success_rate = 0.0
            if system_metrics.total_documents > 0:
                success_rate = system_metrics.successful_processing / system_metrics.total_documents
            
            # Get quality distribution
            quality_distribution = self._get_quality_distribution(days_back)
            
            # Get processing time distribution
            time_distribution = self._get_processing_time_distribution(days_back)
            
            # Get improvement recommendations
            recommendations = self._generate_improvement_recommendations(system_metrics)
            
            # Get benchmark comparison
            benchmark_comparison = self._compare_to_benchmarks(system_metrics)
            
            report = {
                'period': f"Last {days_back} days",
                'generated_at': datetime.now().isoformat(),
                'overview': {
                    'total_documents': system_metrics.total_documents,
                    'success_rate': success_rate,
                    'average_quality_score': system_metrics.average_quality_score,
                    'average_processing_time': system_metrics.average_processing_time,
                    'user_satisfaction': system_metrics.user_satisfaction
                },
                'quality_metrics': {
                    'average_ocr_confidence': system_metrics.average_ocr_confidence,
                    'average_article_extraction': system_metrics.average_article_count,
                    'quality_distribution': quality_distribution,
                    'quality_trends': system_metrics.quality_trends
                },
                'performance_metrics': {
                    'processing_time_distribution': time_distribution,
                    'performance_trends': system_metrics.performance_trends,
                    'error_analysis': system_metrics.common_errors
                },
                'benchmark_comparison': benchmark_comparison,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=7)).isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def export_metrics(self, file_path: str, days_back: int = 30):
        """
        Export metrics to a JSON file.
        
        Args:
            file_path: Path to export file
            days_back: Number of days of data to export
        """
        try:
            metrics = self.get_system_metrics(days_back)
            report = self.get_performance_report(days_back)
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'period_days': days_back,
                'system_metrics': asdict(metrics),
                'performance_report': report
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Metrics exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def _calculate_quality_trends(self, days_back: int) -> Dict[str, float]:
        """Calculate quality trends compared to previous period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            previous_cutoff = cutoff_date - timedelta(days=days_back)
            
            with sqlite3.connect(self.db_path) as conn:
                # Current period
                cursor = conn.execute("""
                    SELECT AVG(quality_score), AVG(ocr_confidence), AVG(text_completeness)
                    FROM processing_metrics 
                    WHERE timestamp > ?
                """, (cutoff_date.isoformat(),))
                current = cursor.fetchone()
                
                # Previous period
                cursor = conn.execute("""
                    SELECT AVG(quality_score), AVG(ocr_confidence), AVG(text_completeness)
                    FROM processing_metrics 
                    WHERE timestamp BETWEEN ? AND ?
                """, (previous_cutoff.isoformat(), cutoff_date.isoformat()))
                previous = cursor.fetchone()
                
                if current and previous and all(x is not None for x in current + previous):
                    return {
                        'quality_score_change': ((current[0] - previous[0]) / previous[0] * 100) if previous[0] > 0 else 0,
                        'ocr_confidence_change': ((current[1] - previous[1]) / previous[1] * 100) if previous[1] > 0 else 0,
                        'text_completeness_change': ((current[2] - previous[2]) / previous[2] * 100) if previous[2] > 0 else 0
                    }
                
        except Exception as e:
            logger.error(f"Failed to calculate quality trends: {e}")
        
        return {'quality_score_change': 0.0, 'ocr_confidence_change': 0.0, 'text_completeness_change': 0.0}
    
    def _calculate_performance_trends(self, days_back: int) -> Dict[str, float]:
        """Calculate performance trends compared to previous period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            previous_cutoff = cutoff_date - timedelta(days=days_back)
            
            with sqlite3.connect(self.db_path) as conn:
                # Current period
                cursor = conn.execute("""
                    SELECT AVG(processing_time), COUNT(*), SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    FROM processing_metrics 
                    WHERE timestamp > ?
                """, (cutoff_date.isoformat(),))
                current = cursor.fetchone()
                
                # Previous period
                cursor = conn.execute("""
                    SELECT AVG(processing_time), COUNT(*), SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    FROM processing_metrics 
                    WHERE timestamp BETWEEN ? AND ?
                """, (previous_cutoff.isoformat(), cutoff_date.isoformat()))
                previous = cursor.fetchone()
                
                if current and previous and all(x is not None for x in current + previous):
                    current_success_rate = current[2] / current[1] if current[1] > 0 else 0
                    previous_success_rate = previous[2] / previous[1] if previous[1] > 0 else 0
                    
                    return {
                        'processing_time_change': ((current[0] - previous[0]) / previous[0] * 100) if previous[0] > 0 else 0,
                        'throughput_change': ((current[1] - previous[1]) / previous[1] * 100) if previous[1] > 0 else 0,
                        'success_rate_change': ((current_success_rate - previous_success_rate) * 100) if previous_success_rate > 0 else 0
                    }
                
        except Exception as e:
            logger.error(f"Failed to calculate performance trends: {e}")
        
        return {'processing_time_change': 0.0, 'throughput_change': 0.0, 'success_rate_change': 0.0}
    
    def _get_quality_distribution(self, days_back: int) -> Dict[str, int]:
        """Get distribution of quality scores."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT quality_score FROM processing_metrics 
                    WHERE timestamp > ? AND quality_score IS NOT NULL
                """, (cutoff_date.isoformat(),))
                
                scores = [row[0] for row in cursor.fetchall()]
                
                distribution = {
                    'excellent': sum(1 for s in scores if s >= 0.9),
                    'good': sum(1 for s in scores if 0.8 <= s < 0.9),
                    'acceptable': sum(1 for s in scores if 0.7 <= s < 0.8),
                    'needs_improvement': sum(1 for s in scores if 0.5 <= s < 0.7),
                    'poor': sum(1 for s in scores if s < 0.5)
                }
                
                return distribution
                
        except Exception as e:
            logger.error(f"Failed to get quality distribution: {e}")
            return {}
    
    def _get_processing_time_distribution(self, days_back: int) -> Dict[str, int]:
        """Get distribution of processing times."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT processing_time FROM processing_metrics 
                    WHERE timestamp > ? AND processing_time IS NOT NULL
                """, (cutoff_date.isoformat(),))
                
                times = [row[0] for row in cursor.fetchall()]
                
                distribution = {
                    'very_fast': sum(1 for t in times if t < 30),    # < 30 seconds
                    'fast': sum(1 for t in times if 30 <= t < 60),   # 30-60 seconds
                    'normal': sum(1 for t in times if 60 <= t < 120), # 1-2 minutes
                    'slow': sum(1 for t in times if 120 <= t < 300),  # 2-5 minutes
                    'very_slow': sum(1 for t in times if t >= 300)    # > 5 minutes
                }
                
                return distribution
                
        except Exception as e:
            logger.error(f"Failed to get processing time distribution: {e}")
            return {}
    
    def _generate_improvement_recommendations(self, metrics: SystemMetrics) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        # Quality-based recommendations
        if metrics.average_quality_score < 0.7:
            recommendations.append("Quality scores are below target - consider OCR parameter tuning")
        
        if metrics.average_ocr_confidence < 0.8:
            recommendations.append("OCR confidence is low - implement enhanced preprocessing")
        
        # Performance-based recommendations
        if metrics.average_processing_time > 120:  # > 2 minutes
            recommendations.append("Processing time is high - optimize extraction pipeline")
        
        if metrics.failed_processing > metrics.total_documents * 0.1:  # > 10% failure rate
            recommendations.append("High failure rate - review error handling and input validation")
        
        # User satisfaction recommendations
        if metrics.user_satisfaction < 3.5:  # < 3.5/5
            recommendations.append("User satisfaction is low - gather more feedback and improve UX")
        
        # Trend-based recommendations
        if metrics.quality_trends.get('quality_score_change', 0) < -5:
            recommendations.append("Quality declining - investigate recent changes and data sources")
        
        if not recommendations:
            recommendations.append("System performing well - continue current practices")
        
        return recommendations
    
    def _compare_to_benchmarks(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Compare current metrics to established benchmarks."""
        benchmarks = self.qa_framework.get_quality_metrics()
        
        comparison = {
            'ocr_quality': {
                'current': metrics.average_ocr_confidence,
                'target': benchmarks['ocr_quality']['character_accuracy'],
                'status': 'good' if metrics.average_ocr_confidence >= benchmarks['ocr_quality']['character_accuracy'] else 'needs_improvement'
            },
            'processing_success_rate': {
                'current': metrics.successful_processing / max(metrics.total_documents, 1),
                'target': benchmarks['overall_quality']['processing_success_rate'],
                'status': 'good' if (metrics.successful_processing / max(metrics.total_documents, 1)) >= benchmarks['overall_quality']['processing_success_rate'] else 'needs_improvement'
            },
            'user_satisfaction': {
                'current': metrics.user_satisfaction / 5.0,  # Convert to 0-1 scale
                'target': benchmarks['overall_quality']['client_satisfaction'],
                'status': 'good' if (metrics.user_satisfaction / 5.0) >= benchmarks['overall_quality']['client_satisfaction'] else 'needs_improvement'
            }
        }
        
        return comparison
    
    def _create_empty_system_metrics(self) -> SystemMetrics:
        """Create empty system metrics for when no data is available."""
        return SystemMetrics(
            total_documents=0,
            successful_processing=0,
            failed_processing=0,
            average_processing_time=0.0,
            average_quality_score=0.0,
            average_ocr_confidence=0.0,
            average_article_count=0.0,
            common_errors=[],
            quality_trends={},
            performance_trends={},
            user_satisfaction=0.0
        )


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_processing_session(session_id: str, file_name: str, processing_result: Dict[str, Any], 
                             processing_time: float):
    """
    Convenience function to record a complete processing session.
    
    Args:
        session_id: Unique session identifier
        file_name: Name of processed file
        processing_result: Complete processing result dictionary
        processing_time: Time taken for processing in seconds
    """
    try:
        collector = get_metrics_collector()
        
        # Extract metrics from processing result
        metadata = processing_result.get('metadata', {})
        quality_validation = processing_result.get('quality_validation', {})
        quality_metrics = quality_validation.get('quality_metrics', {})
        articles = processing_result.get('articles', [])
        
        metrics = ProcessingMetrics(
            session_id=session_id,
            file_name=file_name,
            file_size=metadata.get('file_size', 0),
            page_count=metadata.get('page_count', 1),
            processing_time=processing_time,
            extraction_method=metadata.get('extraction_method', 'unknown'),
            ocr_confidence=quality_metrics.get('ocr_confidence', 0.0),
            text_completeness=quality_metrics.get('text_completeness', 0.0),
            article_count=len(articles),
            quality_score=quality_metrics.get('overall_score', 0.0),
            status='success' if quality_validation.get('is_valid', False) else 'failed',
            error_count=len(processing_result.get('processing_errors', [])),
            warnings_count=len(quality_validation.get('recommendations', []))
        )
        
        collector.record_processing_metrics(metrics)
        
    except Exception as e:
        logger.error(f"Failed to record processing session: {e}")