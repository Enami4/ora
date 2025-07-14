"""
Configuration settings for the regulatory document processor.
"""

from dataclasses import dataclass
from typing import Optional
import logging

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for the document processor."""
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    max_file_size_mb: int = 200
    
    log_level: str = "INFO"
    log_file: str = "regulatory_processor.log"
    
    extract_metadata: bool = True
    clean_text: bool = True
    
    include_page_numbers: bool = True
    
    excel_max_cell_length: int = 32767
    
    supported_encodings: tuple = ('utf-8', 'latin-1', 'cp1252')
    
    # AI Validation settings
    enable_ai_validation: bool = False
    anthropic_api_key: Optional[str] = None
    ai_model: str = "claude-3-haiku-20240307"
    validation_batch_size: int = 10
    
    # Article extraction settings
    extract_articles: bool = True
    assess_materiality: bool = True
    
    # Export settings
    client_friendly_export: bool = False
    
    # Enhanced OCR settings (Phase 1)
    use_enhanced_ocr: bool = True
    ocr_target_dpi: int = 300
    ocr_languages: str = 'fra+eng'
    ocr_confidence_threshold: float = 0.6
    ocr_preprocessing: bool = True
    ocr_postprocessing: bool = True
    
    # Enhanced AI prompts settings (Phase 1)
    use_enhanced_prompts: bool = True
    enhanced_ai_model: str = "claude-3-haiku-20240307"
    ai_max_tokens: int = 4000
    ai_temperature: float = 0.1
    enable_adaptive_prompts: bool = True
    
    # Quality validation settings (Phase 1)
    enable_quality_validation: bool = True
    quality_score_threshold: float = 0.7
    auto_retry_on_low_quality: bool = True
    max_quality_retries: int = 2
    
    # Metrics and feedback settings (Phase 1)
    enable_metrics_collection: bool = True
    metrics_db_path: str = ".regulatory_cache/metrics.db"
    enable_performance_monitoring: bool = True
    auto_export_metrics: bool = False
    metrics_export_interval_days: int = 7
    
    # Advanced processing settings
    enable_structure_analysis: bool = True
    enable_cross_reference_detection: bool = True
    enable_table_extraction: bool = True
    enable_context_analysis: bool = True
    
    # Error recovery and resilience
    max_processing_retries: int = 3
    enable_graceful_degradation: bool = True
    fallback_to_basic_ocr: bool = True
    fallback_to_basic_ai: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")
        
        # Validate AI settings
        if self.enable_ai_validation and not self.anthropic_api_key:
            import os
            self.anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not self.anthropic_api_key:
                logger.warning("AI validation enabled but no API key provided. Will use fallback validation.")
        
        # Validate enhanced OCR settings
        if self.ocr_target_dpi < 150 or self.ocr_target_dpi > 600:
            logger.warning(f"OCR DPI {self.ocr_target_dpi} is outside recommended range (150-600)")
        
        if not (0.0 <= self.ocr_confidence_threshold <= 1.0):
            raise ValueError("ocr_confidence_threshold must be between 0.0 and 1.0")
        
        # Validate AI prompt settings
        if self.ai_max_tokens < 100 or self.ai_max_tokens > 8000:
            logger.warning(f"AI max_tokens {self.ai_max_tokens} is outside typical range (100-8000)")
        
        if not (0.0 <= self.ai_temperature <= 2.0):
            raise ValueError("ai_temperature must be between 0.0 and 2.0")
        
        # Validate quality settings
        if not (0.0 <= self.quality_score_threshold <= 1.0):
            raise ValueError("quality_score_threshold must be between 0.0 and 1.0")
        
        if self.max_quality_retries < 0 or self.max_quality_retries > 10:
            raise ValueError("max_quality_retries must be between 0 and 10")
        
        # Validate metrics settings
        if self.metrics_export_interval_days < 1:
            raise ValueError("metrics_export_interval_days must be at least 1")
        
        # Validate error recovery settings
        if self.max_processing_retries < 0 or self.max_processing_retries > 10:
            raise ValueError("max_processing_retries must be between 0 and 10")
        
        # Auto-adjust configurations based on dependencies
        if self.use_enhanced_prompts and not self.enable_ai_validation:
            logger.info("Enhanced prompts enabled but AI validation disabled. Enabling AI validation.")
            self.enable_ai_validation = True
        
        if self.enable_quality_validation and not self.use_enhanced_ocr:
            logger.info("Quality validation works best with enhanced OCR. Consider enabling enhanced OCR.")
        
        # Create metrics directory if metrics collection is enabled
        if self.enable_metrics_collection:
            import os
            metrics_dir = os.path.dirname(self.metrics_db_path)
            if metrics_dir and not os.path.exists(metrics_dir):
                try:
                    os.makedirs(metrics_dir, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Could not create metrics directory {metrics_dir}: {e}")


def setup_logging(config: ProcessorConfig) -> logging.Logger:
    """Set up logging based on configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('regulatory_processor')