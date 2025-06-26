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
    
    max_file_size_mb: int = 100
    
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