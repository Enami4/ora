"""
Regulatory Document Processor Module

A Python module for processing regulatory PDF documents with:
- Text extraction and validation
- Document chunking  
- Structured data storage
- Excel export capabilities
- AI-powered analysis and article extraction
"""

from .processor import RegulatoryDocumentProcessor
from .config import ProcessorConfig
from .validators import AIValidator, ValidationChain, MaterialityLevel
from .interfaces import (
    TextExtractor, DocumentValidator, DocumentExporter, 
    ArticleExtractor, MaterialityAssessor, ConfigurationProvider, CacheProvider
)
from .error_handling import (
    ErrorHandler, ProcessorException, DocumentProcessingError, 
    ValidationError, ExportError, ErrorSeverity, ErrorCategory
)
from .base_exporters import BaseExcelExporter

__version__ = "2.1.0"
__author__ = "JABE Technology"

__all__ = [
    "RegulatoryDocumentProcessor", 
    "ProcessorConfig", 
    "AIValidator", 
    "ValidationChain", 
    "MaterialityLevel",
    "TextExtractor",
    "DocumentValidator", 
    "DocumentExporter",
    "ArticleExtractor",
    "MaterialityAssessor",
    "ConfigurationProvider",
    "CacheProvider", 
    "ErrorHandler",
    "ProcessorException",
    "DocumentProcessingError",
    "ValidationError",
    "ExportError", 
    "ErrorSeverity",
    "ErrorCategory",
    "BaseExcelExporter"
]