"""
Abstract interfaces for regulatory processor components.
Enables better testing, extensibility, and loose coupling.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


class TextExtractor(ABC):
    """Abstract interface for text extractors."""
    
    @abstractmethod
    def extract_text(self, file_path: str) -> Tuple[str, Dict[int, str]]:
        """
        Extract text from file.
        
        Returns:
            Tuple of (full_text, page_texts_dict)
        """
        pass
        
    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from file."""
        pass


class DocumentValidator(ABC):
    """Abstract interface for document validators."""
    
    @abstractmethod
    def validate_document(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate document and return validation results.
        
        Args:
            document_data: Document data including text, metadata, chunks
            
        Returns:
            Validation results with scores and analysis
        """
        pass


class DocumentExporter(ABC):
    """Abstract interface for document exporters."""
    
    @abstractmethod
    def export(self, documents: List[Dict[str, Any]], output_path: str, **kwargs) -> None:
        """
        Export documents to specified format.
        
        Args:
            documents: List of processed documents
            output_path: Path for output file
            **kwargs: Additional export options
        """
        pass


class ArticleExtractor(ABC):
    """Abstract interface for article extractors."""
    
    @abstractmethod
    def extract_articles(self, text: str, document_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract regulatory articles from text.
        
        Args:
            text: Document text
            document_context: Additional context (filename, type, etc.)
            
        Returns:
            List of extracted articles with metadata
        """
        pass


class MaterialityAssessor(ABC):
    """Abstract interface for materiality assessment."""
    
    @abstractmethod
    def assess_materiality(self, article: Dict[str, Any]) -> Dict[str, str]:
        """
        Assess the materiality/importance of an article.
        
        Args:
            article: Article data including content and context
            
        Returns:
            Dictionary with materiality level and reasoning
        """
        pass


class ConfigurationProvider(ABC):
    """Abstract interface for configuration providers."""
    
    @abstractmethod
    def get_config(self, profile: str = "default") -> 'ProcessorConfig':
        """Get configuration for specified profile."""
        pass
    
    @abstractmethod
    def validate_config(self, config: 'ProcessorConfig') -> List[str]:
        """Validate configuration and return any errors."""
        pass


class CacheProvider(ABC):
    """Abstract interface for caching providers."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with optional TTL."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass