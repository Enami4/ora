"""
Main processor module that orchestrates the document processing pipeline.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .config import ProcessorConfig, setup_logging
from .extractors import PDFExtractor
from .chunkers import TextChunker, SemanticChunker
from .exporters import ExcelExporter, CSVExporter
from .validators import AIValidator, ValidationChain
from .client_exporter import ClientExcelExporter
from .utils import (
    calculate_file_hash, clean_text, identify_document_type,
    validate_pdf_file, get_pdf_files, create_output_directory
)

logger = logging.getLogger(__name__)


class RegulatoryDocumentProcessor:
    """Main class for processing regulatory PDF documents."""
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        Initialize the processor with configuration.
        
        Args:
            config: Configuration object (uses defaults if not provided)
        """
        self.config = config or ProcessorConfig()
        self.logger = setup_logging(self.config)
        
        self.extractor = PDFExtractor()
        self.chunker = SemanticChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.excel_exporter = ExcelExporter(
            max_cell_length=self.config.excel_max_cell_length
        )
        self.csv_exporter = CSVExporter()
        self.client_excel_exporter = ClientExcelExporter(
            max_cell_length=self.config.excel_max_cell_length
        )
        
        # Initialize AI validation if enabled
        self.validator = None
        self.validation_chain = None
        if self.config.enable_ai_validation or self.config.extract_articles:
            self.validator = AIValidator(
                api_key=self.config.anthropic_api_key,
                model=self.config.ai_model
            )
            self.validation_chain = ValidationChain(self.validator)
        
        self.processed_documents = []
        self.processing_errors = []
    
    def process_document(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single PDF document through the entire pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing all processed data, or None if processing failed
        """
        logger.info(f"Starting processing: {pdf_path}")
        
        is_valid, error_msg = validate_pdf_file(
            pdf_path, 
            max_size_mb=self.config.max_file_size_mb
        )
        if not is_valid:
            logger.error(f"Validation failed for {pdf_path}: {error_msg}")
            self.processing_errors.append({
                'file': pdf_path,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            return None
        
        try:
            metadata = self._extract_metadata(pdf_path)
            
            full_text, page_texts = self.extractor.extract_text(pdf_path)
            
            if not full_text or not self.extractor.validate_extraction(full_text):
                logger.warning(f"No valid text extracted from {pdf_path}")
                return self._create_empty_document(metadata)
            
            cleaned_text = clean_text(full_text) if self.config.clean_text else full_text
            
            chunks = self.chunker.chunk_by_sections(cleaned_text, metadata)
            if not chunks:
                chunks = self.chunker.chunk_by_sentences(cleaned_text, metadata)
            
            statistics = self._calculate_statistics(cleaned_text, chunks)
            
            document_data = {
                'metadata': metadata,
                'full_text': full_text,
                'cleaned_text': cleaned_text,
                'page_texts': page_texts,
                'chunks': chunks,
                'statistics': statistics
            }
            
            # Apply AI validation and article extraction if enabled
            if self.validation_chain and (self.config.enable_ai_validation or self.config.extract_articles):
                logger.info(f"Running AI validation and article extraction for {pdf_path}")
                document_data = self.validation_chain.validate_document(document_data)
                
                # Log validation results
                if 'validation_results' in document_data:
                    overall_score = document_data['validation_results']['document_validation'].get('overall_score', 0)
                    logger.info(f"Document validation score: {overall_score:.1f}/100")
                    
                if 'articles' in document_data:
                    logger.info(f"Extracted {len(document_data['articles'])} articles")
            
            self.processed_documents.append(document_data)
            logger.info(f"Successfully processed {pdf_path}: {statistics['total_chunks']} chunks created")
            logger.info(f"Total processed documents now: {len(self.processed_documents)}")
            
            return document_data
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {str(e)}", exc_info=True)
            self.processing_errors.append({
                'file': pdf_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def process_directory(self, directory_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories
            
        Returns:
            List of successfully processed documents
        """
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        pdf_files = get_pdf_files(directory_path, recursive)
        logger.info(f"Found {len(pdf_files)} PDF files to process in {directory_path}")
        
        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"Processing file {i}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
            result = self.process_document(pdf_path)
            if result:
                results.append(result)
        
        logger.info(f"Completed processing: {len(results)} successful, {len(pdf_files) - len(results)} failed")
        return results
    
    def process_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process a list of specific PDF files.
        
        Args:
            file_paths: List of PDF file paths
            
        Returns:
            List of successfully processed documents
        """
        results = []
        for pdf_path in file_paths:
            result = self.process_document(pdf_path)
            if result:
                results.append(result)
        return results
    
    def export_results(self, output_path: str, format: str = 'excel', 
                      include_full_text: bool = False, include_validation: bool = None,
                      include_articles: bool = None, user_info: Optional[Dict[str, str]] = None) -> None:
        """
        Export processing results to file.
        
        Args:
            output_path: Path for output file
            format: Export format ('excel' or 'csv')
            include_full_text: Whether to include full text in export
            include_validation: Whether to include validation results (defaults to config)
            include_articles: Whether to include articles (defaults to config)
            user_info: User information to include in metadata
        """
        logger.info(f"Starting export with {len(self.processed_documents)} processed documents")
        if not self.processed_documents:
            logger.warning("No documents to export")
            logger.warning(f"Processed documents list is empty. Total processing errors: {len(self.processing_errors)}")
            if self.processing_errors:
                logger.error("Processing errors occurred:")
                for error in self.processing_errors[-5:]:  # Show last 5 errors
                    logger.error(f"  - {error['file']}: {error['error']}")
            return
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Use config defaults if not specified
        if include_validation is None:
            include_validation = self.config.enable_ai_validation
        if include_articles is None:
            include_articles = self.config.extract_articles
        
        if format.lower() == 'excel':
            self.excel_exporter.export_to_excel(
                self.processed_documents,
                output_path,
                include_full_text=include_full_text,
                include_statistics=True,
                include_validation=include_validation,
                include_articles=include_articles,
                user_info=user_info
            )
        elif format.lower() == 'csv':
            csv_path = output_path.replace('.xlsx', '.csv')
            self.csv_exporter.export_chunks_to_csv(
                self.processed_documents,
                csv_path
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        if self.processing_errors:
            self._export_errors(output_path)
    
    def export_client_report(
        self,
        output_path: str,
        user_info: Optional[Dict[str, str]] = None
    ):
        """
        Export processing results in client-friendly format.
        
        Args:
            output_path: Path for output Excel file
            user_info: User information (first_name, last_name)
        """
        if not self.processed_documents:
            logger.warning("No documents to export")
            return
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.client_exporter.export_client_report(
            self.processed_documents,
            output_path,
            user_info=user_info
        )
        
        if self.processing_errors:
            self._export_errors(output_path)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary statistics."""
        if not self.processed_documents:
            return {'status': 'No documents processed'}
        
        doc_types = {}
        total_pages = 0
        total_chunks = 0
        total_words = 0
        
        for doc in self.processed_documents:
            doc_type = doc['metadata'].get('document_type', 'OTHER')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            total_pages += doc['metadata'].get('page_count', 0)
            total_chunks += doc['statistics'].get('total_chunks', 0)
            total_words += doc['statistics'].get('total_words', 0)
        
        return {
            'total_documents': len(self.processed_documents),
            'successful_extractions': sum(
                1 for d in self.processed_documents 
                if d['statistics'].get('extraction_success', False)
            ),
            'total_errors': len(self.processing_errors),
            'document_types': doc_types,
            'total_pages': total_pages,
            'total_chunks': total_chunks,
            'total_words': total_words,
            'average_chunks_per_document': round(
                total_chunks / len(self.processed_documents), 2
            ) if self.processed_documents else 0
        }
    
    def _extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract and enhance metadata."""
        metadata = self.extractor.extract_metadata(pdf_path)
        
        metadata['file_hash'] = calculate_file_hash(pdf_path)
        metadata['document_type'] = identify_document_type(pdf_path)
        
        return metadata
    
    def _calculate_statistics(self, text: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate document statistics."""
        return {
            'total_characters': len(text),
            'total_words': len(text.split()),
            'total_sentences': text.count('.') + text.count('!') + text.count('?'),
            'total_chunks': len(chunks),
            'average_chunk_size': sum(
                c['text_length'] for c in chunks
            ) / len(chunks) if chunks else 0,
            'extraction_success': True,
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _create_empty_document(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create document structure for failed extraction."""
        return {
            'metadata': metadata,
            'full_text': '',
            'cleaned_text': '',
            'page_texts': {},
            'chunks': [],
            'statistics': {
                'total_characters': 0,
                'total_words': 0,
                'total_sentences': 0,
                'total_chunks': 0,
                'average_chunk_size': 0,
                'extraction_success': False,
                'processing_timestamp': datetime.now().isoformat()
            }
        }
    
    def _export_errors(self, output_path: str) -> None:
        """Export processing errors to a separate file."""
        error_path = output_path.replace('.xlsx', '_errors.txt')
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(f"Processing Errors Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Errors: {len(self.processing_errors)}\n")
            f.write("=" * 80 + "\n\n")
            
            for error in self.processing_errors:
                f.write(f"File: {error['file']}\n")
                f.write(f"Error: {error['error']}\n")
                f.write(f"Timestamp: {error['timestamp']}\n")
                f.write("-" * 40 + "\n")
        
        logger.info(f"Error report saved to: {error_path}")