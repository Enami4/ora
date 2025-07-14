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
from .cartographie_exporter import CartographieExporter
from .quality_validator import QualityValidator
from .metrics_collector import get_metrics_collector, record_processing_session
from .page_selector import PageSelection, PageRangeParser
from .caching import get_cache
from .error_recovery import get_error_manager, with_retry, safe_execute
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
        
        # Initialize PDF extractor with enhanced OCR if enabled
        self.extractor = PDFExtractor(use_enhanced_ocr=self.config.use_enhanced_ocr)
        
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
        self.cartographie_exporter = CartographieExporter(config=self.config)
        
        # Initialize quality validator if enabled
        self.quality_validator = None
        if self.config.enable_quality_validation:
            self.quality_validator = QualityValidator()
        
        # Initialize AI validation if enabled
        self.validator = None
        self.validation_chain = None
        if self.config.enable_ai_validation or self.config.extract_articles:
            self.validator = AIValidator(
                api_key=self.config.anthropic_api_key,
                model=self.config.enhanced_ai_model or self.config.ai_model,
                use_enhanced_prompts=self.config.use_enhanced_prompts
            )
            self.validation_chain = ValidationChain(self.validator)
        
        # Initialize metrics collector if enabled
        self.metrics_collector = None
        if self.config.enable_metrics_collection:
            self.metrics_collector = get_metrics_collector()
        
        self.processed_documents = []
        self.processing_errors = []
    
    @with_retry(max_retries=1, base_delay=2.0, retry_exceptions=(OSError, PermissionError))
    def process_document(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single PDF document through the entire pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing all processed data, or None if processing failed
        """
        # Generate session ID and start timing
        import uuid
        session_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting processing: {pdf_path} (Session: {session_id})")
        error_manager = get_error_manager()
        
        # File validation with error recovery
        validation_result = safe_execute(
            validate_pdf_file,
            pdf_path, 
            max_size_mb=self.config.max_file_size_mb,
            default_return=(False, "Validation failed")
        )
        
        is_valid, error_msg = validation_result
        if not is_valid:
            logger.error(f"Validation failed for {pdf_path}: {error_msg}")
            self.processing_errors.append({
                'file': pdf_path,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
            # Record failed processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            try:
                metrics_collector = get_metrics_collector()
                metrics_collector.record_error(
                    session_id=session_id,
                    error_type="validation_error",
                    error_message=error_msg,
                    function_name="process_document"
                )
            except Exception as e:
                logger.warning(f"Failed to record error metrics: {e}")
            
            return None
        
        try:
            # Extract metadata with error recovery
            metadata = safe_execute(
                self._extract_metadata,
                pdf_path,
                default_return={'file_name': os.path.basename(pdf_path), 'extraction_date': datetime.now().isoformat()}
            )
            
            # Extract text with built-in error recovery (from decorator)
            extraction_result = safe_execute(
                self.extractor.extract_text,
                pdf_path,
                default_return=("", {})
            )
            
            full_text, page_texts = extraction_result
            
            if not full_text or not safe_execute(self.extractor.validate_extraction, full_text, default_return=False):
                logger.warning(f"No valid text extracted from {pdf_path}")
                return self._create_empty_document(metadata)
            
            # Clean text safely
            cleaned_text = safe_execute(
                clean_text,
                full_text,
                default_return=full_text
            ) if self.config.clean_text else full_text
            
            # Chunk text with fallback
            chunks = safe_execute(
                self.chunker.chunk_by_sections,
                cleaned_text, metadata,
                default_return=[]
            )
            
            if not chunks:
                chunks = safe_execute(
                    self.chunker.chunk_by_sentences,
                    cleaned_text, metadata,
                    default_return=[]
                )
            
            # Calculate statistics safely
            statistics = safe_execute(
                self._calculate_statistics,
                cleaned_text, chunks,
                default_return={
                    'total_characters': len(cleaned_text),
                    'total_words': len(cleaned_text.split()),
                    'total_chunks': len(chunks),
                    'extraction_success': bool(full_text)
                }
            )
            
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
                
                # Safely run validation chain
                try:
                    validated_data = self.validation_chain.validate_document(document_data)
                except Exception as e:
                    logger.error(f"Validation chain failed: {e}")
                    # Use the original document data
                    validated_data = document_data
                    
                    # Try basic article extraction as fallback
                    if 'cleaned_text' in document_data:
                        logger.info("Attempting fallback article extraction")
                        articles = self._extract_articles_basic(document_data['cleaned_text'])
                        if articles:
                            validated_data['articles'] = articles
                            logger.info(f"Fallback extraction found {len(articles)} articles")
                
                if validated_data:
                    document_data = validated_data
                
                # Log validation results safely
                if 'validation_results' in document_data:
                    overall_score = document_data['validation_results'].get('document_validation', {}).get('overall_score', 0)
                    logger.info(f"Document validation score: {overall_score:.1f}/100")
                    
                if 'articles' in document_data:
                    logger.info(f"Extracted {len(document_data['articles'])} articles")
            
            # Apply quality validation to the complete processing result if enabled
            quality_validation = None
            if self.quality_validator:
                logger.info(f"Running quality validation for {pdf_path}")
                quality_validation = safe_execute(
                    self.quality_validator.validate_document_processing,
                    document_data,
                    default_return=None
                )
            
            if quality_validation:
                document_data['quality_validation'] = {
                    'is_valid': quality_validation.is_valid,
                    'quality_metrics': {
                        'ocr_confidence': quality_validation.quality_metrics.ocr_confidence,
                        'text_completeness': quality_validation.quality_metrics.text_completeness,
                        'structure_coherence': quality_validation.quality_metrics.structure_coherence,
                        'article_extraction_rate': quality_validation.quality_metrics.article_extraction_rate,
                        'regulatory_term_accuracy': quality_validation.quality_metrics.regulatory_term_accuracy,
                        'overall_score': quality_validation.quality_metrics.overall_score,
                        'status': quality_validation.quality_metrics.status.value,
                        'timestamp': quality_validation.quality_metrics.timestamp.isoformat()
                    },
                    'validation_details': quality_validation.validation_details,
                    'recommendations': quality_validation.recommendations
                }
                
                # Log quality validation results
                overall_quality = quality_validation.quality_metrics.overall_score
                status = quality_validation.quality_metrics.status.value
                logger.info(f"Quality validation - Score: {overall_quality:.2f}, Status: {status}")
                
                if not quality_validation.is_valid:
                    logger.warning(f"Quality validation failed for {pdf_path}")
                    for recommendation in quality_validation.recommendations[:3]:  # Log first 3 recommendations
                        logger.warning(f"Recommendation: {recommendation}")
                else:
                    logger.info(f"Quality validation passed for {pdf_path}")
            
            # Record processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            try:
                record_processing_session(
                    session_id=session_id,
                    file_name=os.path.basename(pdf_path),
                    processing_result=document_data,
                    processing_time=processing_time
                )
                logger.debug(f"Recorded processing metrics for session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to record processing metrics: {e}")
            
            # Add session metadata to document data
            document_data['session_info'] = {
                'session_id': session_id,
                'processing_time': processing_time,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
            
            self.processed_documents.append(document_data)
            logger.info(f"Successfully processed {pdf_path}: {statistics['total_chunks']} chunks created")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            logger.info(f"Total processed documents now: {len(self.processed_documents)}")
            
            return document_data
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {str(e)}", exc_info=True)
            self.processing_errors.append({
                'file': pdf_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            # Record failed processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            try:
                metrics_collector = get_metrics_collector()
                metrics_collector.record_error(
                    session_id=session_id,
                    error_type="processing_error",
                    error_message=str(e),
                    function_name="process_document",
                    stack_trace=str(e.__traceback__) if hasattr(e, '__traceback__') else ""
                )
            except Exception as me:
                logger.warning(f"Failed to record error metrics: {me}")
            
            return None
    
    @with_retry(max_retries=1, base_delay=2.0, retry_exceptions=(OSError, PermissionError))
    def process_document_selective(self, pdf_path: str, page_ranges: str) -> Optional[Dict[str, Any]]:
        """
        Process specific pages of a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            page_ranges: Page range specification (e.g., "1-10, 15, 20-30")
            
        Returns:
            Dictionary containing all processed data, or None if processing failed
        """
        # Generate session ID and start timing
        import uuid
        session_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting selective processing: {pdf_path} (Pages: {page_ranges}, Session: {session_id})")
        error_manager = get_error_manager()
        
        # Validate PDF file
        validation_result = safe_execute(
            validate_pdf_file,
            pdf_path, 
            max_size_mb=self.config.max_file_size_mb,
            default_return=(False, "Validation failed")
        )
        
        is_valid, error_msg = validation_result
        if not is_valid:
            logger.error(f"Validation failed for {pdf_path}: {error_msg}")
            self.processing_errors.append({
                'file': pdf_path,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            return None
        
        try:
            # Parse and validate page ranges
            page_parser = PageRangeParser()
            page_selection = page_parser.create_page_selection(page_ranges, pdf_path)
            
            logger.info(f"Page selection validated: {page_parser.format_ranges_display(page_selection)}")
            
            # Extract metadata
            metadata = safe_execute(
                self.extractor.extract_metadata,
                pdf_path,
                default_return={'file_name': os.path.basename(pdf_path), 'extraction_date': datetime.now().isoformat()}
            )
            
            # Add page selection info to metadata
            metadata['page_selection'] = {
                'ranges': page_ranges,
                'selected_pages': page_selection.selected_count,
                'total_pages': page_selection.total_pages,
                'percentage': page_selection.get_selected_percentage()
            }
            
            # Extract text from selected pages only
            extraction_result = safe_execute(
                self.extractor.extract_text_selective,
                pdf_path, page_selection,
                default_return=("", {})
            )
            
            full_text, page_texts = extraction_result
            
            if not full_text or not safe_execute(self.extractor.validate_extraction, full_text, default_return=False):
                logger.warning(f"No valid text extracted from selected pages in {pdf_path}")
                return self._create_empty_document(metadata)
            
            # Clean text if configured
            cleaned_text = safe_execute(
                clean_text,
                full_text,
                default_return=full_text
            ) if self.config.clean_text else full_text
            
            # Chunk text with fallback
            chunks = safe_execute(
                self.chunker.chunk_by_sections,
                cleaned_text, metadata,
                default_return=[]
            )
            
            if not chunks:
                chunks = safe_execute(
                    self.chunker.chunk_by_sentences,
                    cleaned_text, metadata,
                    default_return=[]
                )
            
            # Calculate statistics
            statistics = safe_execute(
                self._calculate_statistics,
                cleaned_text, chunks,
                default_return={
                    'total_characters': len(cleaned_text),
                    'total_words': len(cleaned_text.split()),
                    'total_chunks': len(chunks),
                    'extraction_success': bool(full_text),
                    'pages_processed': page_selection.selected_count,
                    'selective_processing': True
                }
            )
            
            document_data = {
                'metadata': metadata,
                'full_text': full_text,
                'cleaned_text': cleaned_text,
                'page_texts': page_texts,
                'chunks': chunks,
                'statistics': statistics,
                'page_selection': {
                    'ranges': [(start, end) for start, end in page_selection.ranges],
                    'selected_pages': page_selection.selected_pages,
                    'original_numbering': page_selection.original_numbering
                }
            }
            
            # Apply AI validation and article extraction if enabled
            if self.validation_chain and (self.config.enable_ai_validation or self.config.extract_articles):
                logger.info(f"Running AI validation and article extraction for selected pages")
                
                validated_data = safe_execute(
                    self.validation_chain.validate_document,
                    document_data,
                    default_return=document_data
                )
                
                if validated_data:
                    document_data = validated_data
                
                # Log validation results
                if 'validation_results' in document_data:
                    overall_score = document_data['validation_results'].get('document_validation', {}).get('overall_score', 0)
                    logger.info(f"Document validation score (selective): {overall_score:.1f}/100")
                    
                if 'articles' in document_data:
                    logger.info(f"Extracted {len(document_data['articles'])} articles from selected pages")
            
            # Apply quality validation if enabled
            quality_validation = None
            if self.quality_validator:
                logger.info(f"Running quality validation for selective processing")
                quality_validation = safe_execute(
                    self.quality_validator.validate_document_processing,
                    document_data,
                    default_return=None
                )
            
            if quality_validation:
                document_data['quality_validation'] = {
                    'is_valid': quality_validation.is_valid,
                    'quality_metrics': {
                        'ocr_confidence': quality_validation.quality_metrics.ocr_confidence,
                        'text_completeness': quality_validation.quality_metrics.text_completeness,
                        'structure_coherence': quality_validation.quality_metrics.structure_coherence,
                        'article_extraction_rate': quality_validation.quality_metrics.article_extraction_rate,
                        'regulatory_term_accuracy': quality_validation.quality_metrics.regulatory_term_accuracy,
                        'overall_score': quality_validation.quality_metrics.overall_score,
                        'status': quality_validation.quality_metrics.status.value,
                        'timestamp': quality_validation.quality_metrics.timestamp.isoformat()
                    },
                    'validation_details': quality_validation.validation_details,
                    'recommendations': quality_validation.recommendations
                }
                
                # Log quality validation results
                overall_quality = quality_validation.quality_metrics.overall_score
                status = quality_validation.quality_metrics.status.value
                logger.info(f"Quality validation (selective) - Score: {overall_quality:.2f}, Status: {status}")
            
            # Record processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            try:
                record_processing_session(
                    session_id=session_id,
                    file_name=os.path.basename(pdf_path),
                    processing_result=document_data,
                    processing_time=processing_time
                )
                logger.debug(f"Recorded processing metrics for selective session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to record processing metrics: {e}")
            
            # Add session metadata to document data
            document_data['session_info'] = {
                'session_id': session_id,
                'processing_time': processing_time,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'selective_processing': True,
                'page_ranges': page_ranges
            }
            
            self.processed_documents.append(document_data)
            logger.info(f"Successfully processed selected pages from {pdf_path}: {statistics['total_chunks']} chunks created")
            logger.info(f"Pages processed: {page_selection.selected_count}/{page_selection.total_pages} ({page_selection.get_selected_percentage():.1f}%)")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            
            return document_data
            
        except ValueError as e:
            # Page range parsing errors
            logger.error(f"Invalid page range specification: {e}")
            self.processing_errors.append({
                'file': pdf_path,
                'error': f"Page range error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            })
            
            # Try to provide suggestions
            try:
                parser = PageRangeParser()
                total_pages = parser.get_page_count(pdf_path)
                suggestions = parser.suggest_corrections(page_ranges, total_pages)
                logger.info(f"Suggestions: {'; '.join(suggestions)}")
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path} with selective pages: {str(e)}", exc_info=True)
            self.processing_errors.append({
                'file': pdf_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            # Record failed processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            try:
                metrics_collector = get_metrics_collector()
                metrics_collector.record_error(
                    session_id=session_id,
                    error_type="selective_processing_error",
                    error_message=str(e),
                    function_name="process_document_selective",
                    stack_trace=str(e.__traceback__) if hasattr(e, '__traceback__') else ""
                )
            except Exception as me:
                logger.warning(f"Failed to record error metrics: {me}")
            
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
        
        self.client_excel_exporter.export_client_report(
            self.processed_documents,
            output_path,
            user_info=user_info
        )
        
        if self.processing_errors:
            self._export_errors(output_path)
    
    def export_cartographie_reglementaire(
        self,
        output_path: str,
        user_info: Optional[Dict[str, str]] = None
    ):
        """
        Export regulatory cartography in synthetic mapping format.
        
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
        
        self.cartographie_exporter.export_cartographie_reglementaire(
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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and performance metrics."""
        cache = get_cache()
        return cache.get_stats()
    
    def clear_cache(self, operation_type: Optional[str] = None) -> None:
        """
        Clear processing cache.
        
        Args:
            operation_type: Specific operation type to clear (e.g., 'text_extraction', 'article_extraction')
                          If None, clears all cache
        """
        cache = get_cache()
        cache.clear(operation_type)
        logger.info(f"Cache cleared" + (f" for operation: {operation_type}" if operation_type else " (all)"))
    
    def get_error_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive error summary and recommendations.
        
        Args:
            hours_back: Number of hours to look back for error analysis
            
        Returns:
            Error summary with recommendations
        """
        error_manager = get_error_manager()
        return error_manager.get_error_summary(hours_back)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health report including cache and error metrics."""
        cache_stats = self.get_cache_stats()
        error_summary = self.get_error_summary()
        
        # Calculate health score based on various metrics
        health_score = 100.0
        issues = []
        
        # Check error rates
        total_errors = error_summary.get('total_errors', 0)
        if total_errors > 10:
            health_score -= 20
            issues.append(f"High error count: {total_errors} errors in last 24h")
        elif total_errors > 5:
            health_score -= 10
            issues.append(f"Moderate error count: {total_errors} errors in last 24h")
        
        # Check recovery success rate
        recovery_rates = error_summary.get('recovery_success_rate', {})
        avg_recovery_rate = sum(recovery_rates.values()) / len(recovery_rates) if recovery_rates else 1.0
        if avg_recovery_rate < 0.5:
            health_score -= 15
            issues.append(f"Low recovery success rate: {avg_recovery_rate:.1%}")
        
        # Check cache performance
        cache_hit_rate = cache_stats.get('total_accesses', 0) / max(cache_stats.get('total_disk_entries', 1), 1)
        if cache_hit_rate < 0.3:
            health_score -= 5
            issues.append("Low cache efficiency")
        
        # Check for problematic files
        problematic_files = error_summary.get('most_problematic_files', [])
        if len(problematic_files) > 3:
            health_score -= 10
            issues.append(f"{len(problematic_files)} files with repeated errors")
        
        health_status = "EXCELLENT" if health_score >= 90 else \
                       "GOOD" if health_score >= 75 else \
                       "FAIR" if health_score >= 60 else \
                       "POOR"
        
        return {
            'health_score': health_score,
            'health_status': health_status,
            'issues': issues,
            'cache_stats': cache_stats,
            'error_summary': error_summary,
            'recommendations': error_summary.get('recommendations', []),
            'last_updated': datetime.now().isoformat()
        }
    
    def optimize_performance(self) -> Dict[str, str]:
        """
        Perform automatic performance optimizations.
        
        Returns:
            Dictionary of optimization actions taken
        """
        actions = {}
        
        # Clean expired cache entries
        try:
            cache = get_cache()
            cache._cleanup_expired()
            actions['cache_cleanup'] = "Cleaned expired cache entries"
        except Exception as e:
            actions['cache_cleanup'] = f"Failed: {e}"
        
        # Optimize memory cache
        try:
            cache = get_cache()
            if len(cache.memory_cache) > cache.max_memory_size * 0.8:
                cache._evict_memory_cache()
                actions['memory_optimization'] = "Evicted old memory cache entries"
            else:
                actions['memory_optimization'] = "Memory cache within optimal range"
        except Exception as e:
            actions['memory_optimization'] = f"Failed: {e}"
        
        # Reset processing errors if they're too old
        current_time = datetime.now()
        old_errors = [
            err for err in self.processing_errors
            if (current_time - datetime.fromisoformat(err['timestamp'])).total_seconds() > 86400  # 24 hours
        ]
        
        if old_errors:
            self.processing_errors = [
                err for err in self.processing_errors
                if err not in old_errors
            ]
            actions['error_cleanup'] = f"Removed {len(old_errors)} old error records"
        else:
            actions['error_cleanup'] = "No old errors to clean up"
        
        logger.info(f"Performance optimization completed: {len(actions)} actions taken")
        return actions
    
    def get_performance_metrics(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics and analysis.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with performance metrics and recommendations
        """
        try:
            metrics_collector = get_metrics_collector()
            return metrics_collector.get_performance_report(days_back)
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def export_metrics(self, file_path: str, days_back: int = 30):
        """
        Export metrics to a file for analysis.
        
        Args:
            file_path: Path to export file
            days_back: Number of days of data to export
        """
        try:
            metrics_collector = get_metrics_collector()
            metrics_collector.export_metrics(file_path, days_back)
            logger.info(f"Metrics exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def record_user_feedback(self, session_id: str, rating: int, 
                           comments: str = "", suggestions: str = ""):
        """
        Record user feedback for a processing session.
        
        Args:
            session_id: Processing session identifier
            rating: User rating (1-5)
            comments: User comments
            suggestions: User suggestions for improvement
        """
        try:
            metrics_collector = get_metrics_collector()
            metrics_collector.record_user_feedback(
                session_id=session_id,
                feedback_type="general",
                rating=rating,
                comments=comments,
                suggestions=suggestions
            )
            logger.info(f"User feedback recorded for session {session_id}: {rating}/5")
        except Exception as e:
            logger.error(f"Failed to record user feedback: {e}")
    
    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of system metrics for quick overview.
        
        Returns:
            Dictionary with key system metrics
        """
        try:
            metrics_collector = get_metrics_collector()
            system_metrics = metrics_collector.get_system_metrics(days_back=7)
            
            return {
                'last_7_days': {
                    'documents_processed': system_metrics.total_documents,
                    'success_rate': (system_metrics.successful_processing / 
                                   max(system_metrics.total_documents, 1)) * 100,
                    'average_quality_score': system_metrics.average_quality_score * 100,
                    'average_processing_time': system_metrics.average_processing_time,
                    'user_satisfaction': system_metrics.user_satisfaction,
                    'common_issues': len(system_metrics.common_errors)
                },
                'trends': {
                    'quality_trend': system_metrics.quality_trends.get('quality_score_change', 0),
                    'performance_trend': system_metrics.performance_trends.get('processing_time_change', 0)
                },
                'status': 'healthy' if system_metrics.average_quality_score > 0.7 else 'needs_attention',
                'generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics summary: {e}")
            return {
                'error': str(e),
                'status': 'unknown',
                'generated_at': datetime.now().isoformat()
            }
    
    def _extract_articles_basic(self, text: str) -> List[Dict[str, Any]]:
        """Basic article extraction using regex patterns."""
        import re
        articles = []
        
        # Common article patterns
        patterns = [
            r'Article\s+(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?(?:\s*(?:bis|ter|quater))?)\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!Article\s+\d+)[\s\S]){20,5000})',
            r'Art\.\s*(\d+)\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!Art\.\s*\d+)[\s\S]){20,3000})',
            r'ARTICLE\s+(\d+(?:[\.,]\d+)?)\s*\n\s*([^\n]{0,200})?\n?((?:(?!ARTICLE\s+\d+)[\s\S]){20,5000})',
        ]
        
        seen_numbers = set()
        
        for pattern in patterns:
            try:
                matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    article_number = match.group(1).strip()
                    if article_number in seen_numbers:
                        continue
                    seen_numbers.add(article_number)
                    
                    title = match.group(2).strip() if match.lastindex >= 2 and match.group(2) else ""
                    content = match.group(3).strip() if match.lastindex >= 3 and match.group(3) else match.group(0)
                    
                    # Clean content
                    content = re.sub(r'\s+', ' ', content).strip()
                    if len(content) > 50:  # Only include substantial articles
                        articles.append({
                            'number': f"Article {article_number}",
                            'title': title[:200] if title else f"Article {article_number}",
                            'content': content[:5000],  # Limit content length
                            'materiality': {'level': 'MEDIUM', 'reasoning': 'Fallback extraction'},
                            'start_position': match.start(),
                            'end_position': match.end()
                        })
            except Exception as e:
                logger.warning(f"Pattern extraction failed: {e}")
                continue
        
        # Sort articles by number
        def get_article_sort_key(article):
            num_str = article['number'].replace('Article', '').strip()
            try:
                # Handle formats like "12", "12.1", "12-1", "12 bis"
                match = re.match(r'(\d+)(?:[.-](\d+))?', num_str)
                if match:
                    main = int(match.group(1))
                    sub = int(match.group(2)) if match.group(2) else 0
                    return (main, sub)
                return (9999, 0)
            except:
                return (9999, 0)
        
        articles.sort(key=get_article_sort_key)
        
        logger.info(f"Basic extraction found {len(articles)} articles")
        return articles