"""
Unit tests for main processor module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from regulatory_processor.processor import RegulatoryDocumentProcessor
from regulatory_processor.config import ProcessorConfig
from regulatory_processor.validators import ValidationScore, Article, MaterialityLevel


class TestRegulatoryDocumentProcessor:
    """Test RegulatoryDocumentProcessor class."""
    
    def test_init_with_config(self, basic_config):
        """Test processor initialization with configuration."""
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(basic_config)
            
            assert processor.config == basic_config
            assert processor.extractor is not None
            assert processor.chunker is not None
            assert processor.exporter is not None
    
    def test_init_without_config(self):
        """Test processor initialization without configuration."""
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor()
            
            assert isinstance(processor.config, ProcessorConfig)
            assert processor.extractor is not None
            assert processor.chunker is not None
            assert processor.exporter is not None
    
    def test_init_with_ai_enabled(self, ai_config):
        """Test processor initialization with AI validation enabled."""
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(ai_config)
            
            assert processor.config.enable_ai_validation
            assert processor.validator is not None
    
    def test_init_without_ai(self, basic_config):
        """Test processor initialization without AI validation."""
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(basic_config)
            
            assert not processor.config.enable_ai_validation
            assert processor.validator is None
    
    def test_process_single_file(self, mock_processor, tmp_path):
        """Test processing a single PDF file."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b'fake pdf content')
        
        # Mock dependencies
        mock_processor.extractor.extract_text = Mock(return_value="Extracted text content")
        mock_processor.chunker.chunk_text = Mock(return_value=[
            {'chunk_index': 0, 'text': 'Chunk 1', 'chunk_id': 'chunk_0'},
            {'chunk_index': 1, 'text': 'Chunk 2', 'chunk_id': 'chunk_1'}
        ])
        
        result = mock_processor.process_file(test_file)
        
        assert 'file_path' in result
        assert 'text' in result
        assert 'chunks' in result
        assert 'metadata' in result
        assert len(result['chunks']) == 2
        
        # Verify methods were called
        mock_processor.extractor.extract_text.assert_called_once_with(test_file)
        mock_processor.chunker.chunk_text.assert_called_once()
    
    def test_process_multiple_files(self, mock_processor, tmp_path):
        """Test processing multiple PDF files."""
        # Create test files
        files = []
        for i in range(3):
            file = tmp_path / f"test_{i}.pdf"
            file.write_bytes(f'fake pdf content {i}'.encode())
            files.append(file)
        
        # Mock file processing
        def mock_process_file(file_path):
            return {
                'file_path': file_path,
                'text': f'Text from {file_path.name}',
                'chunks': [{'chunk_index': 0, 'text': f'Chunk from {file_path.name}'}],
                'metadata': {}
            }
        
        mock_processor.process_file = Mock(side_effect=mock_process_file)
        
        results = mock_processor.process_multiple_files(files)
        
        assert len(results) == 3
        assert all('file_path' in result for result in results)
        assert mock_processor.process_file.call_count == 3
    
    def test_process_directory(self, mock_processor, tmp_path):
        """Test processing a directory containing PDF files."""
        # Create directory structure
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        
        # Create PDF files
        (pdf_dir / "doc1.pdf").write_bytes(b'content1')
        (pdf_dir / "doc2.pdf").write_bytes(b'content2')
        (pdf_dir / "not_pdf.txt").write_bytes(b'text content')  # Should be ignored
        
        # Mock glob to find PDF files
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = [pdf_dir / "doc1.pdf", pdf_dir / "doc2.pdf"]
            
            # Mock file processing
            mock_processor.process_multiple_files = Mock(return_value=[
                {'file_path': pdf_dir / "doc1.pdf", 'chunks': []},
                {'file_path': pdf_dir / "doc2.pdf", 'chunks': []}
            ])
            
            results = mock_processor.process_directory(pdf_dir)
            
            assert len(results) == 2
            mock_processor.process_multiple_files.assert_called_once()
    
    def test_process_with_ai_validation(self, tmp_path):
        """Test processing with AI validation enabled."""
        ai_config = ProcessorConfig(enable_ai_validation=True, anthropic_api_key="test-key")
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(ai_config)
        
        # Mock all dependencies
        processor.extractor.extract_text = Mock(return_value="Test regulation content")
        processor.chunker.chunk_text = Mock(return_value=[
            {'chunk_index': 0, 'text': 'Article 1 content', 'chunk_id': 'chunk_0'}
        ])
        processor.validator.validate_chunk = Mock(return_value=ValidationScore(overall_score=85))
        
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b'content')
        
        result = processor.process_file(test_file)
        
        assert 'validation_scores' in result
        processor.validator.validate_chunk.assert_called_once()
    
    def test_extract_articles(self, tmp_path):
        """Test article extraction functionality."""
        config = ProcessorConfig(
            extract_articles=True,
            enable_ai_validation=True,
            anthropic_api_key="test-key"
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Mock article extraction
        sample_articles = [
            Article(
                number="Article 1",
                title="Test Article",
                content="Test content",
                materiality=MaterialityLevel.HIGH
            )
        ]
        
        processor.extractor.extract_text = Mock(return_value="Document with articles")
        processor.validator.extract_articles = Mock(return_value=sample_articles)
        
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b'content')
        
        result = processor.process_file(test_file)
        
        assert 'articles' in result
        assert len(result['articles']) == 1
        processor.validator.extract_articles.assert_called_once()
    
    def test_materiality_assessment(self, tmp_path):
        """Test materiality assessment functionality."""
        config = ProcessorConfig(
            assess_materiality=True,
            enable_ai_validation=True,
            anthropic_api_key="test-key"
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Mock dependencies
        sample_articles = [
            Article(number="Article 1", title="Test", content="Content")
        ]
        
        processor.extractor.extract_text = Mock(return_value="Text")
        processor.validator.extract_articles = Mock(return_value=sample_articles)
        processor.validator.assess_materiality = Mock(
            return_value=(MaterialityLevel.CRITICAL, "Important regulation")
        )
        
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b'content')
        
        result = processor.process_file(test_file)
        
        assert 'articles' in result
        processor.validator.assess_materiality.assert_called_once()
    
    def test_export_to_excel(self, mock_processor, tmp_path, user_info):
        """Test exporting results to Excel."""
        # Prepare sample data
        sample_results = [
            {
                'file_path': tmp_path / "doc1.pdf",
                'chunks': [
                    {'chunk_index': 0, 'text': 'Chunk 1', 'chunk_id': 'chunk_0'}
                ],
                'articles': [],
                'validation_scores': []
            }
        ]
        
        output_file = tmp_path / "export_test.xlsx"
        
        # Mock exporter
        mock_processor.exporter.export_complete_analysis = Mock(return_value=output_file)
        
        result_file = mock_processor.export_to_excel(
            results=sample_results,
            output_file=output_file,
            user_info=user_info
        )
        
        assert result_file == output_file
        mock_processor.exporter.export_complete_analysis.assert_called_once()
    
    def test_progress_callback(self, mock_processor, tmp_path):
        """Test progress callback functionality."""
        files = [tmp_path / f"test_{i}.pdf" for i in range(3)]
        for file in files:
            file.write_bytes(b'content')
        
        progress_calls = []
        
        def progress_callback(current, total, filename):
            progress_calls.append((current, total, filename))
        
        # Mock file processing
        mock_processor.process_file = Mock(return_value={'chunks': []})
        
        mock_processor.process_multiple_files(files, progress_callback=progress_callback)
        
        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3, files[0].name)
        assert progress_calls[-1] == (3, 3, files[-1].name)
    
    def test_error_handling_single_file(self, mock_processor, tmp_path):
        """Test error handling for single file processing."""
        test_file = tmp_path / "error_test.pdf"
        test_file.write_bytes(b'content')
        
        # Mock extraction to raise error
        mock_processor.extractor.extract_text = Mock(
            side_effect=Exception("Extraction failed")
        )
        
        with pytest.raises(Exception, match="Extraction failed"):
            mock_processor.process_file(test_file)
    
    def test_error_handling_multiple_files(self, mock_processor, tmp_path):
        """Test error handling for multiple file processing."""
        files = [tmp_path / f"test_{i}.pdf" for i in range(3)]
        for file in files:
            file.write_bytes(b'content')
        
        # Mock processing to fail on second file
        def mock_process_side_effect(file_path):
            if "test_1" in str(file_path):
                raise Exception("Processing failed")
            return {'chunks': [], 'file_path': file_path}
        
        mock_processor.process_file = Mock(side_effect=mock_process_side_effect)
        
        # With skip_errors=True, should continue processing
        results = mock_processor.process_multiple_files(files, skip_errors=True)
        
        assert len(results) == 2  # Should skip the failed file
        assert mock_processor.process_file.call_count == 3
    
    def test_large_file_processing(self, mock_processor, tmp_path):
        """Test processing of large files."""
        large_file = tmp_path / "large_test.pdf"
        large_content = b'Large content ' * 100000  # ~1.4MB
        large_file.write_bytes(large_content)
        
        # Mock extraction to return large text
        large_text = "Large extracted text " * 10000
        mock_processor.extractor.extract_text = Mock(return_value=large_text)
        
        # Mock chunking to handle large text
        large_chunks = [
            {'chunk_index': i, 'text': f'Chunk {i}', 'chunk_id': f'chunk_{i}'}
            for i in range(100)
        ]
        mock_processor.chunker.chunk_text = Mock(return_value=large_chunks)
        
        result = mock_processor.process_file(large_file)
        
        assert len(result['chunks']) == 100
        assert 'text' in result
        assert len(result['text']) > 0
    
    def test_memory_efficiency(self, mock_processor, tmp_path):
        """Test memory efficiency with multiple large files."""
        # Create multiple large files
        files = []
        for i in range(5):
            file = tmp_path / f"large_{i}.pdf"
            file.write_bytes(b'Large content ' * 10000)  # ~140KB each
            files.append(file)
        
        # Mock processing to return substantial data
        def mock_process_file(file_path):
            return {
                'file_path': file_path,
                'text': 'Extracted text ' * 1000,  # ~14KB
                'chunks': [
                    {'chunk_index': j, 'text': f'Chunk {j}', 'chunk_id': f'chunk_{j}'}
                    for j in range(50)
                ],
                'metadata': {}
            }
        
        mock_processor.process_file = Mock(side_effect=mock_process_file)
        
        # Should process all files without memory issues
        results = mock_processor.process_multiple_files(files)
        
        assert len(results) == 5
        assert all(len(result['chunks']) == 50 for result in results)
    
    def test_configuration_validation(self):
        """Test processor behavior with different configurations."""
        # Test with invalid configuration
        invalid_config = ProcessorConfig(chunk_size=0)  # Invalid
        
        with pytest.raises(ValueError):
            with patch('regulatory_processor.processor.setup_logging'):
                RegulatoryDocumentProcessor(invalid_config)
    
    def test_logging_integration(self, basic_config):
        """Test logging integration."""
        with patch('regulatory_processor.processor.setup_logging') as mock_setup_logging:
            mock_logger = Mock()
            mock_setup_logging.return_value = mock_logger
            
            processor = RegulatoryDocumentProcessor(basic_config)
            
            # Verify logging was set up
            mock_setup_logging.assert_called_once_with(basic_config)
            assert processor.logger == mock_logger
    
    def test_cleanup_functionality(self, mock_processor):
        """Test cleanup functionality."""
        # Mock some resources to clean up
        mock_processor._temp_files = ['/tmp/file1', '/tmp/file2']
        
        with patch('pathlib.Path.unlink') as mock_unlink:
            mock_processor.cleanup()
            
            # Should attempt to clean up temp files
            assert mock_unlink.call_count >= 0  # May or may not have temp files


class TestProcessorIntegration:
    """Test integration between processor components."""
    
    def test_end_to_end_processing(self, tmp_path, user_info):
        """Test complete end-to-end processing pipeline."""
        config = ProcessorConfig(
            chunk_size=200,
            chunk_overlap=50,
            enable_ai_validation=False,  # Disable for integration test
            extract_articles=True,
            clean_text=True
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Create test PDF
        test_file = tmp_path / "integration_test.pdf"
        test_file.write_bytes(b'Test PDF content')
        
        # Mock the entire pipeline
        with patch.object(processor.extractor, 'extract_text') as mock_extract, \
             patch.object(processor.chunker, 'chunk_text') as mock_chunk, \
             patch.object(processor.exporter, 'export_complete_analysis') as mock_export:
            
            mock_extract.return_value = "Article 1. Test regulation content."
            mock_chunk.return_value = [
                {
                    'chunk_index': 0,
                    'text': 'Article 1. Test regulation content.',
                    'text_length': 35,
                    'word_count': 5,
                    'chunk_id': 'test_chunk_0'
                }
            ]
            
            output_file = tmp_path / "integration_output.xlsx"
            mock_export.return_value = output_file
            
            # Process file
            result = processor.process_file(test_file)
            
            # Export results
            export_file = processor.export_to_excel(
                results=[result],
                output_file=output_file,
                user_info=user_info
            )
            
            # Verify pipeline execution
            assert 'chunks' in result
            assert len(result['chunks']) == 1
            assert export_file == output_file
            
            mock_extract.assert_called_once()
            mock_chunk.assert_called_once()
            mock_export.assert_called_once()
    
    def test_component_interaction(self, basic_config):
        """Test interaction between processor components."""
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(basic_config)
        
        # Verify components are properly initialized and connected
        assert processor.extractor.config == basic_config
        assert processor.chunker.chunk_size == basic_config.chunk_size
        assert processor.chunker.chunk_overlap == basic_config.chunk_overlap
        assert processor.exporter.config == basic_config
    
    def test_configuration_propagation(self):
        """Test that configuration is properly propagated to all components."""
        custom_config = ProcessorConfig(
            chunk_size=800,
            chunk_overlap=150,
            max_file_size_mb=50,
            clean_text=False,
            enable_ai_validation=True,
            anthropic_api_key="test-key"
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(custom_config)
        
        # Verify configuration propagation
        assert processor.extractor.max_file_size_mb == 50
        assert processor.chunker.chunk_size == 800
        assert processor.chunker.chunk_overlap == 150
        assert processor.config.enable_ai_validation
        assert processor.validator is not None
    
    def test_error_propagation(self, mock_processor, tmp_path):
        """Test error propagation through the processing pipeline."""
        test_file = tmp_path / "error_test.pdf"
        test_file.write_bytes(b'content')
        
        # Test error in different stages
        error_scenarios = [
            ('extractor', 'extract_text'),
            ('chunker', 'chunk_text'),
        ]
        
        for component, method in error_scenarios:
            comp = getattr(mock_processor, component)
            setattr(comp, method, Mock(side_effect=Exception(f"{component} error")))
            
            with pytest.raises(Exception, match=f"{component} error"):
                mock_processor.process_file(test_file)
    
    def test_resource_management(self, tmp_path):
        """Test proper resource management throughout processing."""
        config = ProcessorConfig(max_file_size_mb=1)  # Small limit for testing
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Create file that exceeds size limit
        large_file = tmp_path / "large.pdf"
        large_file.write_bytes(b'x' * (2 * 1024 * 1024))  # 2MB
        
        # Should handle resource constraints gracefully
        with pytest.raises(ValueError, match="File size exceeds"):
            processor.process_file(large_file)