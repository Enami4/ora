"""
Integration tests for component interactions.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from regulatory_processor.processor import RegulatoryDocumentProcessor
from regulatory_processor.config import ProcessorConfig
from regulatory_processor.extractor import PDFTextExtractor
from regulatory_processor.chunker import TextChunker
from regulatory_processor.validators import AIValidator, ValidationScore
from regulatory_processor.exporters import ExcelExporter


@pytest.mark.integration
class TestComponentIntegration:
    """Test integration between different processor components."""
    
    def test_extractor_chunker_integration(self, tmp_path):
        """Test integration between text extractor and chunker."""
        config = ProcessorConfig(chunk_size=200, chunk_overlap=50)
        
        extractor = PDFTextExtractor(config)
        chunker = TextChunker(config)
        
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b'Test PDF content')
        
        # Mock PDF extraction
        sample_text = """
        This is a long regulation document that will be split into multiple chunks.
        Article 1. First article content with detailed explanations and requirements.
        Article 2. Second article with additional provisions and compliance details.
        Article 3. Final article with implementation guidelines and enforcement measures.
        """
        
        with patch.object(extractor, 'extract_text') as mock_extract:
            mock_extract.return_value = sample_text
            
            # Extract text
            extracted_text = extractor.extract_text(test_file)
            
            # Chunk the extracted text
            chunks = chunker.chunk_text(extracted_text, document_id="test_doc")
            
            # Verify integration
            assert extracted_text == sample_text
            assert len(chunks) > 1  # Should be split into multiple chunks
            
            # Verify chunk structure
            for chunk in chunks:
                assert 'text' in chunk
                assert 'chunk_index' in chunk
                assert 'chunk_id' in chunk
                assert len(chunk['text']) <= config.chunk_size
                assert 'test_doc' in chunk['chunk_id']
    
    def test_chunker_validator_integration(self, ai_config, sample_text):
        """Test integration between chunker and AI validator."""
        chunker = TextChunker(ai_config)
        validator = AIValidator(ai_config)
        
        # Create chunks
        chunks = chunker.chunk_text(sample_text, document_id="validation_test")
        
        # Mock AI validation
        mock_score = ValidationScore(
            completeness_score=85,
            reliability_score=90,
            legal_structure_score=80,
            overall_score=85
        )
        
        with patch.object(validator, 'validate_chunk') as mock_validate:
            mock_validate.return_value = mock_score
            
            # Validate chunks
            validation_results = []
            for chunk in chunks:
                score = validator.validate_chunk(chunk)
                validation_results.append(score)
            
            # Verify integration
            assert len(validation_results) == len(chunks)
            assert all(isinstance(score, ValidationScore) for score in validation_results)
            assert mock_validate.call_count == len(chunks)
            
            # Verify validator received proper chunk structure
            for call in mock_validate.call_args_list:
                chunk_arg = call[0][0]
                assert 'text' in chunk_arg
                assert 'chunk_id' in chunk_arg
    
    def test_processor_exporter_integration(self, tmp_path, user_info):
        """Test integration between processor and Excel exporter."""
        config = ProcessorConfig(enable_ai_validation=False)
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        exporter = ExcelExporter(config)
        
        # Create test processing results
        test_file = tmp_path / "integration_test.pdf"
        test_file.write_bytes(b'Test content')
        
        # Mock processing result
        sample_result = {
            'file_path': test_file,
            'text': 'Processed regulation text content',
            'chunks': [
                {
                    'chunk_index': 0,
                    'text': 'Article 1. Banking regulations',
                    'text_length': 30,
                    'word_count': 4,
                    'chunk_id': 'test_chunk_0'
                },
                {
                    'chunk_index': 1,
                    'text': 'Article 2. Capital requirements',
                    'text_length': 32,
                    'word_count': 4,
                    'chunk_id': 'test_chunk_1'
                }
            ],
            'articles': [],
            'validation_scores': [],
            'metadata': {'file_size': 1024, 'processing_time': 1.5}
        }
        
        output_file = tmp_path / "integration_export.xlsx"
        
        # Mock the export process
        with patch.object(exporter, 'export_complete_analysis') as mock_export:
            mock_export.return_value = output_file
            
            # Use processor's export method
            result_file = processor.export_to_excel(
                results=[sample_result],
                output_file=output_file,
                user_info=user_info
            )
            
            # Verify integration
            assert result_file == output_file
            mock_export.assert_called_once()
            
            # Verify export parameters
            call_args = mock_export.call_args
            export_kwargs = call_args[1]
            
            assert 'chunks' in export_kwargs
            assert 'user_info' in export_kwargs
            assert export_kwargs['user_info'] == user_info
            assert len(export_kwargs['chunks']) == 2
    
    def test_ai_components_integration(self, tmp_path):
        """Test integration between AI validation and article extraction."""
        config = ProcessorConfig(
            enable_ai_validation=True,
            extract_articles=True,
            assess_materiality=True,
            anthropic_api_key="test-key"
        )
        
        validator = AIValidator(config)
        
        sample_text = """
        REGLEMENT BANCAIRE N°2024-01
        
        Article 1. Capital minimum
        Les banques doivent maintenir un capital minimum de 8%.
        
        Article 2. Liquidité
        Le ratio de liquidité ne peut être inférieur à 25%.
        """
        
        # Mock AI responses
        with patch.object(validator, '_call_anthropic_api') as mock_api:
            # Mock article extraction response
            mock_api.return_value = {
                'articles': [
                    {
                        'number': 'Article 1',
                        'title': 'Capital minimum',
                        'content': 'Les banques doivent maintenir...',
                        'materiality': 'HIGH',
                        'reasoning': 'Defines minimum capital requirements'
                    },
                    {
                        'number': 'Article 2',
                        'title': 'Liquidité',
                        'content': 'Le ratio de liquidité...',
                        'materiality': 'MEDIUM',
                        'reasoning': 'Sets liquidity ratio requirements'
                    }
                ]
            }
            
            # Extract articles
            articles = validator.extract_articles(sample_text)
            
            # Verify integration
            assert len(articles) == 2
            assert articles[0].number == 'Article 1'
            assert articles[0].title == 'Capital minimum'
            assert articles[1].number == 'Article 2'
            assert articles[1].title == 'Liquidité'
            
            # Test materiality assessment integration
            for article in articles:
                with patch.object(validator, 'assess_materiality') as mock_materiality:
                    mock_materiality.return_value = (article.materiality, article.materiality_reasoning)
                    
                    materiality, reasoning = validator.assess_materiality(article)
                    
                    assert materiality == article.materiality
                    assert reasoning == article.materiality_reasoning
    
    def test_configuration_propagation(self):
        """Test configuration propagation across components."""
        custom_config = ProcessorConfig(
            chunk_size=800,
            chunk_overlap=150,
            max_file_size_mb=50,
            enable_ai_validation=True,
            anthropic_api_key="test-key-123",
            ai_model="claude-3-sonnet-20240229",
            clean_text=False,
            extract_articles=True
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(custom_config)
        
        # Verify configuration propagation to components
        assert processor.extractor.config == custom_config
        assert processor.chunker.config == custom_config
        assert processor.exporter.config == custom_config
        
        if processor.validator:
            assert processor.validator.config == custom_config
            assert processor.validator.api_key == custom_config.anthropic_api_key
            assert processor.validator.model == custom_config.ai_model
        
        # Verify specific settings
        assert processor.extractor.max_file_size_mb == 50
        assert processor.chunker.chunk_size == 800
        assert processor.chunker.chunk_overlap == 150
    
    def test_error_handling_across_components(self, tmp_path):
        """Test error handling across component boundaries."""
        config = ProcessorConfig(enable_ai_validation=True, anthropic_api_key="test-key")
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        test_file = tmp_path / "error_test.pdf"
        test_file.write_bytes(b'Test content')
        
        # Test error propagation from extractor
        with patch.object(processor.extractor, 'extract_text') as mock_extract:
            mock_extract.side_effect = Exception("Extraction failed")
            
            with pytest.raises(Exception, match="Extraction failed"):
                processor.process_file(test_file)
        
        # Test error handling in chunker
        with patch.object(processor.extractor, 'extract_text') as mock_extract, \
             patch.object(processor.chunker, 'chunk_text') as mock_chunk:
            
            mock_extract.return_value = "Valid text"
            mock_chunk.side_effect = Exception("Chunking failed")
            
            with pytest.raises(Exception, match="Chunking failed"):
                processor.process_file(test_file)
        
        # Test error handling in validator (should not crash processor)
        with patch.object(processor.extractor, 'extract_text') as mock_extract, \
             patch.object(processor.validator, 'validate_chunk') as mock_validate:
            
            mock_extract.return_value = "Valid text"
            mock_validate.side_effect = Exception("Validation failed")
            
            # Should still complete processing despite validation error
            result = processor.process_file(test_file)
            
            # Should have basic results even if validation failed
            assert 'text' in result
            assert 'chunks' in result
    
    def test_data_flow_integrity(self, tmp_path):
        """Test data integrity throughout the processing pipeline."""
        config = ProcessorConfig(
            chunk_size=300,
            chunk_overlap=50,
            enable_ai_validation=False
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        test_file = tmp_path / "data_flow_test.pdf"
        test_file.write_bytes(b'Test content')
        
        original_text = "Article 1. Test content for data flow integrity. " * 10
        
        with patch.object(processor.extractor, 'extract_text') as mock_extract:
            mock_extract.return_value = original_text
            
            result = processor.process_file(test_file)
            
            # Verify data integrity
            assert result['text'] == original_text
            
            # Verify all chunks contain parts of original text
            all_chunk_text = ' '.join(chunk['text'] for chunk in result['chunks'])
            
            # Should contain most of the original text (accounting for overlap)
            assert len(all_chunk_text) >= len(original_text)
            
            # Key phrases should be preserved
            assert 'Article 1' in all_chunk_text
            assert 'Test content' in all_chunk_text
            assert 'data flow integrity' in all_chunk_text
    
    def test_performance_across_components(self, tmp_path):
        """Test performance characteristics across component integration."""
        config = ProcessorConfig(
            chunk_size=500,
            chunk_overlap=100,
            enable_ai_validation=False
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Create multiple test files
        test_files = []
        for i in range(5):
            file = tmp_path / f"perf_test_{i}.pdf"
            file.write_bytes(f'Performance test content {i}'.encode() * 100)
            test_files.append(file)
        
        # Mock extraction to return substantial text
        def mock_extract_large(file_path):
            return f"Large regulation text for {file_path.name}. " * 200
        
        with patch.object(processor.extractor, 'extract_text') as mock_extract:
            mock_extract.side_effect = mock_extract_large
            
            import time
            start_time = time.time()
            
            # Process all files
            results = processor.process_multiple_files(test_files)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify performance
            assert len(results) == 5
            assert processing_time < 10.0  # Should complete within 10 seconds
            
            # Verify all results have proper structure
            for result in results:
                assert 'chunks' in result
                assert len(result['chunks']) > 0
    
    def test_concurrent_component_usage(self, tmp_path):
        """Test concurrent usage of components."""
        config = ProcessorConfig(enable_ai_validation=False)
        
        # Create multiple processor instances (simulating concurrent usage)
        processors = []
        for i in range(3):
            with patch('regulatory_processor.processor.setup_logging'):
                processor = RegulatoryDocumentProcessor(config)
            processors.append(processor)
        
        # Create test files
        test_files = []
        for i in range(3):
            file = tmp_path / f"concurrent_test_{i}.pdf"
            file.write_bytes(f'Concurrent test content {i}'.encode())
            test_files.append(file)
        
        # Mock extraction for each processor
        results = []
        for i, processor in enumerate(processors):
            with patch.object(processor.extractor, 'extract_text') as mock_extract:
                mock_extract.return_value = f"Text from processor {i}"
                
                result = processor.process_file(test_files[i])
                results.append(result)
        
        # Verify concurrent processing worked correctly
        assert len(results) == 3
        for i, result in enumerate(results):
            assert f"Text from processor {i}" in result['text']
            assert result['file_path'] == test_files[i]
    
    def test_memory_management_integration(self, tmp_path):
        """Test memory management across component integration."""
        config = ProcessorConfig(
            chunk_size=1000,
            chunk_overlap=200,
            enable_ai_validation=False
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Create large file
        large_file = tmp_path / "memory_test.pdf"
        large_file.write_bytes(b'Large content ' * 50000)  # ~700KB
        
        # Simulate large text extraction
        large_text = "Regulation content. " * 5000  # ~100KB of text
        
        with patch.object(processor.extractor, 'extract_text') as mock_extract:
            mock_extract.return_value = large_text
            
            # Process large file
            result = processor.process_file(large_file)
            
            # Verify memory-efficient processing
            assert 'chunks' in result
            assert len(result['chunks']) > 0
            
            # Verify chunks are properly sized
            for chunk in result['chunks']:
                assert len(chunk['text']) <= config.chunk_size
            
            # Verify text is preserved
            assert result['text'] == large_text