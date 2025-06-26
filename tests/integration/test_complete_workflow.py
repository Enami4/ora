"""
Integration tests for complete document processing workflows.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from regulatory_processor.processor import RegulatoryDocumentProcessor
from regulatory_processor.config import ProcessorConfig
from regulatory_processor.validators import MaterialityLevel, ValidationScore


@pytest.mark.integration
class TestCompleteWorkflow:
    """Test complete document processing workflows."""
    
    def test_single_file_processing_workflow(self, tmp_path, user_info):
        """Test complete workflow for processing a single PDF file."""
        # Setup configuration
        config = ProcessorConfig(
            chunk_size=500,
            chunk_overlap=100,
            enable_ai_validation=False,  # Disable for integration test
            extract_articles=True,
            clean_text=True
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Create test file
        test_file = tmp_path / "regulation_test.pdf"
        test_file.write_bytes(b'Mock PDF content')
        
        # Mock the PDF extraction
        sample_text = """
        REGLEMENT N°R-2024-01
        
        Article 1. Objet et champ d'application
        Le présent règlement définit les modalités de calcul des ratios prudentiels.
        
        Article 2. Définitions
        Au sens du présent règlement, on entend par établissement de crédit...
        
        Article 3. Obligations
        Les établissements doivent respecter les ratios minimums.
        """
        
        with patch.object(processor.extractor, 'extract_text') as mock_extract:
            mock_extract.return_value = sample_text
            
            # Process the file
            result = processor.process_file(test_file)
            
            # Verify result structure
            assert 'file_path' in result
            assert 'text' in result
            assert 'chunks' in result
            assert 'metadata' in result
            assert result['file_path'] == test_file
            assert result['text'] == sample_text
            assert len(result['chunks']) > 0
            
            # Verify chunks contain expected data
            for chunk in result['chunks']:
                assert 'chunk_index' in chunk
                assert 'text' in chunk
                assert 'text_length' in chunk
                assert 'word_count' in chunk
                assert 'chunk_id' in chunk
    
    def test_multiple_files_processing_workflow(self, tmp_path, user_info):
        """Test complete workflow for processing multiple PDF files."""
        config = ProcessorConfig(
            chunk_size=300,
            chunk_overlap=50,
            enable_ai_validation=False
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Create multiple test files
        test_files = []
        sample_texts = [
            "Regulation 1: Banking requirements...",
            "Regulation 2: Capital adequacy ratios...",
            "Regulation 3: Risk management procedures..."
        ]
        
        for i, text in enumerate(sample_texts):
            file = tmp_path / f"regulation_{i+1}.pdf"
            file.write_bytes(f'PDF content {i+1}'.encode())
            test_files.append(file)
        
        # Mock extraction for each file
        with patch.object(processor.extractor, 'extract_text') as mock_extract:
            mock_extract.side_effect = sample_texts
            
            # Process multiple files
            results = processor.process_multiple_files(test_files)
            
            # Verify results
            assert len(results) == 3
            
            for i, result in enumerate(results):
                assert result['file_path'] == test_files[i]
                assert result['text'] == sample_texts[i]
                assert len(result['chunks']) > 0
    
    def test_directory_processing_workflow(self, tmp_path):
        """Test complete workflow for processing a directory of PDF files."""
        config = ProcessorConfig(enable_ai_validation=False)
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Create directory structure
        pdf_dir = tmp_path / "regulations"
        pdf_dir.mkdir()
        
        # Create PDF files
        pdf_files = []
        for i in range(3):
            file = pdf_dir / f"reg_{i}.pdf"
            file.write_bytes(f'Regulation {i} content'.encode())
            pdf_files.append(file)
        
        # Create non-PDF file (should be ignored)
        (pdf_dir / "readme.txt").write_text("Not a PDF")
        
        # Mock file discovery and processing
        with patch.object(processor, 'process_multiple_files') as mock_process:
            mock_process.return_value = [
                {'file_path': file, 'chunks': [], 'text': f'Text from {file.name}'}
                for file in pdf_files
            ]
            
            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = pdf_files
                
                results = processor.process_directory(pdf_dir)
                
                assert len(results) == 3
                mock_process.assert_called_once_with(pdf_files)
    
    def test_export_workflow(self, tmp_path, user_info):
        """Test complete export workflow."""
        config = ProcessorConfig(enable_ai_validation=False)
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Prepare sample processing results
        sample_results = [
            {
                'file_path': tmp_path / "doc1.pdf",
                'text': 'Document 1 content',
                'chunks': [
                    {
                        'chunk_index': 0,
                        'text': 'Article 1. Banking regulations...',
                        'text_length': 35,
                        'word_count': 5,
                        'chunk_id': 'doc1_chunk_0'
                    }
                ],
                'articles': [],
                'validation_scores': [],
                'metadata': {'file_size': 1024}
            }
        ]
        
        output_file = tmp_path / "export_test.xlsx"
        
        # Mock the export process
        with patch.object(processor.exporter, 'export_complete_analysis') as mock_export:
            mock_export.return_value = output_file
            
            result_file = processor.export_to_excel(
                results=sample_results,
                output_file=output_file,
                user_info=user_info
            )
            
            assert result_file == output_file
            mock_export.assert_called_once()
            
            # Verify export was called with correct parameters
            call_args = mock_export.call_args
            assert 'chunks' in call_args[1] or len(call_args[0]) > 0
            assert call_args[1]['user_info'] == user_info
    
    def test_ai_validation_workflow(self, tmp_path):
        """Test workflow with AI validation enabled."""
        config = ProcessorConfig(
            enable_ai_validation=True,
            anthropic_api_key="test-key-12345",
            extract_articles=True,
            assess_materiality=True
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        test_file = tmp_path / "ai_test.pdf"
        test_file.write_bytes(b'Test content')
        
        sample_text = """
        Article 1. Capital Requirements
        Banks must maintain minimum capital ratios of 8%.
        """
        
        # Mock all AI-related operations
        with patch.object(processor.extractor, 'extract_text') as mock_extract, \
             patch.object(processor.validator, 'validate_chunk') as mock_validate, \
             patch.object(processor.validator, 'extract_articles') as mock_articles, \
             patch.object(processor.validator, 'assess_materiality') as mock_materiality:
            
            mock_extract.return_value = sample_text
            mock_validate.return_value = ValidationScore(
                completeness_score=85,
                reliability_score=90,
                legal_structure_score=80,
                overall_score=85
            )
            mock_articles.return_value = [
                Mock(number="Article 1", title="Capital Requirements", 
                     content="Banks must maintain...", materiality=MaterialityLevel.HIGH)
            ]
            mock_materiality.return_value = (MaterialityLevel.CRITICAL, "Mandatory capital requirement")
            
            result = processor.process_file(test_file)
            
            # Verify AI components were called
            assert 'validation_scores' in result
            assert 'articles' in result
            assert len(result['validation_scores']) > 0
            assert len(result['articles']) > 0
            
            mock_validate.assert_called()
            mock_articles.assert_called_once_with(sample_text)
    
    def test_error_recovery_workflow(self, tmp_path):
        """Test workflow error recovery mechanisms."""
        config = ProcessorConfig(enable_ai_validation=False)
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Create test files - one will fail
        good_file = tmp_path / "good.pdf"
        bad_file = tmp_path / "bad.pdf"
        good_file.write_bytes(b'Good content')
        bad_file.write_bytes(b'Bad content')
        
        files = [good_file, bad_file]
        
        # Mock extraction to fail on bad file
        def mock_extract_side_effect(file_path):
            if "bad" in str(file_path):
                raise Exception("Extraction failed")
            return "Extracted text from good file"
        
        with patch.object(processor.extractor, 'extract_text') as mock_extract:
            mock_extract.side_effect = mock_extract_side_effect
            
            # Process with error recovery
            results = processor.process_multiple_files(files, skip_errors=True)
            
            # Should only return successful result
            assert len(results) == 1
            assert results[0]['file_path'] == good_file
    
    def test_large_file_workflow(self, tmp_path):
        """Test workflow with large file processing."""
        config = ProcessorConfig(
            chunk_size=1000,
            chunk_overlap=200,
            enable_ai_validation=False
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        large_file = tmp_path / "large_regulation.pdf"
        large_file.write_bytes(b'Large content ' * 10000)  # ~140KB
        
        # Simulate large text extraction
        large_text = "Article content. " * 1000  # ~17KB of text
        
        with patch.object(processor.extractor, 'extract_text') as mock_extract:
            mock_extract.return_value = large_text
            
            result = processor.process_file(large_file)
            
            # Should handle large file without issues
            assert 'chunks' in result
            assert len(result['chunks']) > 1  # Should be chunked
            assert result['text'] == large_text
            
            # Verify all chunks have proper structure
            for chunk in result['chunks']:
                assert len(chunk['text']) <= config.chunk_size
                assert 'chunk_id' in chunk
    
    def test_unicode_content_workflow(self, tmp_path):
        """Test workflow with Unicode content."""
        config = ProcessorConfig(enable_ai_validation=False)
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        unicode_file = tmp_path / "unicode_règlement.pdf"
        unicode_file.write_bytes(b'Unicode content')
        
        unicode_text = """
        RÈGLEMENT N°R-2024-01
        
        Article 1. Définitions
        Au sens du présent règlement, on entend par:
        - Établissement financier: toute entité agréée...
        - Ratio de solvabilité: le rapport entre les fonds propres...
        
        Article 2. Exigences
        Les établissements doivent maintenir un ratio minimum de 8%.
        """
        
        with patch.object(processor.extractor, 'extract_text') as mock_extract:
            mock_extract.return_value = unicode_text
            
            result = processor.process_file(unicode_file)
            
            # Should handle Unicode content properly
            assert result['text'] == unicode_text
            assert len(result['chunks']) > 0
            
            # Verify Unicode characters are preserved in chunks
            combined_chunks = ' '.join(chunk['text'] for chunk in result['chunks'])
            assert 'RÈGLEMENT' in combined_chunks
            assert 'Définitions' in combined_chunks
            assert 'Établissement' in combined_chunks
    
    def test_memory_efficient_workflow(self, tmp_path):
        """Test memory-efficient processing of multiple large files."""
        config = ProcessorConfig(
            chunk_size=500,
            chunk_overlap=100,
            enable_ai_validation=False
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Create multiple large files
        large_files = []
        for i in range(5):
            file = tmp_path / f"large_reg_{i}.pdf"
            file.write_bytes(b'Large regulation content ' * 1000)  # ~25KB each
            large_files.append(file)
        
        # Mock extraction to return substantial text
        def mock_extract_large(file_path):
            return f"Large regulation text content for {file_path.name}. " * 500  # ~25KB text
        
        with patch.object(processor.extractor, 'extract_text') as mock_extract:
            mock_extract.side_effect = mock_extract_large
            
            # Process all files
            results = processor.process_multiple_files(large_files)
            
            # Should process all files successfully
            assert len(results) == 5
            
            # Verify each result has proper structure
            for i, result in enumerate(results):
                assert result['file_path'] == large_files[i]
                assert len(result['chunks']) > 0
                assert len(result['text']) > 0
    
    def test_end_to_end_workflow(self, tmp_path, user_info):
        """Test complete end-to-end workflow from file to Excel export."""
        config = ProcessorConfig(
            chunk_size=400,
            chunk_overlap=80,
            enable_ai_validation=False,
            extract_articles=True,
            clean_text=True
        )
        
        with patch('regulatory_processor.processor.setup_logging'):
            processor = RegulatoryDocumentProcessor(config)
        
        # Create test files
        regulation_dir = tmp_path / "regulations"
        regulation_dir.mkdir()
        
        test_files = []
        for i in range(2):
            file = regulation_dir / f"regulation_{i+1}.pdf"
            file.write_bytes(f'Regulation {i+1} PDF content'.encode())
            test_files.append(file)
        
        sample_texts = [
            "Regulation 1: Capital adequacy requirements for banks...",
            "Regulation 2: Risk management procedures and controls..."
        ]
        
        output_file = tmp_path / "complete_analysis.xlsx"
        
        # Mock the entire pipeline
        with patch.object(processor.extractor, 'extract_text') as mock_extract, \
             patch.object(processor.exporter, 'export_complete_analysis') as mock_export:
            
            mock_extract.side_effect = sample_texts
            mock_export.return_value = output_file
            
            # Step 1: Process directory
            with patch('pathlib.Path.glob', return_value=test_files):
                results = processor.process_directory(regulation_dir)
            
            # Step 2: Export results
            export_result = processor.export_to_excel(
                results=results,
                output_file=output_file,
                user_info=user_info
            )
            
            # Verify complete workflow
            assert len(results) == 2
            assert export_result == output_file
            
            # Verify all components were called
            assert mock_extract.call_count == 2
            mock_export.assert_called_once()
            
            # Verify export parameters
            export_args = mock_export.call_args[1]
            assert 'user_info' in export_args
            assert export_args['user_info'] == user_info