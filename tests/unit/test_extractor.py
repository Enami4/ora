"""
Unit tests for text extraction module.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from regulatory_processor.extractor import PDFTextExtractor, TextExtractor
from regulatory_processor.config import ProcessorConfig


class TestPDFTextExtractor:
    """Test cases for PDFTextExtractor class."""
    
    def test_init_with_config(self, basic_config):
        """Test extractor initialization with configuration."""
        extractor = PDFTextExtractor(basic_config)
        assert extractor.config == basic_config
        assert extractor.max_file_size_mb == basic_config.max_file_size_mb
    
    def test_init_without_config(self):
        """Test extractor initialization without configuration."""
        extractor = PDFTextExtractor()
        assert isinstance(extractor.config, ProcessorConfig)
        assert extractor.max_file_size_mb == 100  # default value
    
    def test_validate_file_exists(self, tmp_path):
        """Test file existence validation."""
        extractor = PDFTextExtractor()
        
        # Create a test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b'test content')
        
        # Should pass validation
        extractor._validate_file(test_file)
        
        # Should raise error for non-existent file
        with pytest.raises(FileNotFoundError):
            extractor._validate_file(tmp_path / "nonexistent.pdf")
    
    def test_validate_file_size(self, tmp_path):
        """Test file size validation."""
        config = ProcessorConfig(max_file_size_mb=1)  # 1MB limit
        extractor = PDFTextExtractor(config)
        
        # Create small file
        small_file = tmp_path / "small.pdf"
        small_file.write_bytes(b'small content')
        extractor._validate_file(small_file)  # Should pass
        
        # Create large file (simulate)
        large_file = tmp_path / "large.pdf"
        large_file.write_bytes(b'x' * (2 * 1024 * 1024))  # 2MB
        
        with pytest.raises(ValueError, match="File size exceeds maximum"):
            extractor._validate_file(large_file)
    
    def test_validate_file_extension(self, tmp_path):
        """Test file extension validation."""
        extractor = PDFTextExtractor()
        
        # Valid PDF extension
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b'test')
        extractor._validate_file(pdf_file)  # Should pass
        
        # Invalid extension
        txt_file = tmp_path / "test.txt"
        txt_file.write_bytes(b'test')
        
        with pytest.raises(ValueError, match="must be a PDF file"):
            extractor._validate_file(txt_file)
    
    @patch('regulatory_processor.extractor.PyPDF2.PdfReader')
    def test_extract_with_pypdf2_success(self, mock_pdf_reader, tmp_path):
        """Test successful text extraction with PyPDF2."""
        # Setup mock
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test PDF content"
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b'fake pdf content')
        
        extractor = PDFTextExtractor()
        result = extractor._extract_with_pypdf2(test_file)
        
        assert result == "Test PDF content"
        mock_pdf_reader.assert_called_once_with(test_file)
    
    @patch('regulatory_processor.extractor.pdfplumber.open')
    def test_extract_with_pdfplumber_success(self, mock_pdfplumber, tmp_path):
        """Test successful text extraction with pdfplumber."""
        # Setup mock
        mock_page = Mock()
        mock_page.extract_text.return_value = "Pdfplumber content"
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        mock_pdfplumber.return_value = mock_pdf
        
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b'fake pdf content')
        
        extractor = PDFTextExtractor()
        result = extractor._extract_with_pdfplumber(test_file)
        
        assert result == "Pdfplumber content"
        mock_pdfplumber.assert_called_once_with(test_file)
    
    @patch('regulatory_processor.extractor.pdfplumber.open')
    @patch('regulatory_processor.extractor.PyPDF2.PdfReader')
    def test_extract_text_fallback_strategy(self, mock_pypdf2, mock_pdfplumber, tmp_path):
        """Test extraction fallback from PyPDF2 to pdfplumber."""
        # PyPDF2 fails
        mock_pypdf2.side_effect = Exception("PyPDF2 error")
        
        # pdfplumber succeeds
        mock_page = Mock()
        mock_page.extract_text.return_value = "Fallback content"
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        mock_pdfplumber.return_value = mock_pdf
        
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b'fake pdf content')
        
        extractor = PDFTextExtractor()
        result = extractor.extract_text(test_file)
        
        assert result == "Fallback content"
        # Verify both methods were attempted
        mock_pypdf2.assert_called_once()
        mock_pdfplumber.assert_called_once()
    
    @patch('regulatory_processor.extractor.pdfplumber.open')
    @patch('regulatory_processor.extractor.PyPDF2.PdfReader')
    def test_extract_text_both_methods_fail(self, mock_pypdf2, mock_pdfplumber, tmp_path):
        """Test behavior when both extraction methods fail."""
        # Both methods fail
        mock_pypdf2.side_effect = Exception("PyPDF2 error")
        mock_pdfplumber.side_effect = Exception("pdfplumber error")
        
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b'fake pdf content')
        
        extractor = PDFTextExtractor()
        
        with pytest.raises(RuntimeError, match="Failed to extract text from PDF"):
            extractor.extract_text(test_file)
    
    def test_clean_text_basic(self):
        """Test basic text cleaning functionality."""
        extractor = PDFTextExtractor()
        
        dirty_text = "  Hello   world  \n\n\n  Test  \t\t  "
        cleaned = extractor._clean_text(dirty_text)
        
        assert cleaned == "Hello world Test"
    
    def test_clean_text_special_characters(self):
        """Test cleaning of special characters."""
        extractor = PDFTextExtractor()
        
        text_with_specials = "Hello\x00world\ufeff test"
        cleaned = extractor._clean_text(text_with_specials)
        
        assert "\x00" not in cleaned
        assert "\ufeff" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned
    
    def test_extract_metadata(self, tmp_path):
        """Test metadata extraction from PDF."""
        extractor = PDFTextExtractor()
        
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b'fake pdf content')
        
        with patch('regulatory_processor.extractor.PyPDF2.PdfReader') as mock_reader:
            mock_metadata = {
                '/Title': 'Test Document',
                '/Author': 'Test Author',
                '/CreationDate': 'D:20240101000000'
            }
            mock_reader.return_value.metadata = mock_metadata
            
            metadata = extractor._extract_metadata(test_file)
            
            assert metadata['title'] == 'Test Document'
            assert metadata['author'] == 'Test Author'
            assert 'creation_date' in metadata


class TestTextExtractor:
    """Test cases for general TextExtractor functionality."""
    
    def test_extract_from_multiple_files(self, tmp_path):
        """Test extraction from multiple PDF files."""
        config = ProcessorConfig()
        extractor = PDFTextExtractor(config)
        
        # Create multiple test files
        file1 = tmp_path / "doc1.pdf"
        file2 = tmp_path / "doc2.pdf"
        file1.write_bytes(b'content1')
        file2.write_bytes(b'content2')
        
        files = [file1, file2]
        
        with patch.object(extractor, 'extract_text') as mock_extract:
            mock_extract.side_effect = ["Content 1", "Content 2"]
            
            results = extractor.extract_from_multiple_files(files)
            
            assert len(results) == 2
            assert results[0]['file_path'] == file1
            assert results[0]['text'] == "Content 1"
            assert results[1]['file_path'] == file2
            assert results[1]['text'] == "Content 2"
    
    def test_extract_from_directory(self, tmp_path):
        """Test extraction from directory containing PDFs."""
        extractor = PDFTextExtractor()
        
        # Create directory with PDF files
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        
        (pdf_dir / "doc1.pdf").write_bytes(b'content1')
        (pdf_dir / "doc2.pdf").write_bytes(b'content2')
        (pdf_dir / "not_pdf.txt").write_bytes(b'text content')  # Should be ignored
        
        with patch.object(extractor, 'extract_text') as mock_extract:
            mock_extract.return_value = "Extracted content"
            
            results = extractor.extract_from_directory(pdf_dir)
            
            assert len(results) == 2  # Only PDF files
            assert all('text' in result for result in results)
            assert all('file_path' in result for result in results)
    
    def test_get_file_info(self, tmp_path):
        """Test file information extraction."""
        extractor = PDFTextExtractor()
        
        test_file = tmp_path / "test.pdf"
        test_content = b'test content for file info'
        test_file.write_bytes(test_content)
        
        info = extractor._get_file_info(test_file)
        
        assert info['file_name'] == 'test.pdf'
        assert info['file_size'] == len(test_content)
        assert 'modification_time' in info
        assert info['file_extension'] == '.pdf'
    
    def test_batch_processing_error_handling(self, tmp_path):
        """Test error handling in batch processing."""
        extractor = PDFTextExtractor()
        
        # Create files: one valid, one that will cause error
        valid_file = tmp_path / "valid.pdf"
        error_file = tmp_path / "error.pdf"
        valid_file.write_bytes(b'valid content')
        error_file.write_bytes(b'error content')
        
        files = [valid_file, error_file]
        
        with patch.object(extractor, 'extract_text') as mock_extract:
            mock_extract.side_effect = ["Valid content", Exception("Extraction error")]
            
            results = extractor.extract_from_multiple_files(files, skip_errors=True)
            
            # Should return only successful extraction
            assert len(results) == 1
            assert results[0]['text'] == "Valid content"
    
    def test_progress_callback(self, tmp_path):
        """Test progress callback functionality."""
        extractor = PDFTextExtractor()
        
        files = [tmp_path / f"doc{i}.pdf" for i in range(3)]
        for file in files:
            file.write_bytes(b'content')
        
        progress_calls = []
        
        def progress_callback(current, total, filename):
            progress_calls.append((current, total, filename))
        
        with patch.object(extractor, 'extract_text', return_value="Content"):
            extractor.extract_from_multiple_files(files, progress_callback=progress_callback)
        
        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3, files[0].name)
        assert progress_calls[1] == (2, 3, files[1].name)
        assert progress_calls[2] == (3, 3, files[2].name)


class TestExtractionEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_pdf(self, tmp_path):
        """Test handling of empty PDF files."""
        extractor = PDFTextExtractor()
        
        empty_file = tmp_path / "empty.pdf"
        empty_file.write_bytes(b'')  # Empty file
        
        with patch('regulatory_processor.extractor.PyPDF2.PdfReader') as mock_reader:
            mock_reader.return_value.pages = []  # No pages
            
            result = extractor.extract_text(empty_file)
            assert result == ""
    
    def test_corrupted_pdf(self, tmp_path):
        """Test handling of corrupted PDF files."""
        extractor = PDFTextExtractor()
        
        corrupted_file = tmp_path / "corrupted.pdf"
        corrupted_file.write_bytes(b'not a real pdf content')
        
        with patch('regulatory_processor.extractor.PyPDF2.PdfReader') as mock_pypdf2:
            with patch('regulatory_processor.extractor.pdfplumber.open') as mock_pdfplumber:
                mock_pypdf2.side_effect = Exception("Corrupted PDF")
                mock_pdfplumber.side_effect = Exception("Cannot open")
                
                with pytest.raises(RuntimeError):
                    extractor.extract_text(corrupted_file)
    
    def test_large_text_content(self):
        """Test handling of very large text content."""
        extractor = PDFTextExtractor()
        
        # Generate large text content
        large_text = "Test content " * 100000  # ~1.3MB of text
        
        cleaned = extractor._clean_text(large_text)
        
        # Should handle large content without errors
        assert len(cleaned) > 0
        assert "Test content" in cleaned
    
    def test_unicode_content(self):
        """Test handling of Unicode characters."""
        extractor = PDFTextExtractor()
        
        unicode_text = "Texte français avec accents: éàüñ 中文字符 العربية"
        
        cleaned = extractor._clean_text(unicode_text)
        
        # Should preserve Unicode characters
        assert "français" in cleaned
        assert "éàüñ" in cleaned
        assert "中文字符" in cleaned
        assert "العربية" in cleaned
    
    def test_memory_efficiency(self, tmp_path):
        """Test memory usage with multiple large files."""
        extractor = PDFTextExtractor()
        
        # Create multiple files
        files = []
        for i in range(5):
            file = tmp_path / f"large_{i}.pdf"
            file.write_bytes(b'large content' * 1000)  # Simulate large files
            files.append(file)
        
        with patch.object(extractor, 'extract_text') as mock_extract:
            # Simulate large text extraction
            mock_extract.return_value = "Large extracted content " * 10000
            
            # Should process without memory issues
            results = extractor.extract_from_multiple_files(files)
            
            assert len(results) == 5
            # Verify results are properly structured
            for result in results:
                assert 'text' in result
                assert 'file_path' in result
                assert 'metadata' in result