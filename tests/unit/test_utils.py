"""
Unit tests for utility functions.
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from regulatory_processor.utils import (
    text_utils, file_utils, validation_utils, 
    format_utils, logging_utils
)


class TestTextUtils:
    """Test text utility functions."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        dirty_text = "  Hello   world  \n\n\n  Test  \t\t  "
        cleaned = text_utils.clean_text(dirty_text)
        
        assert cleaned == "Hello world Test"
    
    def test_clean_text_special_chars(self):
        """Test cleaning special characters."""
        text_with_specials = "Hello\x00world\ufeff test\u200b"
        cleaned = text_utils.clean_text(text_with_specials)
        
        assert "\x00" not in cleaned
        assert "\ufeff" not in cleaned
        assert "\u200b" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Text\twith\nvarious\r\nwhitespace   types"
        normalized = text_utils.normalize_whitespace(text)
        
        assert "\t" not in normalized
        assert "\n" not in normalized
        assert "\r" not in normalized
        assert "   " not in normalized
        assert "Text with various whitespace types" == normalized
    
    def test_extract_sentences(self):
        """Test sentence extraction."""
        text = "First sentence. Second sentence! Third sentence? Fourth one."
        sentences = text_utils.extract_sentences(text)
        
        assert len(sentences) == 4
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"
        assert sentences[3] == "Fourth one."
    
    def test_word_count(self):
        """Test word counting."""
        text = "This is a test with five words"
        count = text_utils.word_count(text)
        
        assert count == 7
    
    def test_character_count(self):
        """Test character counting."""
        text = "Hello world"
        
        count_with_spaces = text_utils.character_count(text, include_spaces=True)
        count_without_spaces = text_utils.character_count(text, include_spaces=False)
        
        assert count_with_spaces == 11
        assert count_without_spaces == 10
    
    def test_detect_language(self):
        """Test language detection."""
        french_text = "Ceci est un texte en français."
        english_text = "This is an English text."
        
        # Mock language detection
        with patch('regulatory_processor.utils.text_utils.detect') as mock_detect:
            mock_detect.side_effect = ['fr', 'en']
            
            fr_lang = text_utils.detect_language(french_text)
            en_lang = text_utils.detect_language(english_text)
            
            assert fr_lang == 'fr'
            assert en_lang == 'en'
    
    def test_truncate_text(self):
        """Test text truncation."""
        long_text = "This is a very long text that needs to be truncated"
        
        truncated = text_utils.truncate_text(long_text, max_length=20)
        
        assert len(truncated) <= 20
        assert truncated.endswith('...')
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = "This regulation defines banking requirements for financial institutions"
        
        keywords = text_utils.extract_keywords(text, top_k=3)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 3
        # Should contain important words, not stopwords
        assert not any(word in ['this', 'for', 'the'] for word in keywords)


class TestFileUtils:
    """Test file utility functions."""
    
    def test_validate_file_path(self, tmp_path):
        """Test file path validation."""
        # Valid file
        valid_file = tmp_path / "valid.pdf"
        valid_file.write_bytes(b'content')
        
        assert file_utils.validate_file_path(valid_file)
        
        # Invalid file
        invalid_file = tmp_path / "nonexistent.pdf"
        
        assert not file_utils.validate_file_path(invalid_file)
    
    def test_get_file_size(self, tmp_path):
        """Test file size calculation."""
        test_file = tmp_path / "test.pdf"
        content = b'test content'
        test_file.write_bytes(content)
        
        size_bytes = file_utils.get_file_size(test_file)
        size_mb = file_utils.get_file_size(test_file, unit='MB')
        
        assert size_bytes == len(content)
        assert size_mb == len(content) / (1024 * 1024)
    
    def test_ensure_directory(self, tmp_path):
        """Test directory creation."""
        new_dir = tmp_path / "new" / "nested" / "directory"
        
        file_utils.ensure_directory(new_dir)
        
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_safe_filename(self):
        """Test safe filename generation."""
        unsafe_name = "file/with\\unsafe:characters*?.txt"
        safe_name = file_utils.safe_filename(unsafe_name)
        
        assert '/' not in safe_name
        assert '\\' not in safe_name
        assert ':' not in safe_name
        assert '*' not in safe_name
        assert '?' not in safe_name
    
    def test_backup_file(self, tmp_path):
        """Test file backup functionality."""
        original_file = tmp_path / "original.txt"
        original_file.write_text("original content")
        
        backup_path = file_utils.backup_file(original_file)
        
        assert backup_path.exists()
        assert backup_path.read_text() == "original content"
        assert "backup" in str(backup_path) or backup_path.suffix == ".bak"
    
    def test_cleanup_temp_files(self, tmp_path):
        """Test temporary file cleanup."""
        temp_files = []
        for i in range(3):
            temp_file = tmp_path / f"temp_{i}.tmp"
            temp_file.write_text(f"temp content {i}")
            temp_files.append(temp_file)
        
        file_utils.cleanup_temp_files(temp_files)
        
        # Files should be removed
        for temp_file in temp_files:
            assert not temp_file.exists()
    
    def test_copy_file_with_metadata(self, tmp_path):
        """Test file copying with metadata preservation."""
        source = tmp_path / "source.txt"
        source.write_text("source content")
        
        destination = tmp_path / "destination.txt"
        
        file_utils.copy_file_with_metadata(source, destination)
        
        assert destination.exists()
        assert destination.read_text() == "source content"


class TestValidationUtils:
    """Test validation utility functions."""
    
    def test_validate_email(self):
        """Test email validation."""
        valid_emails = [
            "user@example.com",
            "test.email@domain.org",
            "user+tag@example.co.uk"
        ]
        
        invalid_emails = [
            "invalid-email",
            "@domain.com",
            "user@",
            "user space@domain.com"
        ]
        
        for email in valid_emails:
            assert validation_utils.validate_email(email)
        
        for email in invalid_emails:
            assert not validation_utils.validate_email(email)
    
    def test_validate_api_key(self):
        """Test API key validation."""
        valid_keys = [
            "sk-ant-api03-1234567890abcdef",
            "test-key-12345"
        ]
        
        invalid_keys = [
            "",
            "short",
            None
        ]
        
        for key in valid_keys:
            assert validation_utils.validate_api_key(key)
        
        for key in invalid_keys:
            assert not validation_utils.validate_api_key(key)
    
    def test_validate_chunk_size(self):
        """Test chunk size validation."""
        valid_sizes = [100, 1000, 5000]
        invalid_sizes = [0, -100, 50001]  # Too small, negative, too large
        
        for size in valid_sizes:
            assert validation_utils.validate_chunk_size(size)
        
        for size in invalid_sizes:
            assert not validation_utils.validate_chunk_size(size)
    
    def test_validate_file_extension(self):
        """Test file extension validation."""
        pdf_files = ["doc.pdf", "document.PDF", "/path/to/file.pdf"]
        non_pdf_files = ["doc.txt", "image.png", "file.docx"]
        
        for file in pdf_files:
            assert validation_utils.validate_file_extension(file, ['.pdf'])
        
        for file in non_pdf_files:
            assert not validation_utils.validate_file_extension(file, ['.pdf'])
    
    def test_validate_json_structure(self):
        """Test JSON structure validation."""
        valid_json = {
            "completeness_score": 85,
            "reliability_score": 90,
            "overall_score": 87
        }
        
        invalid_json = {
            "completeness_score": "not_a_number",
            "missing_required": "field"
        }
        
        required_fields = ["completeness_score", "reliability_score", "overall_score"]
        
        assert validation_utils.validate_json_structure(valid_json, required_fields)
        assert not validation_utils.validate_json_structure(invalid_json, required_fields)
    
    def test_sanitize_input(self):
        """Test input sanitization."""
        dangerous_input = "<script>alert('xss')</script>Hello"
        safe_input = validation_utils.sanitize_input(dangerous_input)
        
        assert "<script>" not in safe_input
        assert "alert" not in safe_input
        assert "Hello" in safe_input


class TestFormatUtils:
    """Test formatting utility functions."""
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        test_datetime = datetime(2024, 12, 18, 14, 30, 22)
        
        filename_format = format_utils.format_timestamp(test_datetime, format_type='filename')
        display_format = format_utils.format_timestamp(test_datetime, format_type='display')
        iso_format = format_utils.format_timestamp(test_datetime, format_type='iso')
        
        assert filename_format == "20241218_143022"
        assert display_format == "18/12/2024 14:30:22"
        assert iso_format == "2024-12-18T14:30:22"
    
    def test_format_file_size(self):
        """Test file size formatting."""
        sizes_bytes = [512, 1024, 1048576, 1073741824]
        expected = ["512 B", "1.0 KB", "1.0 MB", "1.0 GB"]
        
        for size, expected_format in zip(sizes_bytes, expected):
            formatted = format_utils.format_file_size(size)
            assert formatted == expected_format
    
    def test_format_percentage(self):
        """Test percentage formatting."""
        values = [0.85, 0.901, 0.7, 1.0]
        expected = ["85%", "90%", "70%", "100%"]
        
        for value, expected_format in zip(values, expected):
            formatted = format_utils.format_percentage(value)
            assert formatted == expected_format
    
    def test_format_duration(self):
        """Test duration formatting."""
        durations = [30, 75, 3665, 86461]  # seconds
        expected = ["30s", "1m 15s", "1h 1m 5s", "1d 1m 1s"]
        
        for duration, expected_format in zip(durations, expected):
            formatted = format_utils.format_duration(duration)
            assert formatted == expected_format
    
    def test_format_number(self):
        """Test number formatting."""
        numbers = [1000, 1500.5, 1000000]
        expected = ["1,000", "1,500.5", "1,000,000"]
        
        for number, expected_format in zip(numbers, expected):
            formatted = format_utils.format_number(number)
            assert formatted == expected_format
    
    def test_truncate_with_ellipsis(self):
        """Test text truncation with ellipsis."""
        long_text = "This is a very long text that needs truncation"
        
        truncated = format_utils.truncate_with_ellipsis(long_text, max_length=20)
        
        assert len(truncated) <= 20
        assert truncated.endswith('...')
        assert "This is a very" in truncated


class TestLoggingUtils:
    """Test logging utility functions."""
    
    def test_setup_logger(self):
        """Test logger setup."""
        logger = logging_utils.setup_logger(
            name="test_logger",
            level="DEBUG",
            log_file="test.log"
        )
        
        assert logger.name == "test_logger"
        assert logger.level <= 10  # DEBUG level
    
    def test_log_performance(self):
        """Test performance logging decorator."""
        @logging_utils.log_performance
        def test_function():
            import time
            time.sleep(0.1)
            return "result"
        
        with patch('regulatory_processor.utils.logging_utils.logger') as mock_logger:
            result = test_function()
            
            assert result == "result"
            mock_logger.info.assert_called()
    
    def test_log_error_context(self):
        """Test error context logging."""
        with patch('regulatory_processor.utils.logging_utils.logger') as mock_logger:
            with logging_utils.log_error_context("test_operation"):
                raise ValueError("Test error")
            
            # Should have logged the error context
            mock_logger.error.assert_called()
    
    def test_format_log_message(self):
        """Test log message formatting."""
        message = logging_utils.format_log_message(
            "Processing file",
            file_name="test.pdf",
            chunk_count=5,
            processing_time=1.23
        )
        
        assert "Processing file" in message
        assert "test.pdf" in message
        assert "5" in message
        assert "1.23" in message
    
    def test_sanitize_log_data(self):
        """Test log data sanitization."""
        sensitive_data = {
            "api_key": "sk-ant-api03-secret123",
            "password": "secret_password",
            "user_name": "john_doe",
            "file_content": "public content"
        }
        
        sanitized = logging_utils.sanitize_log_data(sensitive_data)
        
        assert "***" in sanitized["api_key"]
        assert "***" in sanitized["password"]
        assert sanitized["user_name"] == "john_doe"  # Not sensitive
        assert sanitized["file_content"] == "public content"  # Not sensitive


class TestUtilsIntegration:
    """Test integration between utility functions."""
    
    def test_file_processing_pipeline(self, tmp_path):
        """Test complete file processing pipeline using utils."""
        # Create test file
        test_file = tmp_path / "test_file.pdf"
        test_content = "Test content for file processing"
        test_file.write_text(test_content)
        
        # Validate file
        assert file_utils.validate_file_path(test_file)
        
        # Get file info
        file_size = file_utils.get_file_size(test_file)
        assert file_size > 0
        
        # Clean and process content
        cleaned_content = text_utils.clean_text(test_content)
        word_count = text_utils.word_count(cleaned_content)
        
        assert word_count > 0
        assert cleaned_content == test_content  # Already clean
    
    def test_validation_and_formatting_pipeline(self):
        """Test validation and formatting pipeline."""
        # Test data
        email = "user@example.com"
        file_size = 1048576  # 1MB
        percentage = 0.856
        
        # Validate
        assert validation_utils.validate_email(email)
        
        # Format
        formatted_size = format_utils.format_file_size(file_size)
        formatted_percentage = format_utils.format_percentage(percentage)
        
        assert formatted_size == "1.0 MB"
        assert formatted_percentage == "86%"
    
    def test_error_handling_across_utils(self, tmp_path):
        """Test error handling across different utility functions."""
        # Test file utils with invalid file
        invalid_file = tmp_path / "nonexistent.pdf"
        
        assert not file_utils.validate_file_path(invalid_file)
        
        # Test text utils with None input
        with pytest.raises((TypeError, AttributeError)):
            text_utils.clean_text(None)
        
        # Test validation utils with invalid input
        assert not validation_utils.validate_email(None)
        assert not validation_utils.validate_chunk_size(-1)
    
    def test_performance_optimization(self):
        """Test performance optimization utilities."""
        # Test with large text processing
        large_text = "Word " * 10000  # 50KB of text
        
        # Should handle large text efficiently
        cleaned = text_utils.clean_text(large_text)
        word_count = text_utils.word_count(cleaned)
        sentences = text_utils.extract_sentences(cleaned)
        
        assert len(cleaned) > 0
        assert word_count == 10000
        assert len(sentences) >= 1
    
    def test_unicode_handling_across_utils(self):
        """Test Unicode handling across utility functions."""
        unicode_text = "Texte français 中文 العربية 日本語"
        
        # Text utils should handle Unicode
        cleaned = text_utils.clean_text(unicode_text)
        word_count = text_utils.word_count(cleaned)
        
        assert "français" in cleaned
        assert "中文" in cleaned
        assert word_count > 0
        
        # File utils should handle Unicode filenames
        unicode_filename = "français_中文.pdf"
        safe_filename = file_utils.safe_filename(unicode_filename)
        
        assert len(safe_filename) > 0
        # Should preserve or safely convert Unicode characters
    
    def test_memory_efficiency(self):
        """Test memory efficiency of utility functions."""
        # Create large dataset
        large_data = ["Text chunk " + str(i) for i in range(10000)]
        
        # Process in batches using utils
        processed_count = 0
        for text in large_data:
            cleaned = text_utils.clean_text(text)
            if validation_utils.validate_chunk_size(len(cleaned)):
                processed_count += 1
        
        assert processed_count == len(large_data)
        # Should complete without memory issues