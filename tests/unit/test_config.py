"""
Unit tests for configuration module.
"""

import pytest
import os
from unittest.mock import patch
from regulatory_processor.config import ProcessorConfig, setup_logging


class TestProcessorConfig:
    """Test cases for ProcessorConfig class."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = ProcessorConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_file_size_mb == 100
        assert config.log_level == "INFO"
        assert config.clean_text is True
        assert config.extract_articles is True
        assert config.enable_ai_validation is False
        assert config.assess_materiality is True
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = ProcessorConfig(
            chunk_size=800,
            chunk_overlap=150,
            max_file_size_mb=50,
            log_level="DEBUG",
            enable_ai_validation=True,
            anthropic_api_key="test-key"
        )
        
        assert config.chunk_size == 800
        assert config.chunk_overlap == 150
        assert config.max_file_size_mb == 50
        assert config.log_level == "DEBUG"
        assert config.enable_ai_validation is True
        assert config.anthropic_api_key == "test-key"
    
    def test_chunk_size_validation(self):
        """Test chunk size validation."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ProcessorConfig(chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ProcessorConfig(chunk_size=-100)
    
    def test_chunk_overlap_validation(self):
        """Test chunk overlap validation."""
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            ProcessorConfig(chunk_overlap=-50)
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            ProcessorConfig(chunk_size=500, chunk_overlap=600)
    
    def test_log_level_validation(self):
        """Test log level validation."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in valid_levels:
            config = ProcessorConfig(log_level=level)
            assert config.log_level == level
        
        with pytest.raises(ValueError, match="log_level must be one of"):
            ProcessorConfig(log_level="INVALID")
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'env-api-key'})
    def test_api_key_from_environment(self):
        """Test API key loading from environment variable."""
        config = ProcessorConfig(enable_ai_validation=True)
        assert config.anthropic_api_key == 'env-api-key'
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_warning(self):
        """Test warning when API key is missing."""
        with patch('regulatory_processor.config.logger') as mock_logger:
            ProcessorConfig(enable_ai_validation=True)
            mock_logger.warning.assert_called_once()


class TestSetupLogging:
    """Test cases for logging setup."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        config = ProcessorConfig(log_level="INFO", log_file="test.log")
        logger = setup_logging(config)
        
        assert logger.name == 'regulatory_processor'
        assert logger.level <= 20  # INFO level or lower
    
    def test_setup_logging_debug(self):
        """Test debug logging setup."""
        config = ProcessorConfig(log_level="DEBUG", log_file="test_debug.log")
        logger = setup_logging(config)
        
        assert logger.level <= 10  # DEBUG level or lower
    
    @patch('logging.basicConfig')
    def test_logging_configuration(self, mock_basic_config):
        """Test logging configuration parameters."""
        config = ProcessorConfig(log_level="WARNING", log_file="custom.log")
        setup_logging(config)
        
        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        
        # Check that basicConfig was called with correct parameters
        assert call_args[1]['level'] == 30  # WARNING level
        assert any('custom.log' in str(handler) for handler in call_args[1]['handlers'])


class TestConfigurationConsistency:
    """Test configuration consistency and edge cases."""
    
    def test_configuration_immutability(self):
        """Test that configuration behaves consistently."""
        config1 = ProcessorConfig(chunk_size=800)
        config2 = ProcessorConfig(chunk_size=800)
        
        assert config1.chunk_size == config2.chunk_size
        assert config1.chunk_overlap == config2.chunk_overlap
    
    def test_boolean_settings_consistency(self):
        """Test boolean settings are handled correctly."""
        config = ProcessorConfig(
            enable_ai_validation=True,
            extract_articles=False,
            assess_materiality=True,
            clean_text=False
        )
        
        assert config.enable_ai_validation is True
        assert config.extract_articles is False
        assert config.assess_materiality is True
        assert config.clean_text is False
    
    def test_model_selection_options(self):
        """Test AI model selection options."""
        models = [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229"
        ]
        
        for model in models:
            config = ProcessorConfig(ai_model=model)
            assert config.ai_model == model
    
    def test_edge_case_values(self):
        """Test edge case configuration values."""
        # Minimum valid values
        config = ProcessorConfig(
            chunk_size=1,
            chunk_overlap=0,
            max_file_size_mb=1
        )
        assert config.chunk_size == 1
        assert config.chunk_overlap == 0
        assert config.max_file_size_mb == 1
        
        # Large values
        config = ProcessorConfig(
            chunk_size=10000,
            chunk_overlap=5000,
            max_file_size_mb=1000
        )
        assert config.chunk_size == 10000
        assert config.chunk_overlap == 5000
        assert config.max_file_size_mb == 1000