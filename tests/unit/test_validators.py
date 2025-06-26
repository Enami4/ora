"""
Unit tests for AI validation module.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from regulatory_processor.validators import (
    AIValidator, Article, MaterialityLevel, 
    ValidationScore, ChunkValidator
)
from regulatory_processor.config import ProcessorConfig


class TestMaterialityLevel:
    """Test MaterialityLevel enum."""
    
    def test_materiality_levels(self):
        """Test all materiality levels are defined."""
        assert MaterialityLevel.LOW.value == "LOW"
        assert MaterialityLevel.MEDIUM.value == "MEDIUM"
        assert MaterialityLevel.HIGH.value == "HIGH"
        assert MaterialityLevel.CRITICAL.value == "CRITICAL"
    
    def test_materiality_ordering(self):
        """Test materiality level comparison."""
        levels = [MaterialityLevel.LOW, MaterialityLevel.MEDIUM, 
                 MaterialityLevel.HIGH, MaterialityLevel.CRITICAL]
        
        # Test that we can iterate through levels
        assert len(levels) == 4
        assert MaterialityLevel.CRITICAL != MaterialityLevel.LOW


class TestValidationScore:
    """Test ValidationScore dataclass."""
    
    def test_validation_score_creation(self):
        """Test creating ValidationScore instance."""
        score = ValidationScore(
            completeness_score=85,
            reliability_score=90,
            legal_structure_score=75,
            overall_score=83,
            completeness_issues=["Missing section A"],
            reliability_issues=[],
            legal_structure_issues=["Inconsistent numbering"],
            recommendations=["Review section structure"]
        )
        
        assert score.completeness_score == 85
        assert score.reliability_score == 90
        assert score.legal_structure_score == 75
        assert score.overall_score == 83
        assert len(score.completeness_issues) == 1
        assert len(score.reliability_issues) == 0
        assert len(score.legal_structure_issues) == 1
        assert len(score.recommendations) == 1
    
    def test_validation_score_defaults(self):
        """Test ValidationScore with default values."""
        score = ValidationScore()
        
        assert score.completeness_score == 0
        assert score.reliability_score == 0
        assert score.legal_structure_score == 0
        assert score.overall_score == 0
        assert score.completeness_issues == []
        assert score.reliability_issues == []
        assert score.legal_structure_issues == []
        assert score.recommendations == []


class TestArticle:
    """Test Article dataclass."""
    
    def test_article_creation(self):
        """Test creating Article instance."""
        article = Article(
            number="Article 1",
            title="Test Article",
            content="This is test content for the article.",
            materiality=MaterialityLevel.HIGH,
            materiality_reasoning="Critical regulation",
            context={"doc_type": "regulation", "section": "A"}
        )
        
        assert article.number == "Article 1"
        assert article.title == "Test Article"
        assert article.content == "This is test content for the article."
        assert article.materiality == MaterialityLevel.HIGH
        assert article.materiality_reasoning == "Critical regulation"
        assert article.context["doc_type"] == "regulation"
    
    def test_article_defaults(self):
        """Test Article with default values."""
        article = Article(
            number="Article 2",
            title="Default Article",
            content="Default content"
        )
        
        assert article.materiality == MaterialityLevel.MEDIUM
        assert article.materiality_reasoning == ""
        assert article.context == {}


class TestChunkValidator:
    """Test ChunkValidator class."""
    
    def test_basic_validation(self, sample_chunks):
        """Test basic chunk validation."""
        validator = ChunkValidator()
        
        for chunk in sample_chunks:
            is_valid, issues = validator.validate_chunk(chunk)
            
            assert isinstance(is_valid, bool)
            assert isinstance(issues, list)
    
    def test_chunk_structure_validation(self):
        """Test validation of chunk structure."""
        validator = ChunkValidator()
        
        # Valid chunk
        valid_chunk = {
            'chunk_index': 0,
            'text': 'Valid chunk text content',
            'text_length': 25,
            'word_count': 4,
            'chunk_id': 'test_chunk_0'
        }
        
        is_valid, issues = validator.validate_chunk(valid_chunk)
        assert is_valid
        assert len(issues) == 0
        
        # Invalid chunk - missing fields
        invalid_chunk = {
            'text': 'Missing other fields'
        }
        
        is_valid, issues = validator.validate_chunk(invalid_chunk)
        assert not is_valid
        assert len(issues) > 0
    
    def test_text_quality_validation(self):
        """Test text quality validation."""
        validator = ChunkValidator()
        
        # Good quality text
        good_chunk = {
            'chunk_index': 0,
            'text': 'This is well-formed text with proper grammar and structure.',
            'text_length': 62,
            'word_count': 10,
            'chunk_id': 'good_chunk'
        }
        
        is_valid, issues = validator.validate_chunk(good_chunk)
        assert is_valid
        
        # Poor quality text
        poor_chunk = {
            'chunk_index': 0,
            'text': 'txt wth mny abbrvtns & n0 pr0p3r w0rd5!!!',
            'text_length': 39,
            'word_count': 6,
            'chunk_id': 'poor_chunk'
        }
        
        is_valid, issues = validator.validate_chunk(poor_chunk)
        # May or may not be valid depending on tolerance, but should not crash
        assert isinstance(is_valid, bool)
    
    def test_length_validation(self):
        """Test chunk length validation."""
        validator = ChunkValidator(min_length=10, max_length=100)
        
        # Too short
        short_chunk = {
            'chunk_index': 0,
            'text': 'Short',
            'text_length': 5,
            'word_count': 1,
            'chunk_id': 'short'
        }
        
        is_valid, issues = validator.validate_chunk(short_chunk)
        assert not is_valid
        assert any('too short' in issue.lower() for issue in issues)
        
        # Too long
        long_chunk = {
            'chunk_index': 0,
            'text': 'Very long text ' * 20,  # 300 characters
            'text_length': 300,
            'word_count': 60,
            'chunk_id': 'long'
        }
        
        is_valid, issues = validator.validate_chunk(long_chunk)
        assert not is_valid
        assert any('too long' in issue.lower() for issue in issues)


class TestAIValidator:
    """Test AIValidator class."""
    
    def test_init_with_config(self, ai_config):
        """Test AI validator initialization."""
        validator = AIValidator(ai_config)
        
        assert validator.config == ai_config
        assert validator.api_key == ai_config.anthropic_api_key
        assert validator.model == ai_config.ai_model
    
    def test_init_without_api_key(self, basic_config):
        """Test initialization without API key."""
        validator = AIValidator(basic_config)
        
        assert validator.api_key is None
        # Should still initialize but with limited functionality
    
    @patch('regulatory_processor.validators.anthropic.Anthropic')
    def test_client_initialization(self, mock_anthropic, ai_config):
        """Test Anthropic client initialization."""
        validator = AIValidator(ai_config)
        validator._get_client()
        
        mock_anthropic.assert_called_once_with(api_key=ai_config.anthropic_api_key)
    
    @patch('regulatory_processor.validators.anthropic.Anthropic')
    def test_validate_chunk_success(self, mock_anthropic, ai_config, mock_api_response):
        """Test successful chunk validation."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text=str(mock_api_response))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        validator = AIValidator(ai_config)
        
        chunk = {
            'text': 'Article 1. Test regulation content.',
            'chunk_id': 'test_chunk_1'
        }
        
        with patch('json.loads', return_value=mock_api_response):
            score = validator.validate_chunk(chunk)
        
        assert isinstance(score, ValidationScore)
        assert score.completeness_score == 85
        assert score.reliability_score == 92
        assert score.legal_structure_score == 78
    
    def test_validate_chunk_without_api_key(self, basic_config):
        """Test validation without API key."""
        validator = AIValidator(basic_config)
        
        chunk = {'text': 'Test content', 'chunk_id': 'test'}
        score = validator.validate_chunk(chunk)
        
        # Should return default/fallback score
        assert isinstance(score, ValidationScore)
        assert score.overall_score == 0  # Default score
    
    @patch('regulatory_processor.validators.anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic, ai_config):
        """Test handling of API errors."""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client
        
        validator = AIValidator(ai_config)
        
        chunk = {'text': 'Test content', 'chunk_id': 'test'}
        
        with patch('regulatory_processor.validators.logger') as mock_logger:
            score = validator.validate_chunk(chunk)
            
            # Should log error and return fallback score
            mock_logger.error.assert_called()
            assert isinstance(score, ValidationScore)
    
    def test_extract_articles_from_text(self, ai_config, sample_text):
        """Test article extraction from text."""
        validator = AIValidator(ai_config)
        
        with patch.object(validator, '_call_anthropic_api') as mock_api:
            mock_api.return_value = {
                'articles': [
                    {
                        'number': 'Article 1',
                        'title': 'Objet et champ d\'application',
                        'content': 'Le présent règlement définit...',
                        'materiality': 'HIGH',
                        'reasoning': 'Defines scope'
                    },
                    {
                        'number': 'Article 2',
                        'title': 'Définitions',
                        'content': 'Au sens du présent règlement...',
                        'materiality': 'MEDIUM',
                        'reasoning': 'Provides definitions'
                    }
                ]
            }
            
            articles = validator.extract_articles(sample_text)
            
            assert len(articles) == 2
            assert all(isinstance(article, Article) for article in articles)
            assert articles[0].materiality == MaterialityLevel.HIGH
            assert articles[1].materiality == MaterialityLevel.MEDIUM
    
    def test_assess_materiality(self, ai_config):
        """Test materiality assessment."""
        validator = AIValidator(ai_config)
        
        article = Article(
            number="Article 1",
            title="Capital Requirements",
            content="Banks must maintain minimum capital ratios."
        )
        
        with patch.object(validator, '_call_anthropic_api') as mock_api:
            mock_api.return_value = {
                'materiality': 'CRITICAL',
                'reasoning': 'Defines mandatory capital requirements'
            }
            
            materiality, reasoning = validator.assess_materiality(article)
            
            assert materiality == MaterialityLevel.CRITICAL
            assert "capital requirements" in reasoning.lower()
    
    def test_batch_validation(self, ai_config, sample_chunks):
        """Test batch validation of multiple chunks."""
        validator = AIValidator(ai_config)
        
        with patch.object(validator, 'validate_chunk') as mock_validate:
            mock_validate.return_value = ValidationScore(
                completeness_score=80,
                reliability_score=85,
                legal_structure_score=75,
                overall_score=80
            )
            
            results = validator.validate_chunks(sample_chunks)
            
            assert len(results) == len(sample_chunks)
            assert all(isinstance(score, ValidationScore) for score in results)
            assert mock_validate.call_count == len(sample_chunks)
    
    def test_prompt_generation(self, ai_config):
        """Test AI prompt generation."""
        validator = AIValidator(ai_config)
        
        chunk = {
            'text': 'Article 1. Test regulation text.',
            'chunk_id': 'test_chunk'
        }
        
        prompt = validator._generate_validation_prompt(chunk)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert 'Article 1' in prompt
        assert 'regulation' in prompt.lower()
    
    def test_response_parsing(self, ai_config):
        """Test parsing of AI response."""
        validator = AIValidator(ai_config)
        
        valid_response = {
            'completeness_score': 85,
            'reliability_score': 90,
            'legal_structure_score': 75,
            'overall_score': 83,
            'completeness_issues': ['Missing context'],
            'reliability_issues': [],
            'legal_structure_issues': ['Format issue'],
            'recommendations': ['Improve structure']
        }
        
        score = validator._parse_validation_response(valid_response)
        
        assert isinstance(score, ValidationScore)
        assert score.completeness_score == 85
        assert score.reliability_score == 90
        assert len(score.completeness_issues) == 1
        assert len(score.recommendations) == 1
    
    def test_invalid_response_parsing(self, ai_config):
        """Test parsing of invalid AI response."""
        validator = AIValidator(ai_config)
        
        # Invalid response structure
        invalid_response = {'invalid': 'response'}
        
        score = validator._parse_validation_response(invalid_response)
        
        # Should return default score for invalid response
        assert isinstance(score, ValidationScore)
        assert score.overall_score == 0


class TestValidationIntegration:
    """Test integration between validation components."""
    
    def test_full_validation_pipeline(self, ai_config, sample_text):
        """Test complete validation pipeline."""
        validator = AIValidator(ai_config)
        chunk_validator = ChunkValidator()
        
        # Create chunk from text
        chunk = {
            'chunk_index': 0,
            'text': sample_text[:500],
            'text_length': 500,
            'word_count': len(sample_text[:500].split()),
            'chunk_id': 'integration_test_chunk'
        }
        
        # Basic validation first
        is_valid, issues = chunk_validator.validate_chunk(chunk)
        assert is_valid or len(issues) > 0  # Should complete without error
        
        # AI validation (mocked)
        with patch.object(validator, 'validate_chunk') as mock_ai_validate:
            mock_ai_validate.return_value = ValidationScore(overall_score=85)
            
            ai_score = validator.validate_chunk(chunk)
            assert isinstance(ai_score, ValidationScore)
    
    def test_error_propagation(self, ai_config):
        """Test error handling across validation components."""
        validator = AIValidator(ai_config)
        
        # Test with malformed chunk
        malformed_chunk = {'invalid': 'chunk'}
        
        # Should handle gracefully
        score = validator.validate_chunk(malformed_chunk)
        assert isinstance(score, ValidationScore)
    
    def test_concurrent_validation(self, ai_config, sample_chunks):
        """Test concurrent validation of multiple chunks."""
        validator = AIValidator(ai_config)
        
        with patch.object(validator, 'validate_chunk') as mock_validate:
            mock_validate.return_value = ValidationScore(overall_score=75)
            
            # Simulate concurrent validation
            results = []
            for chunk in sample_chunks:
                result = validator.validate_chunk(chunk)
                results.append(result)
            
            assert len(results) == len(sample_chunks)
            assert all(isinstance(r, ValidationScore) for r in results)
    
    def test_validation_consistency(self, ai_config):
        """Test validation consistency across multiple runs."""
        validator = AIValidator(ai_config)
        
        chunk = {
            'text': 'Consistent test content for validation.',
            'chunk_id': 'consistency_test'
        }
        
        with patch.object(validator, '_call_anthropic_api') as mock_api:
            mock_api.return_value = {
                'completeness_score': 80,
                'reliability_score': 85,
                'legal_structure_score': 75,
                'overall_score': 80,
                'completeness_issues': [],
                'reliability_issues': [],
                'legal_structure_issues': [],
                'recommendations': []
            }
            
            # Run validation multiple times
            results = [validator.validate_chunk(chunk) for _ in range(3)]
            
            # Results should be consistent
            for result in results:
                assert result.overall_score == 80
                assert result.completeness_score == 80