"""
Pytest configuration and shared fixtures for JABE test suite.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

# Import the modules we want to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from regulatory_processor.config import ProcessorConfig
from regulatory_processor.processor import RegulatoryDocumentProcessor


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir():
    """Fixture providing temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp(prefix="jabe_test_"))
    yield temp_path
    # Cleanup after all tests
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def basic_config():
    """Fixture providing basic processor configuration."""
    return ProcessorConfig(
        chunk_size=500,  # Smaller for testing
        chunk_overlap=100,
        enable_ai_validation=False,  # Disable AI for unit tests
        extract_articles=True,
        assess_materiality=False,  # Disable for basic tests
        clean_text=True,
        log_level="DEBUG"
    )


@pytest.fixture
def ai_config():
    """Fixture providing AI-enabled configuration."""
    return ProcessorConfig(
        chunk_size=800,
        chunk_overlap=150,
        enable_ai_validation=True,
        extract_articles=True,
        assess_materiality=True,
        clean_text=True,
        anthropic_api_key="test-key-12345",  # Mock key for testing
        log_level="DEBUG"
    )


@pytest.fixture
def mock_processor(basic_config):
    """Fixture providing a processor instance with mocked dependencies."""
    with patch('regulatory_processor.processor.setup_logging'):
        processor = RegulatoryDocumentProcessor(basic_config)
        return processor


@pytest.fixture
def sample_pdf_path(test_data_dir):
    """Fixture providing path to sample PDF for testing."""
    pdf_path = test_data_dir / "sample_regulation.pdf"
    if not pdf_path.exists():
        # Create a minimal PDF for testing if it doesn't exist
        create_sample_pdf(pdf_path)
    return pdf_path


@pytest.fixture
def sample_text():
    """Fixture providing sample regulatory text."""
    return """
    REGLEMENT N°R-2018-01
    
    Article 1. Objet et champ d'application
    Le présent règlement définit les modalités de calcul des ratios prudentiels
    applicables aux établissements de crédit.
    
    Article 2. Définitions
    Au sens du présent règlement, on entend par:
    - Établissement de crédit: toute personne morale agréée pour exercer
      l'activité bancaire dans la zone CEMAC.
    - Ratio de solvabilité: le rapport entre les fonds propres nets et les
      actifs pondérés du risque.
    
    Article 3. Ratio minimum de solvabilité
    Les établissements de crédit doivent maintenir en permanence un ratio
    de solvabilité d'au moins 8%.
    """


@pytest.fixture
def sample_chunks():
    """Fixture providing sample text chunks."""
    return [
        {
            'chunk_index': 0,
            'text': 'Article 1. Objet et champ d\'application. Le présent règlement définit les modalités.',
            'text_length': 85,
            'word_count': 13,
            'chunk_id': 'test_chunk_0'
        },
        {
            'chunk_index': 1,
            'text': 'Article 2. Définitions. Au sens du présent règlement, on entend par établissement de crédit.',
            'text_length': 91,
            'word_count': 15,
            'chunk_id': 'test_chunk_1'
        }
    ]


@pytest.fixture
def sample_articles():
    """Fixture providing sample extracted articles."""
    from regulatory_processor.validators import Article, MaterialityLevel
    
    return [
        Article(
            number="Article 1",
            title="Objet et champ d'application",
            content="Le présent règlement définit les modalités de calcul des ratios prudentiels.",
            materiality=MaterialityLevel.HIGH,
            materiality_reasoning="Defines the scope of prudential ratios",
            context={'regulation_name': 'R-2018-01', 'document_type': 'REGLEMENT'}
        ),
        Article(
            number="Article 2", 
            title="Définitions",
            content="Au sens du présent règlement, on entend par établissement de crédit...",
            materiality=MaterialityLevel.MEDIUM,
            materiality_reasoning="Provides definitions for key terms",
            context={'regulation_name': 'R-2018-01', 'document_type': 'REGLEMENT'}
        )
    ]


@pytest.fixture
def mock_api_response():
    """Fixture providing mock API response for AI validation."""
    return {
        "completeness_score": 85,
        "completeness_issues": [],
        "reliability_score": 92,
        "reliability_issues": ["Minor formatting inconsistencies"],
        "legal_structure_score": 78,
        "legal_structure_issues": [],
        "overall_score": 85,
        "recommendations": ["Review formatting consistency"],
        "chunk_quality": "GOOD"
    }


@pytest.fixture
def user_info():
    """Fixture providing sample user information."""
    return {
        'name': 'Jean',
        'surname': 'Dupont'
    }


def create_sample_pdf(output_path):
    """Create a minimal PDF file for testing."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create PDF
        c = canvas.Canvas(str(output_path), pagesize=letter)
        c.drawString(100, 750, "REGLEMENT TEST")
        c.drawString(100, 720, "Article 1. Test article content.")
        c.drawString(100, 690, "Article 2. Another test article.")
        c.showPage()
        c.save()
        
    except ImportError:
        # Fallback: create a fake PDF-like file for testing
        with open(output_path, 'wb') as f:
            f.write(b'%PDF-1.4\n%Test PDF for JABE testing\n')


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for complete workflows"
    )
    config.addinivalue_line(
        "markers", "gui: GUI automated tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "ai: Tests requiring AI/API access"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than 30 seconds"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "gui" in str(item.fspath):
            item.add_marker(pytest.mark.gui)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Mark AI tests
        if "ai" in item.name.lower() or "anthropic" in item.name.lower():
            item.add_marker(pytest.mark.ai)
        
        # Mark slow tests
        if "slow" in item.name.lower() or "load" in item.name.lower():
            item.add_marker(pytest.mark.slow)