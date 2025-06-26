#!/usr/bin/env python3
"""Test script to verify PDF processing works correctly."""

import os
import sys
import locale
from datetime import datetime
from regulatory_processor import RegulatoryDocumentProcessor, ProcessorConfig

# Force UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

def test_processing():
    """Test basic processing functionality."""
    print("=== Test de traitement des documents ===\n")
    
    # Configuration
    config = ProcessorConfig(
        chunk_size=1000,
        chunk_overlap=200,
        enable_ai_validation=False,  # Disable AI to test basic functionality
        extract_articles=True,
        clean_text=True,
        log_level="INFO"
    )
    
    # Create processor
    processor = RegulatoryDocumentProcessor(config)
    
    # Test with a single PDF
    test_pdf = "Reglement_COBAC/R-2001_03.pdf"
    if not os.path.exists(test_pdf):
        print(f"Fichier de test non trouvé: {test_pdf}")
        return False
    
    print(f"Traitement de: {test_pdf}")
    result = processor.process_document(test_pdf)
    
    if result:
        print(f"✓ Document traité avec succès")
        print(f"  - Chunks: {result['statistics']['total_chunks']}")
        print(f"  - Articles: {len(result.get('articles', []))}")
        print(f"  - Mots: {result['statistics']['total_words']}")
    else:
        print(f"✗ Échec du traitement")
        return False
    
    # Test export
    output_dir = "resultat"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"test_export_{timestamp}.xlsx")
    
    print(f"\nExport vers: {output_file}")
    processor.export_results(output_file, format='excel')
    
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"✓ Fichier Excel créé: {file_size:,} octets")
        return True
    else:
        print(f"✗ Fichier Excel non créé")
        return False

if __name__ == "__main__":
    success = test_processing()
    print(f"\n{'SUCCÈS' if success else 'ÉCHEC'}")
    sys.exit(0 if success else 1)