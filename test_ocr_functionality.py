#!/usr/bin/env python3
"""Test OCR functionality with the regulatory processor."""

import os
import sys
from regulatory_processor import RegulatoryDocumentProcessor, ProcessorConfig

def test_ocr_extraction():
    """Test OCR extraction on a sample PDF."""
    print("=== Testing OCR Functionality ===\n")
    
    # Configure processor with OCR capabilities
    config = ProcessorConfig(
        chunk_size=1000,
        chunk_overlap=200,
        enable_ai_validation=False,  # Disable AI for this test
        extract_articles=True,
        clean_text=True,
        log_level="INFO"
    )
    
    processor = RegulatoryDocumentProcessor(config)
    
    # Test with a sample PDF
    test_pdfs = [
        "Reglement_COBAC/R-2001_03.pdf",
        "Instruction_COBAC/I-2004_01.pdf"
    ]
    
    for pdf_path in test_pdfs:
        if os.path.exists(pdf_path):
            print(f"\nTesting: {pdf_path}")
            print("-" * 50)
            
            result = processor.process_document(pdf_path)
            
            if result:
                print(f"✓ Document processed successfully")
                print(f"  - Total characters: {result['statistics']['total_characters']:,}")
                print(f"  - Total words: {result['statistics']['total_words']:,}")
                print(f"  - Total chunks: {result['statistics']['total_chunks']}")
                print(f"  - Extraction success: {result['statistics']['extraction_success']}")
                
                # Check if articles were extracted
                articles = result.get('articles', [])
                print(f"  - Articles extracted: {len(articles)}")
                
                if articles:
                    print("\n  First 3 articles found:")
                    for i, article in enumerate(articles[:3]):
                        print(f"    {i+1}. {article.number}: {article.content[:60]}...")
                
                # Sample of extracted text
                sample_text = result.get('full_text', '')[:200]
                print(f"\n  Text sample: {sample_text}...")
                
            else:
                print(f"✗ Failed to process document")
            
            break  # Just test one document
        else:
            print(f"✗ Test PDF not found: {pdf_path}")

if __name__ == "__main__":
    test_ocr_extraction()