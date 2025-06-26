#!/usr/bin/env python3
"""
Interactive demonstration of the regulatory document processor.
This script shows you exactly how each component works.
"""

import os
import sys
from datetime import datetime
from regulatory_processor import RegulatoryDocumentProcessor, ProcessorConfig


def demo_basic_processing():
    """Demonstrate basic processing without AI."""
    print("🔥 DEMO: Basic Document Processing")
    print("=" * 60)
    
    # Step 1: Configuration
    print("📝 Step 1: Creating basic configuration...")
    config = ProcessorConfig(
        chunk_size=500,        # Smaller for demo
        chunk_overlap=100,
        log_level="INFO",
        clean_text=True,
        enable_ai_validation=False,  # No AI for basic demo
        extract_articles=True        # Still extract articles (regex-based)
    )
    print(f"   ✓ Chunk size: {config.chunk_size} characters")
    print(f"   ✓ Overlap: {config.chunk_overlap} characters")
    print(f"   ✓ AI validation: {'Enabled' if config.enable_ai_validation else 'Disabled'}")
    
    # Step 2: Initialize processor
    print("\n🏭 Step 2: Initializing processor...")
    processor = RegulatoryDocumentProcessor(config)
    print("   ✓ PDF extractor ready")
    print("   ✓ Text chunker ready")
    print("   ✓ Excel exporter ready")
    
    # Step 3: Find a sample file
    print("\n🔍 Step 3: Finding sample document...")
    base_path = "/mnt/c/Users/doupa/Desktop/Ventures/Orabank"
    sample_file = None
    
    # Look for a small regulation file
    reg_dir = os.path.join(base_path, "Reglement_COBAC")
    if os.path.exists(reg_dir):
        pdf_files = [f for f in os.listdir(reg_dir) if f.endswith('.pdf')]
        if pdf_files:
            sample_file = os.path.join(reg_dir, pdf_files[0])
            print(f"   ✓ Found sample: {pdf_files[0]}")
    
    if not sample_file:
        print("   ❌ No PDF files found for demo")
        return
    
    # Step 4: Process the document
    print(f"\n⚙️  Step 4: Processing document...")
    print(f"   Processing: {os.path.basename(sample_file)}")
    
    try:
        result = processor.process_document(sample_file)
        
        if result:
            print("   ✓ Document processed successfully!")
            print(f"   ✓ Extracted {result['statistics']['total_words']:,} words")
            print(f"   ✓ Created {result['statistics']['total_chunks']} chunks")
            print(f"   ✓ Found {len(result.get('articles', []))} articles")
            
            # Show some sample chunks
            print("\n📄 Sample chunks:")
            for i, chunk in enumerate(result['chunks'][:2]):
                print(f"   Chunk {i+1}: {chunk['text'][:100]}...")
            
            # Show some articles
            if result.get('articles'):
                print("\n📋 Sample articles:")
                for i, article in enumerate(result['articles'][:2]):
                    print(f"   Article {article.number}: {article.content[:80]}...")
                    print(f"      Materiality: {article.materiality.value}")
        
        else:
            print("   ❌ Processing failed")
            return
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Step 5: Export results
    print(f"\n📊 Step 5: Exporting results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(base_path, f"demo_basic_{timestamp}.xlsx")
    
    processor.export_results(output_file, include_articles=True)
    print(f"   ✓ Results exported to: {os.path.basename(output_file)}")
    
    print("\n🎉 Basic demo complete!")
    print(f"📁 Check your file: {output_file}")


def demo_ai_processing():
    """Demonstrate AI-enhanced processing."""
    print("\n🤖 DEMO: AI-Enhanced Processing")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("⚠️  No ANTHROPIC_API_KEY found.")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("   Continuing with fallback validation...")
    
    # Step 1: AI Configuration
    print("📝 Step 1: Creating AI configuration...")
    config = ProcessorConfig(
        chunk_size=800,
        chunk_overlap=150,
        enable_ai_validation=True,
        extract_articles=True,
        assess_materiality=True,
        anthropic_api_key=api_key,
        ai_model="claude-3-haiku-20240307"
    )
    
    print(f"   ✓ AI validation: Enabled")
    print(f"   ✓ Article extraction: Enabled")
    print(f"   ✓ Materiality assessment: Enabled")
    print(f"   ✓ Model: {config.ai_model}")
    
    # Step 2: Initialize with AI
    print("\n🏭 Step 2: Initializing AI processor...")
    processor = RegulatoryDocumentProcessor(config)
    if processor.validator:
        print("   ✓ AI validator initialized")
        print("   ✓ Validation chain ready")
    else:
        print("   ⚠️  AI validator not available (using fallback)")
    
    # Step 3: Process with AI
    print("\n⚙️  Step 3: Processing with AI validation...")
    base_path = "/mnt/c/Users/doupa/Desktop/Ventures/Orabank"
    
    # Find a regulation file
    reg_dir = os.path.join(base_path, "Reglement_COBAC")
    sample_file = None
    
    if os.path.exists(reg_dir):
        pdf_files = [f for f in os.listdir(reg_dir) if f.endswith('.pdf')]
        if pdf_files:
            # Try to find a file with "R-2018" or similar recent regulation
            for file in pdf_files:
                if any(year in file for year in ['2018', '2019', '2020']):
                    sample_file = os.path.join(reg_dir, file)
                    break
            if not sample_file:
                sample_file = os.path.join(reg_dir, pdf_files[0])
    
    if not sample_file:
        print("   ❌ No sample file found")
        return
    
    print(f"   Processing: {os.path.basename(sample_file)}")
    
    try:
        result = processor.process_document(sample_file)
        
        if result:
            print("   ✓ Document processed with AI validation!")
            
            # Show validation results
            if 'validation_results' in result:
                val_results = result['validation_results']['document_validation']
                print(f"\n🔍 AI Validation Scores:")
                print(f"   Overall Score: {val_results.get('overall_score', 0):.1f}/100")
                
                if 'extraction_quality' in val_results:
                    print(f"   Extraction Quality: {val_results['extraction_quality'].get('score', 0):.1f}/100")
                if 'legal_structure' in val_results:
                    print(f"   Legal Structure: {val_results['legal_structure'].get('score', 0):.1f}/100")
                if 'completeness' in val_results:
                    print(f"   Completeness: {val_results['completeness'].get('score', 0):.1f}/100")
            
            # Show articles with materiality
            articles = result.get('articles', [])
            if articles:
                print(f"\n📋 Articles with AI Materiality Assessment:")
                materiality_counts = {}
                for article in articles[:5]:  # Show first 5
                    level = article.materiality.value
                    materiality_counts[level] = materiality_counts.get(level, 0) + 1
                    print(f"   Article {article.number}: {level}")
                    print(f"      Content: {article.content[:60]}...")
                    print(f"      Reasoning: {article.materiality_reasoning[:80]}...")
                
                print(f"\n📊 Materiality Distribution:")
                for level, count in sorted(materiality_counts.items()):
                    print(f"   {level}: {count} articles")
        
        else:
            print("   ❌ AI processing failed")
            return
    
    except Exception as e:
        print(f"   ❌ Error during AI processing: {e}")
        return
    
    # Step 4: Export AI results
    print(f"\n📊 Step 4: Exporting AI-enhanced results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(base_path, f"demo_ai_{timestamp}.xlsx")
    
    processor.export_results(
        output_file,
        include_validation=True,
        include_articles=True,
        include_full_text=False
    )
    
    print(f"   ✓ AI results exported to: {os.path.basename(output_file)}")
    print("\n🎉 AI demo complete!")
    print(f"📁 Check your enhanced file: {output_file}")
    print("\n📋 Excel sheets generated:")
    print("   - Document_Metadata: Basic file info")
    print("   - Articles: Extracted articles with materiality")
    print("   - Text_Chunks: Chunks with validation scores")
    print("   - Validation_Results: AI quality assessment")
    print("   - Statistics: Processing summary")


def show_file_structure():
    """Show the current file structure."""
    print("\n📁 Current File Structure:")
    print("=" * 60)
    
    base_path = "/mnt/c/Users/doupa/Desktop/Ventures/Orabank"
    
    if os.path.exists(base_path):
        print(f"📂 {base_path}/")
        
        # Show main files
        for item in sorted(os.listdir(base_path)):
            if item.endswith('.pdf'):
                print(f"   📄 {item}")
            elif os.path.isdir(os.path.join(base_path, item)):
                dir_path = os.path.join(base_path, item)
                if item in ['Instruction_COBAC', 'Reglement_COBAC']:
                    pdf_count = len([f for f in os.listdir(dir_path) if f.endswith('.pdf')])
                    print(f"   📂 {item}/ ({pdf_count} PDFs)")
                elif item == 'regulatory_processor':
                    print(f"   📂 {item}/ (Python module)")
        
        # Show any generated files
        generated_files = [f for f in os.listdir(base_path) if f.startswith(('demo_', 'regulatory_'))]
        if generated_files:
            print(f"\n📊 Generated Files:")
            for file in sorted(generated_files):
                print(f"   📊 {file}")


def main():
    """Interactive demo menu."""
    print("🏛️  REGULATORY DOCUMENT PROCESSOR DEMO")
    print("=" * 80)
    
    show_file_structure()
    
    while True:
        print("\n🎯 Choose a demo:")
        print("1. Basic Processing (no AI)")
        print("2. AI-Enhanced Processing")
        print("3. Show File Structure")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            demo_basic_processing()
        elif choice == "2":
            demo_ai_processing()
        elif choice == "3":
            show_file_structure()
        elif choice == "4":
            print("\n👋 Thanks for trying the regulatory processor!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()