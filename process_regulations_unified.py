#!/usr/bin/env python3
"""
Unified Regulatory Document Processor
Works on Windows, Linux, and macOS with both technical and client-friendly output options.
"""

import os
import sys
import argparse
from datetime import datetime
from regulatory_processor import RegulatoryDocumentProcessor, ProcessorConfig


def main():
    """Main function with command-line argument support."""
    
    parser = argparse.ArgumentParser(
        description="Process regulatory documents with AI validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_regulations_unified.py                          # Process all documents (client format)
  python process_regulations_unified.py --technical              # Generate technical report
  python process_regulations_unified.py --cartographie           # Generate regulatory cartography mapping
  python process_regulations_unified.py --both                   # Generate both client and technical reports
  python process_regulations_unified.py --all                    # Generate all report formats
  python process_regulations_unified.py --single document.pdf    # Process single file
  python process_regulations_unified.py --single doc.pdf --pages "1-10,15,20-25"  # Process specific pages
  python process_regulations_unified.py --no-ai                  # Disable AI features
  python process_regulations_unified.py --user "John Doe"        # Add user info to report
        """
    )
    
    parser.add_argument('--client', action='store_true', 
                       help='Generate client-friendly Excel report (default)')
    parser.add_argument('--technical', action='store_true', 
                       help='Generate technical Excel report (detailed format)')
    parser.add_argument('--cartographie', action='store_true', 
                       help='Generate regulatory cartography mapping worksheet')
    parser.add_argument('--both', action='store_true', 
                       help='Generate both client and technical reports')
    parser.add_argument('--all', action='store_true', 
                       help='Generate all report formats (client, technical, cartographie)')
    parser.add_argument('--single', type=str, metavar='FILE',
                       help='Process single PDF file instead of all documents')
    parser.add_argument('--no-ai', action='store_true',
                       help='Disable AI validation and article extraction')
    parser.add_argument('--user', type=str, metavar='NAME',
                       help='User name for report (format: "First Last")')
    parser.add_argument('--output-dir', type=str, metavar='DIR',
                       help='Output directory (default: current directory)')
    parser.add_argument('--model', type=str, default='claude-opus-4-20250514',
                       help='AI model to use (default: claude-opus-4-20250514, options: claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229, claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-opus-4-20250514)')
    parser.add_argument('--pages', type=str, metavar='RANGES',
                       help='Page ranges to process (e.g., "1-10,15,20-25"). Works with --single only.')
    
    args = parser.parse_args()
    
    # Set defaults if no format specified
    if not any([args.client, args.technical, args.cartographie, args.both, args.all]):
        args.client = True  # Default to client-friendly
    
    # Validate pages argument
    if args.pages and not args.single:
        print("Error: --pages can only be used with --single")
        return 1
    
    if args.pages:
        try:
            from regulatory_processor.page_selector import PageRangeParser
            parser = PageRangeParser()
            parser.parse_range_string(args.pages)
        except (ImportError, ValueError) as e:
            print(f"Error: Invalid page ranges '{args.pages}': {e}")
            return 1
    
    # Use current directory as base path (cross-platform)
    base_path = os.getcwd()
    output_dir = args.output_dir or base_path
    
    # Parse user info
    user_info = {}
    if args.user:
        parts = args.user.strip().split()
        if len(parts) >= 2:
            user_info['first_name'] = parts[0]
            user_info['last_name'] = ' '.join(parts[1:])
        else:
            user_info['first_name'] = parts[0]
            user_info['last_name'] = ''
    
    # Configure processor
    config = ProcessorConfig(
        chunk_size=1000,
        chunk_overlap=200,
        log_level="INFO",
        clean_text=True,
        excel_max_cell_length=32767,
        enable_ai_validation=not args.no_ai,
        extract_articles=not args.no_ai,
        assess_materiality=not args.no_ai,
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
        ai_model=args.model
    )
    
    processor = RegulatoryDocumentProcessor(config)
    
    # Print header
    print("=" * 80)
    print("JABE REGULATORY DOCUMENT PROCESSOR")
    print("=" * 80)
    print(f"Base directory: {base_path}")
    print(f"Output directory: {output_dir}")
    print(f"AI Analysis: {'ENABLED' if config.enable_ai_validation else 'DISABLED'}")
    if config.enable_ai_validation:
        print(f"AI Model: {config.ai_model}")
    
    if args.client or (not args.technical and not args.cartographie and not args.both and not args.all):
        print("Output Format: Client-Friendly Excel")
    if args.technical:
        print("Output Format: Technical Excel")
    if args.cartographie:
        print("Output Format: Regulatory Cartography Mapping")
    if args.both:
        print("Output Format: Both Client & Technical Excel")
    if args.all:
        print("Output Format: All Formats (Client, Technical, Cartographie)")
    print("=" * 80)
    
    # Check API key if AI is enabled
    if config.enable_ai_validation and not config.anthropic_api_key:
        print("\nWARNING: AI features enabled but no API key found.")
        print("   Set ANTHROPIC_API_KEY environment variable for full analysis.")
        print("   Example: set ANTHROPIC_API_KEY=your-api-key-here")
        print("   Continuing with basic processing...\n")
    
    # Process documents
    processed_count = 0
    
    if args.single:
        # Process single file
        if not os.path.exists(args.single):
            print(f"Error: File not found - {args.single}")
            return 1
        
        if args.pages:
            print(f"\nProcessing single file: {os.path.basename(args.single)} (pages: {args.pages})")
            if processor.process_document_selective(args.single, args.pages):
                processed_count = 1
        else:
            print(f"\nProcessing single file: {os.path.basename(args.single)}")
            if processor.process_document(args.single):
                processed_count = 1
    else:
        # Process all documents
        print("\nScanning for documents...")
        
        # Individual PDF
        pdf_file = os.path.join(base_path, "Code Pénal Gabonais.pdf")
        if os.path.exists(pdf_file):
            print(f"Processing: {os.path.basename(pdf_file)}")
            if processor.process_document(pdf_file):
                processed_count += 1
        
        # Process directories
        directories = [
            ("COBAC Instructions", "Instruction_COBAC"),
            ("COBAC Regulations", "Reglement_COBAC")
        ]
        
        for dir_name, dir_path in directories:
            full_path = os.path.join(base_path, dir_path)
            if os.path.exists(full_path):
                print(f"\nProcessing {dir_name}...")
                results = processor.process_directory(full_path)
                if results:
                    processed_count += len(results)
                    print(f"   Processed {len(results)} documents")
            else:
                print(f"   Directory not found: {dir_path}")
    
    if processed_count == 0:
        print("\nNo documents were successfully processed.")
        return 1
    
    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_generated = []
    
    print(f"\nGenerating reports... ({processed_count} documents processed)")
    
    try:
        if args.client or (not args.technical and not args.cartographie and not args.both and not args.all):
            # Client-friendly report (default)
            client_file = f"compliance_report_client_{timestamp}.xlsx"
            client_path = os.path.join(output_dir, client_file)
            print(f"Creating client report: {client_file}")
            processor.export_client_report(client_path, user_info=user_info)
            reports_generated.append(('Client Report', client_path))
        
        if args.technical or args.both or args.all:
            # Technical report
            technical_file = f"compliance_report_technical_{timestamp}.xlsx"
            technical_path = os.path.join(output_dir, technical_file)
            print(f"Creating technical report: {technical_file}")
            processor.export_results(
                technical_path,
                include_validation=config.enable_ai_validation,
                include_articles=config.extract_articles,
                include_full_text=True,
                user_info=user_info
            )
            reports_generated.append(('Technical Report', technical_path))
        
        if args.cartographie or args.all:
            # Regulatory cartography mapping
            cartographie_file = f"cartographie_reglementaire_{timestamp}.xlsx"
            cartographie_path = os.path.join(output_dir, cartographie_file)
            print(f"Creating regulatory cartography: {cartographie_file}")
            processor.export_cartographie_reglementaire(cartographie_path, user_info=user_info)
            reports_generated.append(('Regulatory Cartography', cartographie_path))
    
    except Exception as e:
        print(f"Error generating reports: {e}")
        return 1
    
    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    
    summary = processor.get_summary()
    if summary and summary.get('status') != 'No documents processed':
        total_docs = len(processor.processed_documents) + len(processor.processing_errors)
        successful_docs = len(processor.processed_documents)
        
        print(f"Documents analyzed: {total_docs}")
        print(f"Successfully processed: {successful_docs}")
        
        if len(processor.processing_errors) > 0:
            print(f"Failed: {len(processor.processing_errors)}")
        
        # Count articles and priorities if AI was used
        if config.enable_ai_validation:
            total_articles = 0
            priority_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            
            for doc in processor.processed_documents:
                articles = doc.get('articles', [])
                total_articles += len(articles)
                
                for article in articles:
                    materiality = article.get('materiality', 'MEDIUM')
                    if materiality in priority_counts:
                        priority_counts[materiality] += 1
            
            if total_articles > 0:
                print(f"\nCompliance items identified: {total_articles}")
                print("Priority breakdown:")
                print(f"   CRITICAL: {priority_counts['CRITICAL']} items")
                print(f"   HIGH: {priority_counts['HIGH']} items")
                print(f"   MEDIUM: {priority_counts['MEDIUM']} items")
                print(f"   LOW: {priority_counts['LOW']} items")
    
    print(f"\nReports generated:")
    for report_name, report_path in reports_generated:
        print(f"   • {report_name}: {os.path.basename(report_path)}")
    
    print(f"\nTip: Open the Excel file(s) to view the analysis results.")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nProcessing cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)