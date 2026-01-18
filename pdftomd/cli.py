"""
PDFtoMD CLI - Command line interface for PDF to Markdown conversion.

Usage:
    python -m pdftomd input.pdf [-o output.md] [--model MODEL] [--search]
"""

import argparse
import sys
from pathlib import Path

from pdftomd.extractor import PDFExtractor


def main():
    parser = argparse.ArgumentParser(
        prog="pdftomd",
        description="Extract PDF content to Markdown using Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pdftomd document.pdf
  python -m pdftomd document.pdf -o output.md
  python -m pdftomd document.pdf --model gemini-3-pro-preview --search
  
Environment:
  GOOGLE_API_KEY    Required. Your Google API key for Gemini.
        """
    )
    
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the PDF file to extract"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output Markdown file path (default: input.md)"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gemini-3-flash-preview",
        choices=["gemini-3-flash-preview", "gemini-3-pro-preview"],
        help="Gemini model to use (default: gemini-3-flash-preview)"
    )
    
    parser.add_argument(
        "--search",
        action="store_true",
        help="Enable Google Search for additional context (Gemini 3 only)"
    )
    
    parser.add_argument(
        "--url-context",
        action="store_true",
        help="Enable URL context for web references (Gemini 3 only)"
    )
    
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature 0.0-1.0 (default: 0.1)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output extracted data as JSON instead of Markdown"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"[ERROR] File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    if not args.input.suffix.lower() == ".pdf":
        print(f"[ERROR] Input must be a PDF file: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Set output path
    if args.output is None:
        if args.json:
            args.output = args.input.with_suffix(".json")
        else:
            args.output = args.input.with_suffix(".md")
    
    # Configure logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    try:
        print(f"[START] Extracting: {args.input.name}")
        print(f"[INFO] Model: {args.model}")
        
        extractor = PDFExtractor(
            model=args.model,
            temperature=args.temperature,
            use_search=args.search,
            use_url_context=args.url_context,
        )
        
        # Extract content with token tracking
        result = extractor.extract_with_stats(args.input)
        document = result.document
        
        print(f"[OK] Extracted {len(document.sections)} sections")
        print(f"[OK] Found {len(document.tables)} tables, {len(document.images)} images")
        
        # Show truncation warning if applicable
        if result.was_truncated:
            print(f"[WARN] Response was truncated (finish_reason: {result.finish_reason})")
        
        # Write output
        if args.json:
            output_content = document.model_dump_json(indent=2)
        else:
            output_content = extractor.to_markdown(document)
        
        args.output.write_text(output_content, encoding="utf-8")
        
        print(f"[DONE] Saved to: {args.output}")
        
        # Always show token usage for cost analysis
        tokens = result.token_usage
        print(f"[TOKENS] Input: {tokens.prompt_tokens:,} | Output: {tokens.completion_tokens:,} | Total: {tokens.total_tokens:,}")
        
    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
