"""CLI for pdftomd - PDF to Markdown extractor."""
import argparse
import sys
from pathlib import Path

from . import extract_to_file, extract_with_stats, __version__


def main():
    parser = argparse.ArgumentParser(
        prog="pdftomd",
        description="Extract PDF content to Markdown using Gemini"
    )
    parser.add_argument("pdf", help="PDF file to extract")
    parser.add_argument("-o", "--output", help="Output markdown file (default: same name .md)")
    parser.add_argument("-m", "--model", default="gemini-3-flash-preview", help="Gemini model")
    parser.add_argument("-s", "--stats", action="store_true", help="Show token usage and costs")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[START] Extracting {pdf_path.name}...")
    
    try:
        if args.stats:
            output, result = extract_to_file(pdf_path, args.output, args.model, return_stats=True)
            print(f"[OK] Created: {output}")
            print(f"[OK] Size: {output.stat().st_size:,} bytes")
            print(result.format_stats())
        else:
            output = extract_to_file(pdf_path, args.output, args.model)
            print(f"[OK] Created: {output}")
            print(f"[OK] Size: {output.stat().st_size:,} bytes")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

