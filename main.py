"""
CLI entry point for the PDF parser.

Usage:
    python main.py <pdf_path> [--output <dir>] [--ocr] [--pretty]

Examples:
    `python main.py report.pdf`
    python main.py report.pdf --output results/ --ocr
    python main.py report.pdf --pretty
"""

from __future__ import annotations

import argparse
import json
import sys

from pdf_parser import PDFParser


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured content from PDFs into semantic JSON chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py document.pdf
  python main.py document.pdf --output results/ --ocr
  python main.py document.pdf --pretty
        """,
    )

    parser.add_argument(
        "pdf",
        help="Path to the PDF file to parse",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for JSON and extracted images (default: output/)",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        default=True,
        help="Run OCR (pytesseract) on extracted images (default: True)",
    )
    parser.add_argument(
        "--no-ocr",
        dest="ocr",
        action="store_false",
        help="Disable OCR on extracted images",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print the JSON output to stdout",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=50,
        help="Minimum width/height in pixels for extracted images (default: 50)",
    )
    parser.add_argument(
        "--json-file",
        default="parsed_output.json",
        help="Output JSON filename (default: parsed_output.json)",
    )

    args = parser.parse_args()


    print("=" * 60)
    print("  PDF Parser - Structured Content Extraction")
    print("=" * 60)
    print()

    pdf_parser = PDFParser(
        pdf_path=args.pdf,
        output_dir=args.output,
        ocr_images=args.ocr,
        min_image_width=args.min_image_size,
        min_image_height=args.min_image_size,
    )

    chunks = pdf_parser.parse()


    output_path = pdf_parser.save_json(args.json_file)


    summary = pdf_parser.summary()
    print()
    print("-" * 40)
    print("  Summary")
    print("-" * 40)
    print(f"  PDF:          {summary['pdf']}")
    print(f"  Pages:        {summary['total_pages']}")
    print(f"  Chunks:       {summary['total_chunks']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  By type:")
    for t, count in summary["by_type"].items():
        print(f"    {t:12s}: {count}")
    print("-" * 40)


    if args.pretty:
        print()
        print(pdf_parser.to_json(indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
