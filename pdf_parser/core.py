"""
Core PDF parser — orchestrates all extraction and chunking steps.

Usage:
    from pdf_parser import PDFParser

    parser = PDFParser("document.pdf", output_dir="output/")
    chunks = parser.parse()
    parser.save_json("result.json")
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import fitz  # PyMuPDF

from pdf_parser.models import ContentElement, PageLayout
from pdf_parser.text_extractor import extract_text_blocks, classify_headings
from pdf_parser.table_extractor import extract_tables
from pdf_parser.image_extractor import extract_images
from pdf_parser.chunker import build_chunks


class PDFParser:
    """
    Main entry point for structured PDF content extraction.

    Orchestrates:
        1. Text & heading extraction (PyMuPDF)
        2. Table extraction (pdfplumber)
        3. Image extraction (PyMuPDF + optional OCR)
        4. Semantic chunking & assembly
    """

    def __init__(
        self,
        pdf_path: str,
        output_dir: str = "output",
        ocr_images: bool = True,
        min_image_width: int = 50,
        min_image_height: int = 50,
    ):
        """
        Args:
            pdf_path: path to the PDF file
            output_dir: directory for extracted images and output files
            ocr_images: whether to run OCR on extracted images
            min_image_width: minimum width to keep an image (filters icons)
            min_image_height: minimum height to keep an image
        """
        self.pdf_path = os.path.abspath(pdf_path)
        self.output_dir = os.path.abspath(output_dir)
        self.ocr_images = ocr_images
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height

        if not os.path.isfile(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        self.chunks: list[ContentElement] = []
        self._pages: list[PageLayout] = []

    def parse(self) -> list[ContentElement]:
        """
        Run the full extraction pipeline and return structured chunks.

        Pipeline:
            PDF → Text blocks → Heading classification → Table extraction
                → Image extraction → Semantic chunking → ContentElement list
        """
        print(f"[1/5] Opening PDF: {self.pdf_path}")
        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)
        print(f"      {total_pages} page(s) detected")


        print("[2/5] Extracting text blocks & detecting headings...")
        self._pages = extract_text_blocks(doc)
        self._pages = classify_headings(self._pages)

        text_count = sum(len(p.text_blocks) for p in self._pages)
        heading_count = sum(
            1 for p in self._pages
            for b in p.text_blocks if b.get("block_type") == "heading"
        )
        print(f"      {text_count} text blocks, {heading_count} headings found")


        print("[3/5] Extracting tables...")
        self._pages = extract_tables(self.pdf_path, self._pages, doc=doc)
        table_count = sum(len(p.tables) for p in self._pages)
        print(f"      {table_count} table(s) found")


        images_dir = os.path.join(self.output_dir, "images")
        print(f"[4/5] Extracting images (OCR={'on' if self.ocr_images else 'off'})...")
        self._pages = extract_images(
            doc, self._pages, images_dir,
            ocr=self.ocr_images,
            min_width=self.min_image_width,
            min_height=self.min_image_height,
        )
        image_count = sum(len(p.images) for p in self._pages)
        print(f"      {image_count} image(s) extracted")

        doc.close()


        print("[5/5] Building semantic chunks...")
        self.chunks = build_chunks(self._pages)
        print(f"      {len(self.chunks)} chunk(s) created")

        return self.chunks

    def to_dicts(self) -> list[dict]:
        """Return chunks as a list of clean dictionaries."""
        return [chunk.to_dict() for chunk in self.chunks]

    def to_json(self, indent: int = 2) -> str:
        """Return chunks as a JSON string."""
        return json.dumps(self.to_dicts(), indent=indent, ensure_ascii=False)

    def save_json(self, filename: str = "parsed_output.json") -> str:
        """Save chunks to a JSON file and return the filepath."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        print(f"\nSaved {len(self.chunks)} chunks -> {filepath}")
        return filepath

    def summary(self) -> dict:
        """Return a summary of the extraction results."""
        type_counts = {}
        total_tokens = 0
        for chunk in self.chunks:
            type_counts[chunk.type] = type_counts.get(chunk.type, 0) + 1
            total_tokens += chunk.tokens

        return {
            "pdf": os.path.basename(self.pdf_path),
            "total_pages": len(self._pages),
            "total_chunks": len(self.chunks),
            "total_tokens": total_tokens,
            "by_type": type_counts,
        }
