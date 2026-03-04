"""
Text & heading extraction using PyMuPDF (fitz).

Extracts text blocks with position info and classifies them as
headings or paragraphs using font-size heuristics and layout analysis.
"""

from __future__ import annotations

import fitz  # PyMuPDF
from pdf_parser.models import PageLayout
from pdf_parser.utils import clean_text, bbox_to_list


def extract_text_blocks(doc: fitz.Document) -> list[PageLayout]:
    """
    Extract all text blocks from every page, enriched with font metadata.

    Each text block dict contains:
        - text: cleaned string
        - bbox: [x0, y0, x1, y1]
        - font_size: dominant font size in this block
        - is_bold: whether the dominant font is bold
        - flags: raw font flags
        - block_type: "text" (always, for this extractor)
    """
    pages: list[PageLayout] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        rect = page.rect
        layout = PageLayout(
            page_number=page_idx + 1,
            width=round(rect.width, 1),
            height=round(rect.height, 1),
        )

        # Use "dict" extraction for rich font metadata
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in page_dict.get("blocks", []):
            # skip image blocks (type == 1)
            if block.get("type", 0) != 0:
                continue

            block_text_parts = []
            font_sizes = []
            bold_count = 0
            total_spans = 0

            for line in block.get("lines", []):
                line_text_parts = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if span_text.strip():
                        line_text_parts.append(span_text)
                        font_sizes.append(span["size"])
                        total_spans += 1
                        # check bold via flags (bit 4) or font name
                        font_name = span.get("font", "").lower()
                        flags = span.get("flags", 0)
                        if (flags & (1 << 4)) or "bold" in font_name:
                            bold_count += 1

                if line_text_parts:
                    block_text_parts.append("".join(line_text_parts))

            full_text = "\n".join(block_text_parts)
            cleaned = clean_text(full_text)
            if not cleaned:
                continue

            # Dominant font size = most common size (simple mode)
            dominant_size = max(set(font_sizes), key=font_sizes.count) if font_sizes else 12.0
            is_bold = (bold_count > total_spans / 2) if total_spans > 0 else False

            layout.text_blocks.append({
                "text": cleaned,
                "bbox": bbox_to_list(block.get("bbox")),
                "font_size": round(dominant_size, 1),
                "is_bold": is_bold,
                "block_type": "text",
            })

        pages.append(layout)

    return pages


def classify_headings(
    pages: list[PageLayout],
    body_size_threshold: float | None = None,
) -> list[PageLayout]:
    """
    Classify text blocks as 'heading' or 'paragraph' based on font size.

    If `body_size_threshold` is not given, we auto-detect the most common
    font size across the entire document as the "body" size, and treat
    anything significantly larger (or bold + larger) as a heading.
    """
    if body_size_threshold is None:
        body_size_threshold = _detect_body_font_size(pages)

    heading_min = body_size_threshold + 1.5  # at least 1.5pt larger

    for layout in pages:
        for block in layout.text_blocks:
            size = block.get("font_size", 12.0)
            is_bold = block.get("is_bold", False)
            text = block.get("text", "")

            is_heading = False

            # Rule 1: clearly larger font
            if size >= heading_min:
                is_heading = True

            # Rule 2: bold + somewhat larger + short
            if is_bold and size >= body_size_threshold + 0.5 and len(text) < 120:
                is_heading = True

            # Rule 3: ALL CAPS short text (likely a section header)
            if text.isupper() and 3 < len(text) < 80 and "\n" not in text:
                is_heading = True

            block["block_type"] = "heading" if is_heading else "paragraph"

    return pages


def _detect_body_font_size(pages: list[PageLayout]) -> float:
    """Find the most frequently used font size across all text blocks."""
    from collections import Counter
    sizes = Counter()
    for layout in pages:
        for block in layout.text_blocks:
            # weight by text length so body text dominates
            weight = len(block.get("text", ""))
            size = block.get("font_size", 12.0)
            # round to nearest 0.5 to group similar sizes
            rounded = round(size * 2) / 2
            sizes[rounded] += weight

    if not sizes:
        return 12.0
    return sizes.most_common(1)[0][0]
