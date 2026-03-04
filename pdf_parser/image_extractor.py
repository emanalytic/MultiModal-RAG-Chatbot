"""
Image extraction using PyMuPDF with optional OCR via pytesseract.

Extracts embedded images, saves them to disk, and optionally runs
OCR to get text content from figures.
"""

from __future__ import annotations

import os
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import hashlib
import io

from pdf_parser.models import PageLayout
from pdf_parser.utils import bbox_to_list


def extract_images(
    doc: fitz.Document,
    pages: list[PageLayout],
    output_dir: str,
    ocr: bool = False,
    min_width: int = 50,
    min_height: int = 50,
) -> list[PageLayout]:
    """
    Extract images from the PDF and attach them to PageLayout objects.

    Args:
        doc: opened PyMuPDF document
        pages: list of PageLayout objects (one per page)
        output_dir: directory to save extracted images
        ocr: whether to run pytesseract OCR on extracted images
        min_width: minimum image width to keep (filters out tiny icons)
        min_height: minimum image height to keep

    Each image dict contains:
        - filename: saved image filename
        - filepath: full path to the saved image
        - bbox: [x0, y0, x1, y1] or None
        - ocr_text: OCR text if ocr=True and text was found
        - width: image width in pixels
        - height: image height in pixels
        - block_type: "image"
    """
    os.makedirs(output_dir, exist_ok=True)
    img_counter = 0
    seen_hashes: set[str] = set()

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        layout = _find_layout(pages, page_idx + 1)
        if layout is None:
            continue

        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            if not base_image:
                continue

            img_bytes = base_image["image"]

            # Deduplicate by content hash
            img_hash = hashlib.md5(img_bytes).hexdigest()
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            img_ext = base_image.get("ext", "png")
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            # skip tiny images (icons, bullets, etc.)
            if width < min_width or height < min_height:
                continue

            img_counter += 1
            filename = f"image_p{page_idx + 1}_{img_counter:03d}.{img_ext}"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "wb") as f:
                f.write(img_bytes)

            # try to find the image's position on the page
            bbox = _find_image_bbox(page, xref)

            image_entry = {
                "filename": filename,
                "filepath": filepath,
                "bbox": bbox_to_list(bbox) if bbox else None,
                "width": width,
                "height": height,
                "block_type": "image",
                "ocr_text": None,
            }

            # optional OCR
            if ocr:
                image_entry["ocr_text"] = _run_ocr(img_bytes)

            layout.images.append(image_entry)

    return pages


def _find_layout(pages: list[PageLayout], page_number: int) -> PageLayout | None:
    for layout in pages:
        if layout.page_number == page_number:
            return layout
    return None


def _find_image_bbox(page: fitz.Page, xref: int):
    """Try to locate the bounding box of an image on its page by xref."""
    try:
        for img_rect in page.get_image_rects(xref):
            return img_rect
    except Exception:
        pass
    return None


def _run_ocr(img_bytes: bytes) -> str | None:
    """Run pytesseract OCR on image bytes. Returns text or None."""
    try:
        import pytesseract
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img).strip()
        return text if text else None
    except Exception:
        return None
