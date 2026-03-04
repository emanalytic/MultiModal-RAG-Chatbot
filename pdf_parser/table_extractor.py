"""
Table extraction using pdfplumber with PyMuPDF fallback.

Strategy:
    1. Try pdfplumber strict line detection first (most reliable)
    2. Fall back to PyMuPDF's built-in table finder (handles drawn rects)
    3. Validate all detected tables to filter false positives
"""

from __future__ import annotations

import fitz  # PyMuPDF
import pdfplumber
from pdf_parser.models import PageLayout
from pdf_parser.utils import bbox_to_list


def extract_tables(
    pdf_path: str,
    pages: list[PageLayout],
    doc: fitz.Document | None = None,
) -> list[PageLayout]:
    """
    Extract tables from the PDF and attach them to the corresponding
    PageLayout objects.

    Uses pdfplumber first, then falls back to PyMuPDF for pages
    where no tables were found.

    Each table dict contains:
        - rows: list[list[str]]  — the table as a 2D array
        - bbox: [x0, y0, x1, y1]
        - block_type: "table"
    """
    pages_with_tables: set[int] = set()

    # ── Pass 1: pdfplumber (strict line-based) ───────────────────
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for plumber_page in pdf.pages:
                page_num = plumber_page.page_number
                layout = _find_layout(pages, page_num)
                if layout is None:
                    continue

                found_tables = plumber_page.find_tables(
                    table_settings={
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                    }
                )

                for table in found_tables:
                    rows = table.extract()
                    if not _is_valid_table(rows):
                        continue

                    cleaned_rows = _clean_rows(rows)
                    cleaned_rows = _strip_paragraph_rows(cleaned_rows)
                    if not cleaned_rows:
                        continue

                    layout.tables.append({
                        "rows": cleaned_rows,
                        "bbox": bbox_to_list(table.bbox),
                        "block_type": "table",
                    })
                    pages_with_tables.add(page_num)

    except Exception as e:
        print(f"[WARNING] pdfplumber table extraction failed: {e}")

    # ── Pass 2: PyMuPDF fallback for pages without tables ────────
    if doc is not None:
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            if page_num in pages_with_tables:
                continue  # already found tables via pdfplumber

            layout = _find_layout(pages, page_num)
            if layout is None:
                continue

            try:
                page = doc[page_idx]
                tabs = page.find_tables()

                for tab in tabs:
                    rows = tab.extract()
                    if not _is_valid_table(rows):
                        continue

                    cleaned_rows = _clean_rows(rows)
                    cleaned_rows = _strip_paragraph_rows(cleaned_rows)
                    if not cleaned_rows:
                        continue

                    layout.tables.append({
                        "rows": cleaned_rows,
                        "bbox": bbox_to_list(tab.bbox),
                        "block_type": "table",
                    })

            except Exception:
                continue

    return pages


# ── Validation helpers ───────────────────────────────────────────


def _is_valid_table(rows: list[list[str | None]] | None) -> bool:
    """
    Validate that extracted rows actually look like a real table.

    Rejects:
        - None / empty rows
        - Tables with only 1 column (usually paragraph text)
        - Tables where most rows have only 1 non-empty cell
        - Tables with fewer than 2 data rows
    """
    if not rows:
        return False

    # Need at least 2 columns to be a real table
    max_cols = max(len(row) for row in rows)
    if max_cols < 2:
        return False

    # Count non-empty rows (rows where > 1 cell has data)
    data_rows = 0
    for row in rows:
        non_empty = sum(1 for cell in row if cell and cell.strip())
        if non_empty >= 2:
            data_rows += 1

    # Need at least 2 rows with actual multi-column data
    if data_rows < 2:
        return False

    return True


def _clean_rows(rows: list[list[str | None]]) -> list[list[str]]:
    """Clean table rows: strip cells, remove fully-empty rows."""
    cleaned = []
    for row in rows:
        cleaned_row = [(cell.strip() if cell else "") for cell in row]
        # skip rows where every cell is empty
        if any(cell for cell in cleaned_row):
            cleaned.append(cleaned_row)
    return cleaned


def _strip_paragraph_rows(rows: list[list[str]]) -> list[list[str]]:
    """Remove rows that look like absorbed paragraph text rather than table data."""
    if len(rows) <= 2:
        return rows
    cleaned = []
    for row in rows:
        non_empty = [cell for cell in row if cell.strip()]
        # Single long cell = probably a paragraph, not a table row
        if len(non_empty) == 1 and len(non_empty[0]) > 150:
            continue
        # Any cell with very long text + newlines + other cells empty
        skip = False
        for cell in row:
            if len(cell) > 200 and '\n' in cell:
                others_empty = sum(1 for c in row if c != cell and not c.strip())
                if others_empty >= len(row) - 1:
                    skip = True
                    break
        if skip:
            continue
        cleaned.append(row)
    return cleaned


def _find_layout(pages: list[PageLayout], page_number: int) -> PageLayout | None:
    """Find the PageLayout for a given page number."""
    for layout in pages:
        if layout.page_number == page_number:
            return layout
    return None


def table_to_text(rows: list[list[str]]) -> str:
    """
    Serialize a table to a readable Markdown text representation
    (useful for token counting, text search, and LLM comprehension).
    """
    if not rows:
        return ""
    
    lines = []
    for i, row in enumerate(rows):
        # Replace newlines in cells with spaces to keep row intact
        cleaned_row = [cell.replace('\n', ' ') if cell else "" for cell in row]
        lines.append("| " + " | ".join(cleaned_row) + " |")
        
        # Add markdown header separator after the first row
        if i == 0:
            lines.append("|" + "|".join(["---" for _ in cleaned_row]) + "|")
            
    return "\n".join(lines)
