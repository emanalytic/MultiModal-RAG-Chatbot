"""
Semantic chunker — assembles raw extracted elements (text blocks, tables,
images) into coherent ContentElement chunks.

This is where the "magic" happens: instead of rigid page-by-page splitting,
we use soft heuristics to group related content together:

    - Text under the same heading → same section
    - Tables near related text → linked via section_heading
    - Figures near captions → merged into a single figure element
    - Headings propagate downward until the next heading
"""

from __future__ import annotations

from pdf_parser.models import ContentElement, PageLayout
from pdf_parser.table_extractor import table_to_text
from pdf_parser.utils import (
    generate_id,
    reset_id_counter,
    count_tokens,
    merge_bboxes,
    clean_text,
)


def build_chunks(pages: list[PageLayout]) -> list[ContentElement]:
    """
    Convert raw PageLayout data into a flat list of ContentElement chunks.

    Strategy:
        1. Walk through pages in order.
        2. Track the current section heading.
        3. For each text block, create a heading or paragraph element.
        4. For each table, create a table element linked to current section.
        5. For each image, create a figure element; try to attach nearby
           caption text.
        6. Merge consecutive small paragraphs under the same heading into
           a single "paragraph" chunk to avoid micro-fragments.
        7. Skip text blocks that overlap with a detected table bbox
           (deduplication).
    """
    reset_id_counter()
    elements: list[ContentElement] = []
    current_heading: str | None = None

    for layout in pages:
        page_num = layout.page_number

        # Collect table bboxes for deduplication
        table_bboxes = [
            t["bbox"] for t in layout.tables
            if t.get("bbox") and len(t["bbox"]) >= 4
        ]

        # Collect image bboxes for chart-text filtering
        image_bboxes = [
            img["bbox"] for img in layout.images
            if img.get("bbox") and len(img["bbox"]) >= 4
        ]

        # --- Build a unified list of all items sorted by vertical position ---
        items = []

        for block in layout.text_blocks:
            y_pos = block["bbox"][1] if block.get("bbox") else 0
            items.append(("text", block, y_pos))

        for table in layout.tables:
            y_pos = table["bbox"][1] if table.get("bbox") else 0
            items.append(("table", table, y_pos))

        for image in layout.images:
            y_pos = image["bbox"][1] if image.get("bbox") else 0
            items.append(("image", image, y_pos))

        # sort by vertical position on the page
        items.sort(key=lambda x: x[2])

        for item_type, item, _ in items:

            if item_type == "text":
                # Skip text blocks that overlap with a detected table
                text_bbox = item.get("bbox")
                if text_bbox and _overlaps_any_bbox(text_bbox, table_bboxes):
                    continue

                # Filter text that overlaps with images (e.g. chart axis labels)
                if text_bbox and _overlaps_any_bbox(text_bbox, image_bboxes, overlap_threshold=0.3):
                    continue

                block_type = item.get("block_type", "paragraph")
                text = item.get("text", "")

                if block_type == "heading":
                    current_heading = text
                    elements.append(ContentElement(
                        id=generate_id(),
                        type="heading",
                        text=text,
                        page_number=page_num,
                        position=item.get("bbox"),
                        section_heading=None,  # headings don't nest into themselves
                        tokens=count_tokens(text),
                    ))
                else:
                    elements.append(ContentElement(
                        id=generate_id(),
                        type="paragraph",
                        text=text,
                        page_number=page_num,
                        position=item.get("bbox"),
                        section_heading=current_heading,
                        tokens=count_tokens(text),
                    ))

            elif item_type == "table":
                rows = item.get("rows", [])
                serialized = table_to_text(rows)
                elements.append(ContentElement(
                    id=generate_id(),
                    type="table",
                    text=serialized if serialized else None,
                    tables=rows,
                    page_number=page_num,
                    position=item.get("bbox"),
                    section_heading=current_heading,
                    tokens=count_tokens(serialized),
                ))

            elif item_type == "image":
                caption = _find_caption(item, layout)
                ocr_text = item.get("ocr_text")
                display_text = caption or ocr_text or None

                elements.append(ContentElement(
                    id=generate_id(),
                    type="figure",
                    text=display_text,
                    images=[item.get("filename", "unknown.png")],
                    page_number=page_num,
                    position=item.get("bbox"),
                    section_heading=current_heading,
                    tokens=count_tokens(display_text),
                ))

    # post-processing: merge split headings on the same page
    elements = _merge_adjacent_headings(elements)

    # post-processing: merge small adjacent paragraphs for coherent RAG chunks
    elements = _merge_small_paragraphs(elements, max_tokens=100, max_combined=500)

    # post-processing: merge paragraphs split across page boundaries
    elements = _merge_cross_page_paragraphs(elements)

    # re-number chunk IDs sequentially after all merges
    for i, el in enumerate(elements, 1):
        el.id = f"chunk_{i:03d}"

    return elements


def _overlaps_any_bbox(
    text_bbox: list[float],
    table_bboxes: list[list[float]],
    overlap_threshold: float = 0.5,
) -> bool:
    """
    Check if a text block's bbox overlaps significantly with any table bbox.

    Returns True if more than `overlap_threshold` (50%) of the text block's
    area is covered by a table.
    """
    if not text_bbox or len(text_bbox) < 4:
        return False

    tx0, ty0, tx1, ty1 = text_bbox
    text_area = max((tx1 - tx0) * (ty1 - ty0), 1)

    for tb in table_bboxes:
        if not tb or len(tb) < 4:
            continue
        bx0, by0, bx1, by1 = tb

        # compute intersection
        ix0 = max(tx0, bx0)
        iy0 = max(ty0, by0)
        ix1 = min(tx1, bx1)
        iy1 = min(ty1, by1)

        if ix0 < ix1 and iy0 < iy1:
            overlap_area = (ix1 - ix0) * (iy1 - iy0)
            if overlap_area / text_area >= overlap_threshold:
                return True

    return False


def _find_caption(image: dict, layout: PageLayout) -> str | None:
    """
    Look for a caption near an image.

    Heuristic: find the closest text block immediately below the image
    that starts with "Figure", "Fig.", "Image", "Diagram", or similar,
    and is short (< 200 chars).
    """
    img_bbox = image.get("bbox")
    if not img_bbox or len(img_bbox) < 4:
        return None

    img_bottom = img_bbox[3]  # y1 of image
    caption_keywords = ("figure", "fig.", "fig ", "image", "diagram", "chart", "table", "graph")
    best_caption = None
    best_distance = float("inf")

    for block in layout.text_blocks:
        block_bbox = block.get("bbox")
        if not block_bbox or len(block_bbox) < 4:
            continue

        block_top = block_bbox[1]
        text = block.get("text", "").strip()

        # candidate must be below the image and close
        distance = block_top - img_bottom
        if 0 <= distance <= 40 and len(text) < 200:
            lower = text.lower()
            if any(lower.startswith(kw) for kw in caption_keywords):
                if distance < best_distance:
                    best_distance = distance
                    best_caption = text

    return best_caption


def _merge_small_paragraphs(
    elements: list[ContentElement],
    max_tokens: int = 100,
    max_combined: int = 500,
) -> list[ContentElement]:
    """
    Merge consecutive paragraph chunks that share the same section and page
    to create more coherent chunks for embedding/RAG.

    Merges when at least one chunk is small (<= max_tokens) and
    combined size stays under max_combined.
    """
    if not elements:
        return elements

    merged: list[ContentElement] = []
    buffer: ContentElement | None = None

    for el in elements:
        if el.type != "paragraph":
            if buffer:
                merged.append(buffer)
                buffer = None
            merged.append(el)
            continue

        if (
            buffer is not None
            and buffer.section_heading == el.section_heading
            and buffer.page_number == el.page_number
            and (buffer.tokens <= max_tokens or el.tokens <= max_tokens)
            and (buffer.tokens + el.tokens) <= max_combined
        ):
            combined_text = (buffer.text or "") + "\n" + (el.text or "")
            buffer.text = combined_text.strip()
            buffer.tokens = count_tokens(buffer.text)
            buffer.position = merge_bboxes(
                [b for b in [buffer.position, el.position] if b]
            )
        else:
            if buffer:
                merged.append(buffer)
            buffer = el

    if buffer:
        merged.append(buffer)

    return merged


def _merge_adjacent_headings(elements: list[ContentElement]) -> list[ContentElement]:
    """
    Merge consecutive heading chunks on the same page that are vertically
    adjacent (e.g. a heading that wraps across two text blocks).
    """
    if not elements:
        return elements

    merged: list[ContentElement] = []
    i = 0
    while i < len(elements):
        current = elements[i]
        if current.type != "heading":
            merged.append(current)
            i += 1
            continue

        # Try to merge with following heading(s) on same page
        while (
            i + 1 < len(elements)
            and elements[i + 1].type == "heading"
            and elements[i + 1].page_number == current.page_number
            and _vertically_close(current.position, elements[i + 1].position)
        ):
            next_el = elements[i + 1]
            combined = (current.text or "") + " " + (next_el.text or "")
            current.text = combined.strip()
            current.tokens = count_tokens(current.text)
            current.position = merge_bboxes(
                [b for b in [current.position, next_el.position] if b]
            )
            i += 1

        merged.append(current)
        i += 1

    # Fix section_heading references for elements after merged headings
    current_heading = None
    for el in merged:
        if el.type == "heading":
            current_heading = el.text
        elif el.section_heading is not None:
            el.section_heading = current_heading

    return merged


def _vertically_close(bbox1, bbox2, max_gap: float = 30.0) -> bool:
    """Check if two bboxes are vertically adjacent (within max_gap points)."""
    if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
        return True  # if we can't tell, assume close
    gap = abs(bbox2[1] - bbox1[3])
    return gap <= max_gap


def _merge_cross_page_paragraphs(
    elements: list[ContentElement],
    max_combined_tokens: int = 500,
) -> list[ContentElement]:
    """
    Merge paragraphs that were split across page boundaries.

    Heuristic: if a paragraph at the end of page N continues at the
    start of page N+1 (same section, text doesn't end cleanly),
    merge them into one chunk.
    """
    if len(elements) < 2:
        return elements

    merged: list[ContentElement] = []
    i = 0
    while i < len(elements):
        current = elements[i]

        if (
            current.type == "paragraph"
            and i + 1 < len(elements)
            and elements[i + 1].type == "paragraph"
            and elements[i + 1].page_number == current.page_number + 1
            and current.section_heading == elements[i + 1].section_heading
            and (current.tokens + elements[i + 1].tokens) <= max_combined_tokens
        ):
            next_el = elements[i + 1]
            current_text = (current.text or "").rstrip()
            next_text = (next_el.text or "").lstrip()

            # Merge if current doesn't end a sentence
            if not current_text.endswith(('.', '!', '?', '"')):
                combined = current_text + "\n" + next_text
                current.text = combined
                current.tokens = count_tokens(combined)
                current.position = None  # spans pages
                current.metadata = current.metadata or {}
                current.metadata["pages"] = [current.page_number, next_el.page_number]
                merged.append(current)
                i += 2
                continue

        merged.append(current)
        i += 1

    return merged
