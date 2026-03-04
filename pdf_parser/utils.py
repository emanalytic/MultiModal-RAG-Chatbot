"""
Utility functions for token counting, text cleanup, and ID generation.
"""

from __future__ import annotations

import re
import uuid

_encoder = None


def _get_encoder():
    """Lazy-load tiktoken encoder (cl100k_base — GPT-4 compatible)."""
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = "fallback"
    return _encoder


def count_tokens(text: str | None) -> int:
    """Return approximate token count for a text string."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc == "fallback":
        # rough fallback: ~4 chars per token
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def clean_text(text: str) -> str:
    """Normalize whitespace and strip artefacts from extracted text."""
    if not text:
        return ""
    # collapse multiple spaces / tabs
    text = re.sub(r"[ \t]+", " ", text)
    # collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


def is_heading_text(text: str) -> bool:
    """
    Heuristic: a line is likely a heading if it is short,
    possibly uppercase or title-case, and contains no sentence-ending
    punctuation mid-line.
    """
    text = text.strip()
    if not text or len(text) > 200:
        return False
    # very short lines that look like titles
    if len(text) < 80 and not text.endswith((".", ",", ";", ":")):
        # all-caps or title-case
        if text.isupper() or text.istitle():
            return True
    return False

_counter = 0


def generate_id(prefix: str = "chunk") -> str:
    """Generate a sequential chunk ID like chunk_001, chunk_002, …"""
    global _counter
    _counter += 1
    return f"{prefix}_{_counter:03d}"


def reset_id_counter():
    """Reset the ID counter (useful between documents)."""
    global _counter
    _counter = 0

def bbox_to_list(bbox) -> list[float] | None:
    """Convert any bbox-like object to [x0, y0, x1, y1]."""
    if bbox is None:
        return None
    try:
        return [round(float(v), 1) for v in bbox[:4]]
    except (TypeError, IndexError):
        return None


def merge_bboxes(bboxes: list[list[float]]) -> list[float] | None:
    """Merge multiple bounding boxes into their union."""
    valid = [b for b in bboxes if b and len(b) >= 4]
    if not valid:
        return None
    x0 = min(b[0] for b in valid)
    y0 = min(b[1] for b in valid)
    x1 = max(b[2] for b in valid)
    y1 = max(b[3] for b in valid)
    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
