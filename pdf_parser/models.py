"""
Data models for parsed PDF content elements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ContentElement:
    """A single extracted content element from a PDF."""

    id: str
    type: str  # "heading", "paragraph", "table", "figure", "mixed"
    text: Optional[str] = None
    tables: Optional[list[list[str]]] = None
    images: Optional[list[str]] = None
    page_number: int = 1
    position: Optional[list[float]] = None  # [x0, y0, x1, y1]
    section_heading: Optional[str] = None
    tokens: int = 0
    metadata: Optional[dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary, dropping None values for cleaner output."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class PageLayout:
    """Intermediate representation of a single PDF page's raw content."""

    page_number: int
    width: float
    height: float
    text_blocks: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
