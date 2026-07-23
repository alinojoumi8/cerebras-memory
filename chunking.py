"""Paragraph-first text chunking with bounded character overlap."""

from __future__ import annotations

import re


def _hard_split(text: str, size: int) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > size:
        boundary = remaining.rfind(" ", 0, size + 1)
        if boundary < size // 2:
            boundary = size
        parts.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    if len(text) > overlap:
        first_space = tail.find(" ")
        if first_space >= 0:
            tail = tail[first_space + 1 :]
    return tail.strip()


def chunk_text(text: str, target_size: int = 1800, overlap: int = 200) -> list[str]:
    """Split text on paragraphs first, falling back to word boundaries.

    The target is approximate because complete paragraphs are preferred.  No
    emitted chunk exceeds ``target_size + overlap``.
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    if target_size < 100:
        raise ValueError("target_size must be at least 100 characters")
    overlap = min(max(0, overlap), target_size // 2)

    raw_paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    paragraphs: list[str] = []
    for paragraph in raw_paragraphs:
        if len(paragraph) <= target_size:
            paragraphs.append(paragraph)
        else:
            paragraphs.extend(_hard_split(paragraph, target_size))

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= target_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            prefix = _tail_overlap(current, overlap)
            candidate = paragraph if not prefix else f"{prefix}\n\n{paragraph}"

        if len(candidate) <= target_size + overlap:
            current = candidate
            continue

        # This only occurs when an overlap prefix meets a maximum-sized hard
        # split. Preserve the paragraph and trim the overlap, never the content.
        current = paragraph

    if current.strip():
        chunks.append(current.strip())
    return chunks
