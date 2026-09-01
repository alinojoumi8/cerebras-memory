"""Text chunking.

Two strategies. Prose and transcripts use paragraph-first packing with a bounded
character overlap. Markdown uses its heading structure instead, because splitting
a structured document on a raw character count severs sections mid-sentence and
leaves the resulting chunk with nothing that says where it came from.

Each markdown chunk carries a breadcrumb of the headings above it. That
breadcrumb is what gives an otherwise context-free fragment - "we decided to keep
the 30-second timeout" - something to be retrieved by.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SETEXT_UNDERLINE = re.compile(r"^\s*(=+|-{2,})\s*$")

# Headings deeper than this stop being useful context and start being noise.
_MAX_BREADCRUMB_DEPTH = 3


@dataclass(frozen=True)
class Chunk:
    """A chunk plus the heading path it came from (empty for plain text)."""

    text: str
    breadcrumb: str = ""


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


def _iter_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into ``(breadcrumb, body)`` pairs.

    Fenced code is passed through untouched so a ``#`` comment inside a shell
    block cannot be mistaken for a heading.
    """

    lines = text.split("\n")
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    body: list[str] = []
    fence: str | None = None

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            trail = [title for _level, title in stack[:_MAX_BREADCRUMB_DEPTH]]
            sections.append((" › ".join(trail), content))
        body.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif line.strip().startswith(fence):
                fence = None
            body.append(line)
            index += 1
            continue
        if fence is not None:
            body.append(line)
            index += 1
            continue

        heading = _ATX_HEADING.match(line)
        level: int | None = None
        title = ""
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
        elif (
            index + 1 < len(lines)
            and line.strip()
            and _SETEXT_UNDERLINE.match(lines[index + 1])
            and not line.startswith(("-", "*", "+", ">", "|"))
        ):
            level = 1 if lines[index + 1].strip().startswith("=") else 2
            title = line.strip()
            index += 1

        if level is not None and title:
            flush()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body.append(line)
        index += 1

    flush()
    return sections


def _common_breadcrumb(paths: list[str]) -> str:
    """Deepest heading path shared by every packed section."""

    if not paths:
        return ""
    split = [path.split(" › ") if path else [] for path in paths]
    shared: list[str] = []
    for parts in zip(*split):
        if len({part for part in parts}) != 1:
            break
        shared.append(parts[0])
    # Nothing in common: fall back to the first section so the chunk is still
    # attributable rather than anonymous.
    return " › ".join(shared) or paths[0]


def chunk_markdown(text: str, target_size: int = 1800, overlap: int = 200) -> list[Chunk]:
    """Chunk markdown on its heading structure.

    Sections longer than the target fall back to paragraph packing *within* that
    section, so a chunk never straddles two unrelated headings at the top level.
    Consecutive short sections are packed together up to the target, which
    matters: emitting one chunk per heading doubled the chunk count on the real
    corpus and dropped mean chunk size to 650 characters, starving each chunk of
    context and pushing the index past the ANN activation threshold for no gain.
    Packed chunks carry the deepest breadcrumb their sections share.
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    sections = _iter_sections(normalized)
    if not sections:
        return [Chunk(piece) for piece in chunk_text(normalized, target_size, overlap)]

    chunks: list[Chunk] = []
    pending: list[str] = []
    pending_paths: list[str] = []
    pending_length = 0

    def flush_pending() -> None:
        nonlocal pending_length
        if pending:
            chunks.append(Chunk("\n\n".join(pending), _common_breadcrumb(pending_paths)))
        pending.clear()
        pending_paths.clear()
        pending_length = 0

    for breadcrumb, body in sections:
        if len(body) > target_size:
            flush_pending()
            for piece in chunk_text(body, target_size, overlap):
                chunks.append(Chunk(piece, breadcrumb))
            continue
        addition = len(body) + (2 if pending else 0)
        if pending_length + addition > target_size:
            flush_pending()
            addition = len(body)
        pending.append(body)
        pending_paths.append(breadcrumb)
        pending_length += addition

    flush_pending()
    return chunks


def chunk_document(
    text: str,
    *,
    target_size: int = 1800,
    overlap: int = 200,
    markdown: bool = False,
) -> list[Chunk]:
    """Chunk a document with the strategy appropriate to its format."""

    if markdown:
        return chunk_markdown(text, target_size, overlap)
    return [Chunk(piece) for piece in chunk_text(text, target_size, overlap)]
