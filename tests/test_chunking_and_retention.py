"""Heading-aware chunking and the existence/freshness split in reconciliation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from chunking import Chunk, chunk_document, chunk_markdown, chunk_text
from embeddings import HashingEmbedder
from models import IngestDocument, ScanResult
from store import KnowledgeStore, _is_markdown


def _crumbs(chunks: list[Chunk]) -> list[str]:
    return [chunk.breadcrumb for chunk in chunks]


def test_markdown_is_split_on_headings_with_a_breadcrumb():
    text = "\n".join(
        [
            "# Gateway",
            "",
            "Intro paragraph.",
            "",
            "## Threat model",
            "",
            "The gateway does not accept executable code.",
            "",
            "### Replay",
            "",
            "Replay reads the source database read-only.",
        ]
    )
    chunks = chunk_markdown(text, target_size=60, overlap=0)

    assert all(chunk.breadcrumb for chunk in chunks)
    assert "Gateway › Threat model" in _crumbs(chunks)
    assert "Gateway › Threat model › Replay" in _crumbs(chunks)


def test_short_sections_are_packed_so_the_index_does_not_double():
    """One chunk per heading doubled the corpus and halved mean chunk size."""

    sections = "\n\n".join(f"## Section {index}\n\nBody {index}." for index in range(12))
    packed = chunk_markdown(sections, target_size=1800, overlap=200)

    assert len(packed) == 1
    # Nothing is shared above the sections, so the chunk is attributed to the
    # first rather than left anonymous.
    assert packed[0].breadcrumb == "Section 0"
    assert "Body 11." in packed[0].text


def test_packed_chunks_carry_the_deepest_shared_breadcrumb():
    text = "\n".join(
        [
            "# Runbook",
            "",
            "## Recovery",
            "",
            "### Step one",
            "",
            "Do the first thing.",
            "",
            "### Step two",
            "",
            "Do the second thing.",
        ]
    )
    chunks = chunk_markdown(text, target_size=1800, overlap=200)

    assert len(chunks) == 1
    assert chunks[0].breadcrumb == "Runbook › Recovery"


def test_oversized_sections_fall_back_to_paragraph_packing_within_the_section():
    body = "\n\n".join(f"Paragraph {index} " + "x" * 200 for index in range(12))
    text = f"# Big\n\n## Detail\n\n{body}"
    chunks = chunk_markdown(text, target_size=500, overlap=50)

    assert len(chunks) > 1
    assert all(chunk.breadcrumb == "Big › Detail" for chunk in chunks)


def test_hashes_inside_fenced_code_are_not_treated_as_headings():
    text = "\n".join(
        [
            "# Real heading",
            "",
            "```bash",
            "# this is a shell comment, not a heading",
            "echo hi",
            "```",
            "",
            "Body text.",
        ]
    )
    chunks = chunk_markdown(text, target_size=1800, overlap=200)

    assert _crumbs(chunks) == ["Real heading"]
    assert "shell comment" in chunks[0].text


def test_setext_headings_are_recognised():
    text = "Title\n=====\n\nBody.\n\nSubtitle\n--------\n\nMore body."
    chunks = chunk_markdown(text, target_size=40, overlap=0)
    assert "Title" in _crumbs(chunks)[0]


def test_plain_text_gets_no_breadcrumb_and_matches_the_paragraph_chunker():
    text = "\n\n".join(f"Paragraph {index}." for index in range(30))
    via_document = chunk_document(text, target_size=200, overlap=20, markdown=False)
    direct = chunk_text(text, 200, 20)

    assert [chunk.text for chunk in via_document] == direct
    assert _crumbs(via_document) == [""] * len(direct)


def test_only_project_markdown_uses_the_heading_chunker(settings_factory):
    class _Item:
        def __init__(self, source, title, metadata_json='{"extension": ".md"}'):
            self.source = source
            self.title = title
            self.metadata_json = metadata_json

    assert _is_markdown(_Item("projects", "docs/guide.md")) is True
    assert _is_markdown(_Item("projects", "docs/guide.mdx", '{"extension": ".mdx"}')) is True
    assert _is_markdown(_Item("projects", "notes.txt", '{"extension": ".txt"}')) is False
    # Transcripts use USER/ASSISTANT blocks whose "#" lines are dialogue content.
    assert _is_markdown(_Item("claude", "session.md")) is False


def test_aged_out_sessions_are_retained_instead_of_reconciled_away(settings_factory):
    """The rolling window must not hard-delete indexed knowledge.

    A session whose messages have all aged out produces no document, but it is
    still on disk and nothing would ever rebuild what reconciliation removes.
    """

    settings = settings_factory()
    store = KnowledgeStore(settings, HashingEmbedder(32))
    store.upsert_document(
        IngestDocument(
            source="claude",
            source_key="session:old",
            title="Old session",
            text="USER [2026-01-01T00:00:00Z]\nancient but valuable",
            timestamp=datetime.now(timezone.utc) - timedelta(days=200),
        )
    )

    aged_out = ScanResult(source="claude", retained_keys={"session:old"})
    assert store.reconcile_source("claude", aged_out.seen_keys) == 0
    assert store.get_document(
        next(iter(store.search_response("ancient", global_search=True)["results"]))["document_id"]
    ) is not None

    # A session that genuinely vanished is still reconciled away.
    vanished = ScanResult(source="claude")
    assert store.reconcile_source("claude", vanished.seen_keys) == 1


def test_seen_keys_unions_documents_and_retained_keys():
    result = ScanResult(
        source="claude",
        documents=[
            IngestDocument(
                source="claude",
                source_key="session:fresh",
                title="t",
                text="body",
                timestamp=datetime.now(timezone.utc),
            )
        ],
        retained_keys={"session:aged"},
    )
    assert result.seen_keys == {"session:fresh", "session:aged"}
