"""Batched embed-and-write: bounded memory, observable progress, resumable."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from embeddings import HashingEmbedder
from models import IngestDocument
from store import KnowledgeStore


def _docs(count: int, *, prefix: str = "doc") -> list[IngestDocument]:
    return [
        IngestDocument(
            source="projects",
            source_key=f"{prefix}-{index}.md",
            title=f"{prefix}-{index}.md",
            text=f"# Heading {index}\n\nBody text for document {index}.",
            timestamp=datetime.now(timezone.utc),
            project="alpha",
        )
        for index in range(count)
    ]


def test_work_is_committed_per_batch_not_once_per_source(settings_factory):
    """A crash used to lose the entire source.

    Everything was embedded into memory first and written in a single
    transaction, so an interrupted rebuild had nothing durable to show for
    hours of work.
    """

    settings = settings_factory(ingest_batch_documents=4)
    store = KnowledgeStore(settings, HashingEmbedder(32))

    committed: list[int] = []
    original = store._write_prepared  # noqa: SLF001 - observing commit boundaries

    def observe(prepared):
        result = original(prepared)
        with store._connect() as connection:  # noqa: SLF001
            committed.append(
                int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            )
        return result

    store._write_prepared = observe  # type: ignore[method-assign]
    store.upsert_documents(_docs(10))

    # 10 documents at a batch size of 4 => three commits, each durable.
    assert committed == [4, 8, 10]


def test_an_interrupted_batch_keeps_earlier_batches(settings_factory):
    """The half that succeeded stays; the next run re-derives only the rest."""

    settings = settings_factory(ingest_batch_documents=3)
    store = KnowledgeStore(settings, HashingEmbedder(32))

    original = store._write_prepared  # noqa: SLF001
    calls = {"n": 0}

    def fail_on_third(prepared):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated crash mid-source")
        return original(prepared)

    store._write_prepared = fail_on_third  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated crash"):
        store.upsert_documents(_docs(9))

    # Two batches landed before the failure and survived it.
    assert store.stats(recover=False)["documents"] == 6

    # Re-running completes the job; the six already-written documents are
    # recognised as unchanged rather than re-embedded.
    store._write_prepared = original  # type: ignore[method-assign]
    results = store.upsert_documents(_docs(9))
    assert store.stats(recover=False)["documents"] == 9
    assert sum(1 for r in results if r.status == "unchanged") == 6


def test_batching_does_not_change_the_result(settings_factory):
    """Batch size is a memory/latency knob, never a correctness one."""

    one_shot = KnowledgeStore(settings_factory(ingest_batch_documents=1000), HashingEmbedder(32))
    one_shot.upsert_documents(_docs(12))

    batched = KnowledgeStore(settings_factory(ingest_batch_documents=5), HashingEmbedder(32))
    batched.upsert_documents(_docs(12))

    def fingerprint(store: KnowledgeStore) -> list[tuple]:
        with store._connect() as connection:  # noqa: SLF001
            return [
                tuple(row)
                for row in connection.execute(
                    "SELECT id, document_id, ordinal, content_hash, chunker_version "
                    "FROM chunks ORDER BY document_id, ordinal"
                )
            ]

    assert fingerprint(one_shot) == fingerprint(batched)


def test_embedding_reuse_totals_are_aggregated_across_batches(settings_factory):
    settings = settings_factory(ingest_batch_documents=3)
    store = KnowledgeStore(settings, HashingEmbedder(32))

    store.upsert_documents(_docs(9))
    first = store.last_embedding_reuse
    assert first["total"] > 0
    assert first["embedded"] == first["total"]  # nothing to reuse on a cold index

    # Forcing a rebuild re-chunks everything, but the embedding *input* is
    # unchanged, so every vector must come from the cache rather than the model.
    # This is the case that took the real corpus from 2h06m to ~4 minutes.
    store.upsert_documents(_docs(9), force=True)
    second = store.last_embedding_reuse
    assert second["total"] > 0
    assert second["reused"] == second["total"]
    assert second["embedded"] == 0


def test_reuse_keys_on_the_embedded_text_not_the_raw_chunk(settings_factory):
    """The breadcrumb is part of the embedding input, so it must be part of the key.

    Two files with identical bodies live under different paths and therefore
    carry different breadcrumbs; reusing one vector for the other would silently
    discard the document context breadcrumbs exist to provide.
    """

    store = KnowledgeStore(settings_factory(), HashingEmbedder(32))
    store.upsert_documents(_docs(1, prefix="original"))
    store.upsert_documents(_docs(1, prefix="elsewhere"))

    assert store.last_embedding_reuse["reused"] == 0
