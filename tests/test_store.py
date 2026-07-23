from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

import ingest
from embeddings import HashingEmbedder
from models import IngestDocument, ScanResult
from store import KnowledgeStore


NOW = datetime.now(timezone.utc)


def _doc(
    key: str,
    text: str,
    *,
    source: str = "projects",
    project: str = "alpha",
    timestamp: datetime = NOW,
) -> IngestDocument:
    return IngestDocument(
        source=source,
        source_key=key,
        title=key,
        text=text,
        timestamp=timestamp,
        project=project,
        uri=f"fixture://{key}",
    )


def test_schema_version_wal_foreign_keys_and_transactional_replacement(store):
    assert store.schema_version() == 2
    first = store.upsert_document(_doc("one", "alpha " * 300))
    before = store.get_document(first.document_id, limit=50)
    assert before and before["pagination"]["total_chunks"] > 1
    old_ids = [chunk["chunk_id"] for chunk in before["chunks"]]

    second = store.upsert_document(_doc("one", "replacement content"))
    after = store.get_document(second.document_id, limit=50)
    assert second.status == "updated"
    assert after and after["pagination"]["total_chunks"] == 1
    assert after["chunks"][0]["chunk_id"] == old_ids[0]
    assert after["chunks"][0]["content"] == "replacement content"

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].casefold() == "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 2


def test_hybrid_search_filters_citations_and_recency(store):
    old = NOW - timedelta(days=500)
    store.upsert_document(_doc("old", "shared comet phrase", source="claude", timestamp=old))
    store.upsert_document(_doc("new", "shared comet phrase", source="codex", timestamp=NOW))
    store.upsert_document(_doc("other", "unrelated orchard", project="beta"))

    results = store.search("shared comet", limit=5)
    assert results[0]["source"] == "codex"
    assert results[0]["lexical_rank"] is not None
    assert results[0]["vector_rank"] is not None
    assert results[0]["citation"].startswith("cerebras-memory://document/doc_")
    assert results[0]["content_trust"] == "untrusted_evidence"
    assert all(len(result["context_chunks"]) <= 2 for result in results)

    filtered = store.search("shared comet", sources=["claude"], project="alpha")
    assert filtered and {result["source"] for result in filtered} == {"claude"}
    since = (NOW - timedelta(days=2)).isoformat()
    recent = store.search("shared comet", since=since, sources=["codex"])
    assert recent and {result["source"] for result in recent} == {"codex"}


def test_model_change_reembeds_unchanged_content(settings_factory):
    settings = settings_factory()
    first_store = KnowledgeStore(settings, HashingEmbedder(32, "test/hash-v1"))
    document = _doc("model-change", "model migration phrase")
    first_store.upsert_document(document)

    second_store = KnowledgeStore(settings, HashingEmbedder(32, "test/hash-v2"))
    result = second_store.upsert_document(document)
    assert result.status == "updated"
    stats = second_store.stats()
    assert stats["embedding"]["model"] == "test/hash-v2"
    assert stats["embedding"]["pending_reembed"] == 0


def test_v1_to_v2_migration_preserves_documents_chunks_and_citations(settings_factory):
    settings = settings_factory()
    original = KnowledgeStore(settings, HashingEmbedder(32))
    written = original.upsert_document(_doc("migration", "preserve this migration evidence"))
    before = original.get_document(written.document_id)
    assert before is not None

    # Reduce the temporary fixture to the exact v1 surface, then reopen it
    # through the real transactional migration path.
    with sqlite3.connect(settings.database_path) as connection:
        connection.executescript(
            """
            DROP TABLE distillations_fts;
            DROP TABLE distillation_pilot_documents;
            DROP TABLE distillation_unit_state;
            DROP TABLE distillation_state;
            DROP TABLE distillations;
            DROP TABLE vector_index_state;
            DELETE FROM schema_migrations WHERE version = 2;
            PRAGMA user_version = 1;
            """
        )

    migrated = KnowledgeStore(settings, HashingEmbedder(32))
    after = migrated.get_document(written.document_id)
    assert migrated.schema_version() == 2
    assert after is not None
    assert after["document"]["document_id"] == before["document"]["document_id"]
    assert [chunk["chunk_id"] for chunk in after["chunks"]] == [
        chunk["chunk_id"] for chunk in before["chunks"]
    ]
    assert migrated.search("migration evidence", global_search=True)[0]["citation"].endswith(
        f"chunk={after['chunks'][0]['chunk_id']}"
    )


def test_wal_cross_thread_writers(settings_factory):
    settings = settings_factory()

    def write(index: int) -> str:
        local = KnowledgeStore(settings, HashingEmbedder(32))
        return local.upsert_document(_doc(f"thread-{index}", f"parallel writer {index}")).status

    with ThreadPoolExecutor(max_workers=4) as pool:
        statuses = list(pool.map(write, range(12)))
    assert statuses == ["created"] * 12
    assert KnowledgeStore(settings, HashingEmbedder(32)).stats()["documents"] == 12


def test_exact_vector_cache_reloads_after_cross_process_generation_change(settings_factory):
    settings = settings_factory()
    reader = KnowledgeStore(settings, HashingEmbedder(32))
    reader.upsert_document(_doc("before-cache", "initial cache evidence"))
    assert reader.search("initial cache evidence", global_search=True)

    writer = KnowledgeStore(settings, HashingEmbedder(32))
    inserted = writer.upsert_document(_doc("after-cache", "new cross process generation marker"))
    refreshed = reader.search(
        "new cross process generation marker",
        global_search=True,
        rerank=False,
    )
    assert any(result["document_id"] == inserted.document_id for result in refreshed)


def test_confirmed_memory_is_idempotent_and_never_reconciled(store):
    with pytest.raises(PermissionError):
        store.save_memory("No", "not confirmed")
    first = store.save_memory("Remember", "persistent fact", tags=["fixture"], confirmed_by_user=True)
    second = store.save_memory("Different title", "persistent fact", confirmed_by_user=True)
    assert second["status"] == "unchanged"
    assert first["document_id"] == second["document_id"]
    assert store.reconcile_source("memory", set()) == 0
    assert store.get_document(first["document_id"]) is not None
    assert store.forget_memory(first["document_id"])
    assert store.get_document(first["document_id"]) is None


def test_redaction_is_applied_before_sqlite_write(store):
    secret = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    result = store.upsert_document(_doc("redacted", f"credential {secret}"))
    with sqlite3.connect(store.database_path) as connection:
        stored = connection.execute(
            "SELECT content FROM chunks WHERE document_id = ?", (result.document_id,)
        ).fetchone()[0]
    assert secret not in stored
    assert "[REDACTED]" in stored


def test_failed_scan_does_not_reconcile_existing_documents(settings_factory, monkeypatch):
    settings = settings_factory(enabled_sources=frozenset({"claude"}))
    store = KnowledgeStore(settings, HashingEmbedder(32))
    existing = store.upsert_document(_doc("keep", "must survive", source="claude"))

    def failed_scanner(_settings, _cutoff):
        return ScanResult(source="claude", successful=False, error="fixture failure")

    monkeypatch.setenv("CEREBRAS_MEMORY_TEST_EMBEDDER", "1")
    monkeypatch.setattr(ingest, "_scanners", lambda _settings: [("claude", failed_scanner)])
    report = ingest.run_ingestion(settings)
    assert report["ok"] is False
    assert store.get_document(existing.document_id) is not None
    assert store.stats()["sources"]["claude"]["status"] == "failed"


def test_successful_refresh_maintains_vectors_and_distillation_failure_is_nonblocking(
    settings_factory, monkeypatch
):
    base = settings_factory(enabled_sources=frozenset({"codex"}))
    settings = replace(base, distillation=replace(base.distillation, mode="on"))

    def successful_scanner(_settings, _cutoff):
        return ScanResult(
            source="codex",
            documents=[_doc("refresh", "raw ingestion survives derived failures", source="codex")],
            scanned=1,
            successful=True,
            watermark=NOW.isoformat(),
        )

    class DerivedFailureStore(KnowledgeStore):
        def distill_documents(self, **kwargs):
            raise RuntimeError("fixture distillation outage")

    monkeypatch.setenv("CEREBRAS_MEMORY_TEST_EMBEDDER", "1")
    monkeypatch.setattr(ingest, "KnowledgeStore", DerivedFailureStore)
    monkeypatch.setattr(ingest, "_scanners", lambda _settings: [("codex", successful_scanner)])
    report = ingest.run_ingestion(settings)

    assert report["ok"] is True
    assert report["sources"]["codex"]["status"] == "ok"
    assert report["sources"]["codex"]["distillation"]["status"] == "failed"
    assert report["vector_maintenance"]["benchmark"]["chunks"] == 1
    persisted = KnowledgeStore(settings, HashingEmbedder(32))
    assert persisted.stats()["documents"] == 1
