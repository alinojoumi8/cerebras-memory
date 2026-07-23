from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import threading
import time

import pytest

from chunking import chunk_text
from distillation import (
    DISTILLATION_FIELDS,
    DistillationInvalid,
    DistillationUnavailable,
    OllamaDistiller,
    reconstruct_chunks,
    segment_dialogue,
    validate_distillation,
)
from embeddings import HashingEmbedder
from models import IngestDocument
from store import KnowledgeStore


class FixtureDistiller:
    model_name = "test/distiller"

    def __init__(self, *, secret: str | None = None):
        self.calls = 0
        self.secret = secret

    def distill(self, text):
        self.calls += 1
        return {
            "user_goal": self.secret or "Complete the azure rollout",
            "summary": "The blue deployment was completed and verified.",
            "outcome": "Deployment complete",
            "decisions": ["Use the local index"],
            "artifacts": ["deployment-notes.md"],
            "systems": ["Cerebras Memory"],
            "open_questions": [],
            "keywords": ["azure rollout", "local retrieval"],
        }


class InvalidDistiller:
    model_name = "test/distiller"

    def distill(self, text):
        return {"summary": "missing required fields"}


class OfflineDistiller:
    model_name = "test/distiller"

    def distill(self, text):
        raise DistillationUnavailable("ollama_offline_fixture")


class InterruptingDistiller(FixtureDistiller):
    def distill(self, text):
        if self.calls == 1:
            raise KeyboardInterrupt("simulated process interruption")
        return super().distill(text)


class ConcurrentDistiller(FixtureDistiller):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def distill(self, text):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.03)
            return super().distill(text)
        finally:
            with self.lock:
                self.active -= 1


def _dialogue(message_count: int, *, start: datetime | None = None) -> str:
    started = start or datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    messages = []
    for index in range(message_count):
        role = "USER" if index % 2 == 0 else "ASSISTANT"
        timestamp = (started + timedelta(minutes=index)).isoformat().replace("+00:00", "Z")
        content = (
            "Please complete the blue deployment with raw evidence."
            if role == "USER"
            else "The blue deployment was completed and verified against the local index."
        )
        messages.append(f"{role} [{timestamp}]\n{content} Message {index}.")
    return "\n\n".join(messages)


def _agent_doc(key: str, message_count: int, *, text: str | None = None) -> IngestDocument:
    return IngestDocument(
        source="codex",
        source_key=key,
        title=f"Dialogue {key}",
        text=text or _dialogue(message_count),
        timestamp=datetime.now(timezone.utc),
        project="alpha",
        metadata={"message_count": message_count, "roles": ["user", "assistant"]},
    )


def _distillation_store(settings_factory, distiller, *, mode: str = "pilot") -> KnowledgeStore:
    base = settings_factory(enabled_sources=frozenset({"codex"}))
    settings = replace(
        base,
        distillation=replace(base.distillation, mode=mode, min_characters=12_000),
    )
    return KnowledgeStore(settings, HashingEmbedder(32), distiller=distiller)


def test_segmentation_is_stable_after_append_and_obeys_boundaries(settings_factory):
    settings = settings_factory().distillation
    original = _dialogue(14)
    original_chunks = chunk_text(original, 400, 50)
    body, spans = reconstruct_chunks(list(enumerate(original_chunks)))
    units = segment_dialogue(body, spans, settings)
    assert [len(unit.text) <= settings.max_characters_per_unit for unit in units] == [True] * len(units)
    assert len(units) == 2

    appended = original + "\n\n" + _dialogue(
        2,
        start=datetime(2026, 7, 20, 12, 14, tzinfo=timezone.utc),
    )
    appended_chunks = chunk_text(appended, 400, 50)
    appended_body, appended_spans = reconstruct_chunks(list(enumerate(appended_chunks)))
    appended_units = segment_dialogue(appended_body, appended_spans, settings)
    assert appended_units[0].input_hash == units[0].input_hash
    assert appended_units[0].start_ordinal == units[0].start_ordinal

    separated = _dialogue(2) + "\n\n" + _dialogue(
        2,
        start=datetime(2026, 7, 20, 13, 0, tzinfo=timezone.utc),
    )
    separated_body, separated_spans = reconstruct_chunks(list(enumerate(chunk_text(separated, 400, 50))))
    assert len(segment_dialogue(separated_body, separated_spans, settings)) == 2


def test_distillation_cache_reuse_append_and_output_redaction(settings_factory):
    secret = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    distiller = FixtureDistiller(secret=secret)
    store = _distillation_store(settings_factory, distiller)
    document = store.upsert_document(_agent_doc("append", 14))

    first = store.distill_document(document.document_id)
    assert first["status"] == "ready"
    assert first["generated"] == 2
    assert distiller.calls == 2
    with sqlite3.connect(store.database_path) as connection:
        identifiers = [
            row[0]
            for row in connection.execute(
                "SELECT id FROM distillations WHERE document_id = ? ORDER BY unit_ordinal",
                (document.document_id,),
            )
        ]
        payloads = connection.execute(
            "SELECT summary_json, search_text FROM distillations WHERE document_id = ?",
            (document.document_id,),
        ).fetchall()
    assert all(secret not in json.dumps(payload) for payload in payloads)
    assert any("[REDACTED]" in payload[0] for payload in payloads)

    cached = store.distill_document(document.document_id)
    assert cached["generated"] == 0
    assert distiller.calls == 2

    appended_text = _dialogue(14) + "\n\n" + _dialogue(
        2,
        start=datetime(2026, 7, 20, 12, 14, tzinfo=timezone.utc),
    )
    store.upsert_document(_agent_doc("append", 16, text=appended_text))
    pending = store.distillation_status()
    assert pending["states"] == {"pending": 1}
    assert pending["unit_states"] == {"pending": 2}
    refreshed = store.distill_document(document.document_id)
    assert refreshed["status"] == "ready"
    assert refreshed["generated"] == 1
    assert distiller.calls == 3
    assert store.distillation_status()["unit_states"] == {"ready": 2}
    with sqlite3.connect(store.database_path) as connection:
        refreshed_ids = [
            row[0]
            for row in connection.execute(
                "SELECT id FROM distillations WHERE document_id = ? ORDER BY unit_ordinal",
                (document.document_id,),
            )
        ]
    assert refreshed_ids[0] == identifiers[0]


def test_schema_rejection_and_ollama_downtime_are_retryable(settings_factory):
    invalid_store = _distillation_store(settings_factory, InvalidDistiller())
    invalid_doc = invalid_store.upsert_document(_agent_doc("invalid", 8))
    invalid = invalid_store.distill_document(invalid_doc.document_id)
    assert invalid["status"] == "failed"
    assert invalid["failures"] == 1
    assert invalid_store.distillation_status()["states"] == {"failed": 1}
    assert invalid_store.distillation_status()["unit_states"] == {"failed": 1}

    offline_store = _distillation_store(settings_factory, OfflineDistiller())
    offline_doc = offline_store.upsert_document(_agent_doc("offline", 14))
    failed = offline_store.distill_document(offline_doc.document_id)
    assert failed["status"] == "failed"
    assert offline_store.distillation_status()["unit_states"] == {
        "failed": 2,  # includes the earlier invalid-schema fixture document
        "pending": 1,
    }
    assert offline_store.search("blue deployment", global_search=True, rerank=False)
    assert offline_store.stats()["distillation"]["last_error"] == "ollama_offline_fixture"

    replacement = FixtureDistiller()
    offline_store.distiller = replacement
    retry = offline_store.distill_document(offline_doc.document_id)
    assert retry["status"] == "ready"
    assert retry["generated"] == 2


def test_distillation_checkpoints_units_across_process_interruption(settings_factory):
    store = _distillation_store(settings_factory, InterruptingDistiller())
    document = store.upsert_document(_agent_doc("checkpoint", 14))
    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        store.distill_document(document.document_id)

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM distillations WHERE document_id = ?",
            (document.document_id,),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT status FROM distillation_state WHERE document_id = ?",
            (document.document_id,),
        ).fetchone()[0] == "pending"

    replacement = FixtureDistiller()
    store.distiller = replacement
    resumed = store.distill_document(document.document_id)
    assert resumed["status"] == "ready"
    assert resumed["generated"] == 1
    assert replacement.calls == 1


def test_remote_distillation_concurrency_keeps_checkpoints_serialized(settings_factory):
    distiller = ConcurrentDistiller()
    base = settings_factory(enabled_sources=frozenset({"codex"}))
    settings = replace(
        base,
        distillation=replace(
            base.distillation,
            mode="on",
            min_characters=12_000,
            max_concurrent_requests=3,
        ),
    )
    store = KnowledgeStore(settings, HashingEmbedder(32), distiller=distiller)
    document = store.upsert_document(_agent_doc("concurrent", 36))

    result = store.distill_document(document.document_id)

    assert result == {
        "document_id": document.document_id,
        "status": "ready",
        "units": 3,
        "ready": 3,
        "generated": 3,
        "failures": 0,
    }
    assert distiller.max_active >= 2
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM distillations WHERE document_id = ?",
            (document.document_id,),
        ).fetchone()[0] == 3


def test_document_concurrency_respects_the_global_request_limit(settings_factory):
    distiller = ConcurrentDistiller()
    base = settings_factory(enabled_sources=frozenset({"codex"}))
    settings = replace(
        base,
        distillation=replace(
            base.distillation,
            mode="on",
            min_characters=12_000,
            max_concurrent_requests=2,
        ),
    )
    store = KnowledgeStore(settings, HashingEmbedder(32), distiller=distiller)
    for index in range(4):
        store.upsert_document(_agent_doc(f"document-concurrent-{index}", 8))

    result = store.distill_documents()

    assert result["documents"] == 4
    assert result["ready"] == 4
    assert result["failed"] == 0
    assert result["generated"] == 4
    assert distiller.max_active == 2


def test_selective_regeneration_uses_unit_content_hash(settings_factory):
    initial = FixtureDistiller()
    store = _distillation_store(settings_factory, initial, mode="on")
    document = store.upsert_document(_agent_doc("selective-regeneration", 14))
    assert store.distill_document(document.document_id)["generated"] == 2
    with sqlite3.connect(store.database_path) as connection:
        selected_hash = str(
            connection.execute(
                """
                SELECT input_hash FROM distillations
                WHERE document_id = ? ORDER BY unit_ordinal LIMIT 1
                """,
                (document.document_id,),
            ).fetchone()[0]
        )

    replacement = FixtureDistiller()
    store.distiller = replacement
    result = store.distill_document(
        document.document_id,
        force_input_hashes={selected_hash},
    )

    assert result["status"] == "ready"
    assert result["generated"] == 1
    assert replacement.calls == 1


def test_distillation_retrieval_maps_to_raw_evidence_and_cascades(settings_factory):
    distiller = FixtureDistiller()
    store = _distillation_store(settings_factory, distiller, mode="on")
    document = store.upsert_document(_agent_doc("mapped", 8))
    assert store.distill_document(document.document_id)["status"] == "ready"

    response = store.search_response(
        "azure rollout",
        global_search=True,
        rerank=False,
    )
    match = next(item for item in response["results"] if item["document_id"] == document.document_id)
    assert match["distillation_id"].startswith("dst_")
    assert "distillation" in match["matched_via"]
    assert match["citation"].startswith(f"cerebras-memory://document/{document.document_id}?chunk=chk_")
    assert match["content_trust"] == "untrusted_evidence"
    assert response["retrieval"]["distillation"]["evidence"] == "raw_chunks_only"

    assert store.reconcile_source("codex", set()) == 1
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM distillations").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM distillation_state").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM distillation_unit_state").fetchone()[0] == 0


def test_validate_distillation_requires_exact_schema():
    valid = {field: [] for field in DISTILLATION_FIELDS}
    valid.update({"user_goal": "", "summary": "", "outcome": None})
    assert validate_distillation(valid)["outcome"] is None
    with pytest.raises(DistillationInvalid):
        validate_distillation({"summary": "incomplete"})


def test_ollama_body_read_has_absolute_deadline(settings_factory, monkeypatch):
    settings = replace(settings_factory().distillation, timeout_seconds=0.02)

    class SlowResponse:
        def __init__(self):
            self.closed = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            self.closed.set()

        def read(self):
            self.closed.wait(1.0)
            raise OSError("response closed")

    monkeypatch.setattr("distillation.urlopen", lambda *args, **kwargs: SlowResponse())
    started = time.monotonic()
    with pytest.raises(DistillationUnavailable, match="absolute deadline"):
        OllamaDistiller(settings).distill("USER [2026-01-01T00:00:00Z]\nhello")
    assert time.monotonic() - started < 0.5


def test_pilot_cohort_is_pinned_across_document_updates(settings_factory):
    store = _distillation_store(settings_factory, FixtureDistiller())
    for index in range(12):
        store.upsert_document(_agent_doc(f"cohort-{index}", 8))
    first = store._qualifying_distillation_documents(source="codex", pilot=True)
    assert len(first) == 10

    selected_key = next(
        f"cohort-{index}"
        for index in range(12)
        if store.upsert_document(_agent_doc(f"cohort-{index}", 8)).document_id == first[0]
    )
    store.upsert_document(
        _agent_doc(selected_key, 10, text=_dialogue(10) + "\n\ncohort append marker")
    )
    second = store._qualifying_distillation_documents(source="codex", pilot=True)
    assert set(second) == set(first)
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM distillation_pilot_documents WHERE source = 'codex'"
        ).fetchone()[0] == 10


def test_distillation_character_qualification_ignores_chunk_overlap(settings_factory):
    store = _distillation_store(settings_factory, FixtureDistiller())
    text = "USER [2026-07-20T12:00:00Z]\n" + ("evidence " * 1_315)
    assert len(text) < store.settings.distillation.min_characters
    written = store.upsert_document(_agent_doc("overlap-only", 1, text=text))
    with sqlite3.connect(store.database_path) as connection:
        inflated = connection.execute(
            "SELECT SUM(LENGTH(content)) FROM chunks WHERE document_id = ?",
            (written.document_id,),
        ).fetchone()[0]
    assert inflated >= store.settings.distillation.min_characters
    assert written.document_id not in store._qualifying_distillation_documents(source="codex")
