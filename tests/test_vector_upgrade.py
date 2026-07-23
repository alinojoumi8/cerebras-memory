from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from embeddings import HashingEmbedder
from models import IngestDocument
from store import KnowledgeStore
from vector_index import UsearchVectorIndex
import vector_index as vector_index_module


def _doc(index: int, *, project: str = "alpha") -> IngestDocument:
    return IngestDocument(
        source="projects",
        source_key=f"vector-{index}",
        title=f"Vector fixture {index}",
        text=f"vector recall fixture {index} token-{index % 17} " + (f"value-{index} " * 12),
        timestamp=datetime.now(timezone.utc),
        project=project,
    )


def _ann_store(settings_factory, *, minimum: int = 1) -> KnowledgeStore:
    base = settings_factory()
    settings = replace(
        base,
        vector_search=replace(
            base.vector_search,
            backend="auto",
            ann_min_chunks=minimum,
            ann_latency_threshold_ms=10_000.0,
        ),
    )
    return KnowledgeStore(settings, HashingEmbedder(32))


def test_usearch_top_50_recall_and_restart_visibility(settings_factory):
    store = _ann_store(settings_factory)
    recall_index = UsearchVectorIndex(
        replace(store.settings.vector_search, index_dir=store.settings.vector_search.index_dir / "recall"),
        model="test/recall",
        dimensions=32,
    )
    random = np.random.default_rng(42)
    vectors = random.normal(size=(1_000, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = random.normal(size=(32,)).astype(np.float32)
    query /= np.linalg.norm(query)
    keys = np.arange(1, 1_001, dtype=np.uint64)
    recall_index.build(keys, vectors)
    exact_keys = {
        int(key) for key in keys[np.argsort(vectors @ query)[::-1][:50]]
    }
    approximate = recall_index.search(query, 50)
    approximate_keys = {key for _, key in approximate}
    assert len(exact_keys & approximate_keys) / len(exact_keys) >= 0.95

    store.upsert_documents([_doc(index) for index in range(140)])
    built = store.rebuild_vector_index()
    assert built["status"] == "ready"

    restarted = KnowledgeStore(store.settings, HashingEmbedder(32))
    response = restarted.search_response(
        "vector recall fixture token-7",
        global_search=True,
        rerank=False,
    )
    assert response["retrieval"]["vector"]["backend"] == "hnsw"
    assert response["results"]


def test_ann_activation_by_count_or_three_run_median(settings_factory):
    store = _ann_store(settings_factory, minimum=1000)
    store.upsert_documents([_doc(index) for index in range(8)])
    below = store.rebuild_vector_index()
    assert below == {"status": "exact", "built": False, "chunks": 8, "eligible": False}

    benchmark = store.benchmark_vector_search(runs=3)
    assert len(benchmark["runs_ms"]) == 3
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE vector_index_state SET exact_benchmark_ms = ? WHERE id = 1",
            (store.settings.vector_search.ann_latency_threshold_ms + 1.0,),
        )
    latency_activated = store.rebuild_vector_index()
    assert latency_activated["status"] == "ready"

    count_store = _ann_store(
        lambda **overrides: settings_factory(**overrides),
        minimum=1,
    )
    # The factory above shares the pytest temp root; use the already-created
    # corpus and verify the count threshold remains sufficient after restart.
    count_result = count_store.rebuild_vector_index()
    assert count_result["status"] == "ready"


def test_filtered_search_stays_exact_and_stale_or_corrupt_index_falls_back(settings_factory):
    store = _ann_store(settings_factory)
    store.upsert_documents([_doc(index, project="alpha" if index % 2 else "beta") for index in range(30)])
    assert store.rebuild_vector_index()["status"] == "ready"

    global_response = store.search_response("vector recall", global_search=True, rerank=False)
    assert global_response["retrieval"]["vector"]["backend"] == "hnsw"
    filtered = store.search_response(
        "vector recall",
        project="alpha",
        rerank=False,
    )
    assert filtered["retrieval"]["vector"]["backend"] == "exact"

    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE vector_index_state SET built_generation = data_generation - 1 WHERE id = 1"
        )
    stale = store.search_response("vector recall", global_search=True, rerank=False)
    assert stale["retrieval"]["vector"]["backend"] == "exact"

    assert store.rebuild_vector_index(force=True)["status"] == "ready"
    corrupt_path = Path(store.vector_index.path)
    corrupt_path.write_bytes(b"not-a-usearch-index")
    restarted = KnowledgeStore(store.settings, HashingEmbedder(32))
    corrupt = restarted.search_response("vector recall", global_search=True, rerank=False)
    assert corrupt["retrieval"]["vector"]["backend"] == "exact"
    assert corrupt["retrieval"]["vector"]["status"] == "fallback"
    assert corrupt["retrieval"]["vector"]["degraded_reason"]


def test_concurrent_memory_marks_newly_built_sidecar_stale(settings_factory):
    store = _ann_store(settings_factory)
    store.upsert_documents([_doc(index) for index in range(12)])
    original_build = store.vector_index.build

    def build_with_writer(keys, vectors):
        result = original_build(keys, vectors)
        other = KnowledgeStore(store.settings, HashingEmbedder(32))
        other.save_memory(
            "Concurrent memory",
            "memory written while the derived sidecar is being finalized",
            confirmed_by_user=True,
        )
        return result

    store.vector_index.build = build_with_writer  # type: ignore[method-assign]
    result = store.rebuild_vector_index(force=True)
    assert result["status"] == "stale"
    response = store.search_response("vector recall", global_search=True, rerank=False)
    assert response["retrieval"]["vector"]["backend"] == "exact"


def test_model_mismatch_uses_exact_backend(settings_factory):
    store = _ann_store(settings_factory)
    store.upsert_documents([_doc(index) for index in range(10)])
    assert store.rebuild_vector_index()["status"] == "ready"

    changed = KnowledgeStore(store.settings, HashingEmbedder(32, "test/hash-v2"))
    response = changed.search_response("vector recall", global_search=True, rerank=False)
    assert response["retrieval"]["vector"]["backend"] == "exact"


def test_atomic_sidecar_failure_preserves_previous_index(settings_factory, monkeypatch):
    store = _ann_store(settings_factory)
    manager = store.vector_index
    keys = np.arange(1, 11, dtype=np.uint64)
    vectors = np.eye(10, 32, dtype=np.float32)
    manager.build(keys, vectors)
    previous = manager.path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("fixture replacement failure")

    monkeypatch.setattr(vector_index_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        manager.build(keys, np.flip(vectors, axis=1).copy())
    assert manager.path.read_bytes() == previous
    assert not list(manager.path.parent.glob("*.tmp"))
