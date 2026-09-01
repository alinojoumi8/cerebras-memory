"""Concurrency regressions for serving several agents from one process.

Under STDIO each client owns its own worker and issues calls serially, so the
races below were unreachable.  A single hub process answering many agents makes
them reachable, which is what these tests pin down.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from embeddings import FastEmbedder, _QUERY_CACHE_SIZE


class _StubModel:
    def query_embed(self, values):
        return [np.ones(4, dtype=np.float32) for _ in values]


@pytest.fixture
def aggressive_thread_switching():
    """Force the interpreter to preempt threads far more often.

    The unguarded window between ``next(iter(...))`` and ``pop(...)`` is only a
    few bytecodes wide, so at the default 5 ms switch interval it effectively
    never fires -- measured over 32,000 racing calls it produced zero errors.
    At 1 us the same code raises ``RuntimeError`` and ``KeyError`` on nearly
    every thread.  The bug is real either way; this only makes it observable
    within a test run instead of in production weeks later.
    """

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def test_concurrent_query_caching_never_corrupts_the_bounded_cache(
    tmp_path, aggressive_thread_switching
):
    """Insert and eviction must be atomic together.

    ``next(iter(...))`` over an ``OrderedDict`` that another thread is mutating
    raises ``RuntimeError``; two threads evicting the same key raise
    ``KeyError``.  Driving far more distinct queries than the cache holds keeps
    the eviction branch hot for the whole run.
    """

    embedder = FastEmbedder("fixture/model", 4, tmp_path)
    embedder._model = _StubModel()

    errors: list[BaseException] = []

    def worker(offset: int) -> None:
        try:
            for index in range(2_000):
                embedder.embed_query(f"query-{offset}-{index}")
        except BaseException as exc:  # noqa: BLE001 - the failure is the signal
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(worker, range(16)))

    assert errors == []
    assert len(embedder._query_cache) == _QUERY_CACHE_SIZE


def test_concurrent_cold_start_builds_exactly_one_embedding_model(tmp_path, monkeypatch):
    """A cold start under load must not construct several ONNX sessions.

    ``lazy_load=True`` defers session creation into FastEmbed, so two threads
    that both pass the ``is None`` check race inside its loader rather than
    merely wasting a model.  The constructor sleeps so the window is real
    instead of depending on scheduler luck.
    """

    constructions: list[dict] = []
    lock = threading.Lock()

    class CountingTextEmbedding:
        def __init__(self, **kwargs):
            time.sleep(0.05)
            with lock:
                constructions.append(kwargs)

        def query_embed(self, values):
            return [np.ones(4, dtype=np.float32) for _ in values]

    fake = types.ModuleType("fastembed")
    fake.TextEmbedding = CountingTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake)

    embedder = FastEmbedder("fixture/model", 4, tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        models = list(pool.map(lambda _: embedder._load(), range(8)))

    assert len(constructions) == 1
    assert all(model is models[0] for model in models)


def test_concurrent_searches_return_identical_results_and_never_raise(store):
    """Many agents searching one store must not corrupt each other.

    ``_connect`` opens a fresh SQLite connection per call and each is consumed
    on its calling thread, so this pins that property rather than adding a pool.
    """

    store.save_memory(
        "Deployment note",
        "The hub binds loopback and is reached over the tailnet.",
        tags=["fixture"],
        confirmed_by_user=True,
    )
    store.save_memory(
        "Retrieval note",
        "Exact cosine search runs below the ANN activation threshold.",
        tags=["fixture"],
        confirmed_by_user=True,
    )

    expected = store.search_response("tailnet loopback", limit=5, global_search=True)

    errors: list[BaseException] = []
    results: list[dict] = []

    def worker(_: int) -> None:
        try:
            for _ in range(20):
                results.append(
                    store.search_response("tailnet loopback", limit=5, global_search=True)
                )
        except BaseException as exc:  # noqa: BLE001 - the failure is the signal
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(worker, range(16)))

    assert errors == []
    assert len(results) == 320
    expected_ids = [item["document_id"] for item in expected["results"]]
    for response in results:
        assert [item["document_id"] for item in response["results"]] == expected_ids


def test_prewarm_reports_every_stage_without_raising(store):
    """Cold-start cost must be payable up front, and must not be fatal.

    The hub calls this before it accepts traffic.  A stage that cannot warm has
    to be reported rather than raised, or an unavailable reranker would stop the
    process from serving exact search at all.
    """

    status = store.prewarm()

    assert status["embedder"]["ok"] is True
    assert status["embedder"]["model"] == store.embedder.model_name
    assert status["vectors"]["ok"] is True
    assert status["vectors"]["chunks"] == 0
    # The fixture disables reranking; that is a healthy state, not a failure.
    assert status["reranker"] == {"ok": True, "enabled": False}


def test_prewarm_survives_a_broken_embedder(store):
    """One dead stage must not take down the others."""

    class Broken:
        model_name = "broken/model"
        dimensions = 4

        def embed_query(self, text):
            raise RuntimeError("no model")

    store.embedder = Broken()
    status = store.prewarm()

    assert status["embedder"] == {"ok": False, "error": "RuntimeError"}
    assert status["vectors"]["ok"] is True
