"""MCP surface honesty, production guards, and query-embedding caching."""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

import mcp_server
from embeddings import _QUERY_CACHE_SIZE, FastEmbedder, HashingEmbedder
from store import KnowledgeStore


class _FixtureModel:
    """Counts forward passes so cache hits are observable."""

    def __init__(self) -> None:
        self.calls = 0

    def query_embed(self, texts, **_kwargs):
        self.calls += 1
        for _text in texts:
            yield np.ones(4, dtype=np.float32)


def test_every_mcp_tool_is_async_so_none_can_block_the_event_loop():
    """FastMCP calls a sync tool inline in the request coroutine.

    ``func_metadata.call_fn_with_arg_validation`` falls through to a plain
    ``fn(...)`` for non-async tools, so a synchronous tool holds the loop for its
    whole duration - over a second for a search.
    """

    for name in ("kb_search", "kb_get", "kb_save_memory", "kb_stats"):
        tool = getattr(mcp_server, name)
        assert inspect.iscoroutinefunction(tool), f"{name} must be async"


def test_kb_stats_is_read_only_as_annotated(settings_factory, monkeypatch):
    """``readOnlyHint=True`` is what a client uses to auto-approve a call.

    ``stats()`` used to recover stale refresh leases, taking BEGIN IMMEDIATE and
    mutating refresh_runs on every call.
    """

    settings = settings_factory()
    store = KnowledgeStore(settings, HashingEmbedder(32))

    calls: list[str] = []
    monkeypatch.setattr(
        store,
        "recover_stale_refreshes",
        lambda: calls.append("recover") or 0,
    )

    store.stats(recover=False)
    assert calls == []

    store.stats(recover=True)
    assert calls == ["recover"]


def test_test_embedder_hook_cannot_downgrade_a_real_configuration(
    settings_factory,
    monkeypatch,
):
    """An ambient env var must not swap in a bag-of-words hasher.

    MCP clients control the server process environment, so this would otherwise
    be a silent, remote-triggerable retrieval-quality collapse.
    """

    monkeypatch.setenv("CEREBRAS_MEMORY_TEST_EMBEDDER", "1")

    production = settings_factory(embedding_model="BAAI/bge-small-en-v1.5")
    with pytest.raises(RuntimeError, match="Refusing to silently downgrade"):
        KnowledgeStore(production)

    # The sentinel model opts in explicitly, so the fixtures still work.
    store = KnowledgeStore(settings_factory())
    assert store.embedder.model_name == "test/hash-v1"


def test_query_embeddings_are_cached_and_bounded(tmp_path):
    embedder = FastEmbedder("fixture/model", 4, tmp_path)
    model = _FixtureModel()
    embedder._model = model

    first = embedder.embed_query("same question")
    second = embedder.embed_query("same question")
    assert model.calls == 1
    assert np.array_equal(first, second)

    embedder.embed_query("different question")
    assert model.calls == 2

    for index in range(_QUERY_CACHE_SIZE + 50):
        embedder.embed_query(f"query {index}")
    assert len(embedder._query_cache) <= _QUERY_CACHE_SIZE


def test_query_prefix_is_applied_to_queries_only(tmp_path):
    embedder = FastEmbedder("fixture/model", 4, tmp_path, query_prefix="PREFIX: ")
    seen: list[str] = []

    class _Recorder:
        def query_embed(self, texts, **_kwargs):
            seen.extend(texts)
            for _text in texts:
                yield np.ones(4, dtype=np.float32)

    embedder._model = _Recorder()
    embedder.embed_query("what is the retry budget")

    assert seen == ["PREFIX: what is the retry budget"]


def test_benchmark_separates_cold_and_warm_instead_of_hiding_them(
    settings_factory,
    monkeypatch,
):
    """A median over [cold, warm, warm] reports the warm figure while looking
    like it summarised every run."""

    settings = settings_factory()
    store = KnowledgeStore(settings, HashingEmbedder(32))
    timings = iter([0.0, 1.0, 1.2, 1.3])
    monkeypatch.setattr("store.time.perf_counter", lambda: next(timings))

    result = store.benchmark_vector_search(runs=2)

    assert result["cold_ms"] == pytest.approx(1000.0)
    assert result["warm_median_ms"] == pytest.approx(100.0)
    # median_ms is the unrounded warm figure kept for the stored column.
    assert result["median_ms"] == pytest.approx(result["warm_median_ms"])


def test_environment_hook_is_still_honoured_when_opted_in(monkeypatch, settings_factory):
    monkeypatch.setenv("CEREBRAS_MEMORY_TEST_EMBEDDER", "1")
    store = KnowledgeStore(settings_factory())
    # The reranker is disabled alongside the embedder by the same hook.
    assert store.reranker.status()["enabled"] is False
    assert os.environ["CEREBRAS_MEMORY_TEST_EMBEDDER"] == "1"
