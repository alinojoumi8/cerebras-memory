from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from embeddings import HashingEmbedder
from models import IngestDocument
from reranking import RerankerUnavailable
from store import KnowledgeStore


class MarkerReranker:
    model_name = "test/marker-reranker"

    def __init__(self, marker: str):
        self.marker = marker

    def rerank(self, query, passages):
        return [
            {"id": passage["id"], "score": 10.0 if self.marker in passage["text"] else 0.1}
            for passage in passages
        ]

    def status(self):
        return {"enabled": True, "status": "ready", "model": self.model_name}


class BrokenReranker:
    model_name = "test/broken-reranker"

    def rerank(self, query, passages):
        raise RerankerUnavailable("fixture_model_missing")

    def status(self):
        return {"enabled": True, "status": "degraded", "model": self.model_name}


def _document(key: str, text: str, *, project: str = "alpha") -> IngestDocument:
    return IngestDocument(
        source="projects",
        source_key=key,
        title=key,
        text=text,
        timestamp=datetime.now(timezone.utc),
        project=project,
        uri=f"fixture://{key}",
    )


def _three_chunk_text(anchor: str, neighbor: str = "NEXTMARK") -> str:
    first = "opening context " + ("one " * 70)
    middle = f"middle evidence {anchor} " + ("two " * 65)
    last = f"closing context {neighbor} " + ("three " * 60)
    return "\n\n".join((first, middle, last))


def test_candidate_lookup_is_complete_across_batch_boundaries(settings_factory):
    """The batched ``chunk_pk IN (...)`` lookup must return every record.

    620 candidates span two _SQL_PARAM_BATCH windows, so a batching bug would
    drop the tail and silently shrink the result set rather than raise.
    """

    settings = settings_factory(candidate_limit=900)
    store = KnowledgeStore(settings, HashingEmbedder(32))
    for index in range(620):
        store.upsert_document(
            _document(f"bulk-{index}", f"corpus filler document number {index}")
        )

    # A query that matches nothing lexically leaves every vector candidate
    # absent from ``records``, so the whole rank set flows through the batched
    # lookup instead of arriving via the FTS pass.
    response = store.search_response(
        "zzqx unrelated probe term", limit=8, rerank=False, global_search=True
    )

    assert len(response["results"]) == 8
    assert len({result["document_id"] for result in response["results"]}) == 8
    assert all(result["snippet"] for result in response["results"])


def test_load_settings_clamps_candidate_limit(tmp_path):
    import json as _json

    from config import load_settings

    config_path = tmp_path / "config.json"
    config_path.write_text(
        _json.dumps({"candidate_limit": 5000, "database_path": str(tmp_path / "db.sqlite3")}),
        encoding="utf-8",
    )
    assert load_settings(config_path).candidate_limit == 200

    config_path.write_text(
        _json.dumps({"candidate_limit": 1, "database_path": str(tmp_path / "db.sqlite3")}),
        encoding="utf-8",
    )
    assert load_settings(config_path).candidate_limit == 10


def test_document_deduplication_adjacent_context_and_stable_anchor(settings_factory):
    settings = settings_factory()
    store = KnowledgeStore(settings, HashingEmbedder(32))
    written = store.upsert_document(_document("multi", _three_chunk_text("ORBITALNEEDLE")))
    store.upsert_document(_document("second", "ORBITALNEEDLE independent supporting document"))

    response = store.search_response("ORBITALNEEDLE", limit=8, rerank=False, global_search=True)
    results = response["results"]
    assert len({result["document_id"] for result in results}) == len(results)
    result = next(item for item in results if item["document_id"] == written.document_id)
    assert 1 <= len(result["context_chunks"]) <= 2
    assert result["chunk_id"] in {chunk["chunk_id"] for chunk in result["context_chunks"]}
    assert sum(chunk["anchor"] for chunk in result["context_chunks"]) == 1
    assert result["citation"].endswith(f"chunk={result['chunk_id']}")
    assert all(
        chunk["citation"].endswith(f"chunk={chunk['chunk_id']}")
        for chunk in result["context_chunks"]
    )
    assert response["retrieval"]["document_deduplication"] is True
    assert response["retrieval"]["max_context_chunks"] == 2


def test_reranker_selects_neighbor_and_orders_documents(settings_factory):
    base = settings_factory()
    settings = replace(base, reranker=replace(base.reranker, enabled=True))
    reranker = MarkerReranker("NEXTMARK")
    store = KnowledgeStore(settings, HashingEmbedder(32), reranker=reranker)
    preferred = store.upsert_document(
        _document("preferred", _three_chunk_text("rerank anchor", "NEXTMARK"))
    )
    store.upsert_document(_document("ordinary", _three_chunk_text("rerank anchor", "ordinary")))

    response = store.search_response("rerank anchor", limit=2, global_search=True)
    assert response["results"][0]["document_id"] == preferred.document_id
    assert response["results"][0]["score_stage"] == "reranker"
    assert response["results"][0]["rerank_score"] == 10.0
    assert len(response["results"][0]["context_chunks"]) == 2
    assert "NEXTMARK" in response["results"][0]["snippet"]
    assert response["retrieval"]["reranker"]["applied"] is True


def test_reranker_failure_is_fail_open(settings_factory):
    base = settings_factory()
    settings = replace(base, reranker=replace(base.reranker, enabled=True))
    store = KnowledgeStore(settings, HashingEmbedder(32), reranker=BrokenReranker())
    store.upsert_document(_document("fallback", "fallback retrieval evidence"))

    response = store.search_response("fallback retrieval", global_search=True)
    assert response["results"]
    assert response["results"][0]["score_stage"] == "rrf"
    assert response["results"][0]["rerank_score"] is None
    assert response["retrieval"]["reranker"]["status"] == "fallback"
    assert response["retrieval"]["reranker"]["degraded_reason"] == "fixture_model_missing"


def test_project_scope_resolution_order_and_windows_paths(settings_factory):
    settings = settings_factory()
    alpha_root = settings.projects_root / "Alpha"
    beta_root = settings.projects_root / "Beta"
    (alpha_root / "nested").mkdir(parents=True)
    beta_root.mkdir(parents=True)
    store = KnowledgeStore(settings, HashingEmbedder(32))
    store.upsert_document(_document("alpha", "scope marker", project="Alpha"))
    store.upsert_document(_document("beta", "scope marker", project="Beta"))

    explicit = store.search_response(
        "scope marker",
        project="Beta",
        roots=[alpha_root],
        cwd=alpha_root,
    )
    assert explicit["scope"] == {"project": "Beta", "origin": "explicit"}
    assert {result["project"] for result in explicit["results"]} == {"Beta"}

    inferred = store.search_response("scope marker", roots=[alpha_root / "nested"], cwd=beta_root)
    assert inferred["scope"] == {"project": "Alpha", "origin": "client_root"}
    assert {result["project"] for result in inferred["results"]} == {"Alpha"}

    case_variant = store.resolve_project_scope(
        project=None,
        global_search=False,
        roots=[Path(str(alpha_root).upper())],
        cwd=None,
    )
    assert case_variant == {"project": "Alpha", "origin": "client_root"}

    ambiguous = store.search_response("scope marker", roots=[alpha_root, beta_root])
    assert ambiguous["scope"] == {"project": None, "origin": "global_ambiguous_roots"}
    assert {result["project"] for result in ambiguous["results"]} == {"Alpha", "Beta"}

    cwd_scope = store.search_response(
        "scope marker",
        roots=[settings.projects_root / "unknown"],
        cwd=beta_root,
    )
    assert cwd_scope["scope"] == {"project": "Beta", "origin": "process_cwd"}

    global_response = store.search_response(
        "scope marker",
        global_search=True,
        roots=[alpha_root],
        cwd=alpha_root,
    )
    assert global_response["scope"] == {"project": None, "origin": "global_explicit"}

    with pytest.raises(ValueError, match="cannot be combined"):
        store.search_response("scope marker", project="Alpha", global_search=True)


def test_edge_anchor_returns_only_available_neighbor(settings_factory):
    store = KnowledgeStore(settings_factory(), HashingEmbedder(32))
    result = store.upsert_document(
        _document("edge", "EDGEMARK " + ("first " * 70) + "\n\n" + ("second " * 70))
    )
    match = next(
        item
        for item in store.search("EDGEMARK", global_search=True, rerank=False)
        if item["document_id"] == result.document_id
    )
    ordinals = [chunk["ordinal"] for chunk in match["context_chunks"]]
    assert ordinals == sorted(ordinals)
    assert len(ordinals) <= 2
    assert min(ordinals) == 0
