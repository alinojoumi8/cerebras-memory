from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from reranking import FlashRankReranker


class _Tokenizer:
    def encode(self, query: str, text: str):
        del query
        return SimpleNamespace(ids=list(range(int(text))))


class _Ranker:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.batches: list[list[int]] = []

    def rerank(self, request):
        lengths = [int(item["text"]) for item in request.passages]
        self.batches.append(lengths)
        return [dict(item, score=float(item["meta"]["score"])) for item in request.passages]


def test_flashrank_length_buckets_preserve_all_scores(settings_factory):
    base = settings_factory()
    settings = replace(base.reranker, enabled=True, batch_size=3)
    adapter = FlashRankReranker(settings)
    fake = _Ranker()
    adapter._ranker = fake
    passages = [
        {"id": f"p-{index}", "text": str(length), "meta": {"score": index}}
        for index, length in enumerate((500, 10, 400, 20, 300, 30, 200, 40))
    ]

    ranked = adapter.rerank("query", passages)

    assert [item["id"] for item in ranked] == [f"p-{index}" for index in reversed(range(8))]
    assert sum(len(batch) for batch in fake.batches) == len(passages)
    assert all(len(batch) <= 3 for batch in fake.batches)
    assert [max(batch) for batch in fake.batches] == [30, 300, 500]
