from __future__ import annotations

import numpy as np

from embeddings import FastEmbedder


def test_ingestion_embedding_never_spawns_fastembed_worker_processes(tmp_path):
    calls = []

    class FixtureModel:
        def embed(self, values, *, batch_size, parallel):
            calls.append(
                {
                    "count": len(values),
                    "batch_size": batch_size,
                    "parallel": parallel,
                }
            )
            return [np.ones(4, dtype=np.float32) for _ in values]

    embedder = FastEmbedder("fixture/model", 4, tmp_path)
    embedder._model = FixtureModel()

    vectors = embedder.embed_for_ingestion([f"document {index}" for index in range(64)])

    assert len(vectors) == 64
    assert calls == [{"count": 64, "batch_size": 16, "parallel": None}]
