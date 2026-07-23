"""Optional local throughput check for the pinned FastEmbed model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from embeddings import FastEmbedder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()
    settings = load_settings()
    embedder = FastEmbedder(
        settings.embedding_model,
        settings.embedding_dimensions,
        settings.model_cache_dir,
    )
    base = "Local retrieval benchmark paragraph with implementation facts and citations. " * 24
    texts = [f"{base}{index}" for index in range(max(1, args.count))]
    started = perf_counter()
    vectors = embedder.embed_for_ingestion(texts)
    elapsed = perf_counter() - started
    print(
        {
            "vectors": len(vectors),
            "dimensions": len(vectors[0]),
            "seconds": round(elapsed, 2),
            "vectors_per_second": round(len(vectors) / elapsed, 2),
        }
    )


if __name__ == "__main__":
    main()
