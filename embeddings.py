"""Local embedding backends.

Production always uses FastEmbed/ONNX.  A deterministic hashing backend exists
only as an explicit test hook so integration tests never need network access.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Iterable, Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed(self, texts: Iterable[str]) -> list[np.ndarray]: ...


def _normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm:
        array = array / norm
    return np.ascontiguousarray(array, dtype=np.float32)


class FastEmbedder:
    def __init__(self, model_name: str, dimensions: int, cache_dir: Path):
        self.model_name = model_name
        self.dimensions = dimensions
        self.cache_dir = Path(cache_dir)
        self._model = None

    def _load(self):
        if self._model is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
                threads=max(1, min(2, __import__("os").cpu_count() or 1)),
                lazy_load=True,
            )
        return self._model

    def _embed_values(
        self,
        values: list[str],
        *,
        batch_size: int,
        parallel: int | None,
    ) -> list[np.ndarray]:
        vectors = [
            _normalize(vector)
            for vector in self._load().embed(
                values,
                batch_size=min(batch_size, len(values)),
                parallel=parallel,
            )
        ]
        for vector in vectors:
            if vector.shape != (self.dimensions,):
                raise ValueError(
                    f"Embedding dimension mismatch for {self.model_name}: "
                    f"expected {self.dimensions}, got {vector.shape}"
                )
        return vectors

    def embed(self, texts: Iterable[str]) -> list[np.ndarray]:
        values = list(texts)
        if not values:
            return []
        # A single generator call lets ONNX reuse its allocation arena across
        # the full source.  A bounded internal batch avoids the multi-gigabyte
        # arena growth seen with FastEmbed's default batch of 256 on Windows.
        return self._embed_values(values, batch_size=64, parallel=None)

    def embed_query(self, text: str) -> np.ndarray:
        vectors = [_normalize(vector) for vector in self._load().query_embed([text])]
        if len(vectors) != 1 or vectors[0].shape != (self.dimensions,):
            raise ValueError(f"Unexpected query embedding shape for {self.model_name}")
        return vectors[0]

    def embed_for_ingestion(self, texts: Iterable[str]) -> list[np.ndarray]:
        """Embed offline refreshes in one bounded ONNX process.

        FastEmbed's ``parallel`` option starts one full model process per
        worker. On Windows each process can reserve multiple gigabytes, so
        even a modest eight-way refresh can exhaust a desktop. A single
        session with small batches is slower but keeps memory bounded and
        still reuses the ONNX arena across the source.
        """

        values = list(texts)
        if not values:
            return []
        return self._embed_values(values, batch_size=16, parallel=None)


class HashingEmbedder:
    """Deterministic, dependency-light embedder used by tests only."""

    def __init__(self, dimensions: int = 384, model_name: str = "test/hash-v1"):
        self.model_name = model_name
        self.dimensions = dimensions

    def embed(self, texts: Iterable[str]) -> list[np.ndarray]:
        output: list[np.ndarray] = []
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=np.float32)
            for token in re.findall(r"[\w'-]+", text.casefold()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            output.append(_normalize(vector))
        return output

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
