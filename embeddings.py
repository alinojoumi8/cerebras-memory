"""Local embedding backends.

Production always uses FastEmbed/ONNX.  A deterministic hashing backend exists
only as an explicit test hook so integration tests never need network access.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
import re
import threading
from typing import Iterable, Protocol

import numpy as np

# Bounded so a long-lived STDIO worker cannot accumulate query vectors without
# limit; large enough that paging through one result set never re-embeds.
_QUERY_CACHE_SIZE = 256


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
    def __init__(
        self,
        model_name: str,
        dimensions: int,
        cache_dir: Path,
        query_prefix: str = "",
        threads: int = 2,
    ):
        self.model_name = model_name
        self.dimensions = dimensions
        self.cache_dir = Path(cache_dir)
        self.query_prefix = query_prefix
        self.threads = max(1, int(threads))
        self._model = None
        self._model_lock = threading.Lock()
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _load(self):
        # Double-checked locking, the same shape as ``FlashRankReranker._load``.
        # The unlocked first read keeps steady-state calls lock-free; the lock
        # makes a concurrent cold start build exactly one model. Two requests
        # arriving together previously each constructed a ``TextEmbedding``, and
        # because ``lazy_load=True`` defers session creation, they then raced
        # inside FastEmbed's own loader. One STDIO client never hit this; one
        # process serving several agents does.
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            from fastembed import TextEmbedding

            # Thread count is a caller decision, not a constant. Serving keeps a
            # small cap because every desktop client spawns its own STDIO worker
            # and none may reserve the machine; a batch ingest is the only
            # process running and should use the box. Hardcoding the serving cap
            # here previously left a full re-index on a 32-core machine running
            # at 2 threads.
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
                threads=self.threads,
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

    def _cache_query(self, text: str, vector: np.ndarray) -> np.ndarray:
        """Remember a query vector, bounded by simple FIFO eviction.

        Repeated and paginated searches otherwise pay a fresh ONNX forward pass
        every time. Callers treat vectors as read-only, so returning the cached
        array is safe.

        Insert and eviction are held under one lock. ``next(iter(...))`` over an
        ``OrderedDict`` another thread is mutating raises ``RuntimeError``, and
        two threads evicting the same key raise ``KeyError``. Serial calls from a
        single STDIO client never reached either branch concurrently.
        """

        with self._cache_lock:
            self._query_cache[text] = vector
            while len(self._query_cache) > _QUERY_CACHE_SIZE:
                self._query_cache.pop(next(iter(self._query_cache)))
        return vector

    def embed_query(self, text: str) -> np.ndarray:
        # Deliberately unlocked: a single ``dict.get`` is atomic under the GIL,
        # and a lookup racing an eviction can only miss, which costs one extra
        # forward pass rather than returning a wrong vector.
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        # FastEmbed's ``query_embed`` does not apply a model-specific retrieval
        # instruction: ``OnnxTextEmbedding`` never overrides it, so the base
        # implementation just calls ``embed``. Asymmetric models such as BGE are
        # trained with a query-side prefix, and omitting it encodes queries on
        # the passage-side manifold. Documents are correctly left unprefixed, so
        # applying it here needs no re-embedding.
        prepared = f"{self.query_prefix}{text}" if self.query_prefix else text
        vectors = [_normalize(vector) for vector in self._load().query_embed([prepared])]
        if len(vectors) != 1 or vectors[0].shape != (self.dimensions,):
            raise ValueError(f"Unexpected query embedding shape for {self.model_name}")
        return self._cache_query(text, vectors[0])

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
        # Measured on a 32-core box at 16 ingest threads: batch 16 -> 18.5 vec/s,
        # batch 64 -> 19.0, batch 256 -> 17.5. 64 is the peak and matches the
        # serving path, so both use one bounded size.
        return self._embed_values(values, batch_size=64, parallel=None)


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
