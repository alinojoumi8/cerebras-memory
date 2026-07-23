"""Derived USearch HNSW sidecar support.

SQLite embeddings remain authoritative. The sidecar is used only when its
recorded database generation exactly matches the current generation.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import threading
from typing import Any, Sequence

import numpy as np

from config import VectorSearchSettings


class VectorIndexUnavailable(RuntimeError):
    pass


class UsearchVectorIndex:
    def __init__(self, settings: VectorSearchSettings, *, model: str, dimensions: int):
        self.settings = settings
        self.model = model
        self.dimensions = dimensions
        model_key = hashlib.sha256(f"{model}\0{dimensions}".encode("utf-8")).hexdigest()[:16]
        self.path = Path(settings.index_dir) / f"chunks-{model_key}.usearch"
        self._index: Any | None = None
        self._loaded_mtime_ns: int | None = None
        self._lock = threading.Lock()
        self._last_error: str | None = None

    def _load(self) -> Any:
        if not self.path.is_file():
            raise VectorIndexUnavailable("ann_index_missing")
        mtime_ns = self.path.stat().st_mtime_ns
        if self._index is not None and self._loaded_mtime_ns == mtime_ns:
            return self._index
        with self._lock:
            if self._index is not None and self._loaded_mtime_ns == mtime_ns:
                return self._index
            try:
                from usearch.index import Index

                # Load into memory instead of viewing the file so Windows can
                # atomically replace the sidecar during a later refresh.
                self._index = Index.restore(str(self.path), view=False)
                self._loaded_mtime_ns = mtime_ns
                self._last_error = None
                return self._index
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
                raise VectorIndexUnavailable(self._last_error) from exc

    def search(self, query_vector: np.ndarray, limit: int) -> list[tuple[float, int]]:
        try:
            matches = self._load().search(
                np.ascontiguousarray(query_vector, dtype=np.float32),
                max(1, int(limit)),
                threads=1,
            )
            return [
                (1.0 - float(distance), int(key))
                for key, distance in zip(matches.keys, matches.distances, strict=True)
            ]
        except VectorIndexUnavailable:
            raise
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"[:500]
            raise VectorIndexUnavailable(self._last_error) from exc

    def build(
        self,
        keys: Sequence[int] | np.ndarray,
        vectors: Sequence[np.ndarray] | np.ndarray,
    ) -> dict[str, Any]:
        from usearch.index import Index

        key_array = np.asarray(keys, dtype=np.uint64)
        vector_array = np.asarray(vectors, dtype=np.float32)
        if vector_array.ndim != 2 or vector_array.shape[1] != self.dimensions:
            raise ValueError("Unexpected vector matrix dimensions for ANN build")
        if len(key_array) != len(vector_array):
            raise ValueError("ANN keys and vectors have different lengths")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        index = Index(
            ndim=self.dimensions,
            metric="cos",
            dtype=self.settings.dtype,
            connectivity=self.settings.connectivity,
            expansion_add=self.settings.expansion_add,
            expansion_search=self.settings.expansion_search,
        )
        try:
            if len(key_array):
                # A single deterministic builder avoids the lower-recall graph
                # variation observed when USearch auto-selects every CPU thread.
                index.add(key_array, vector_array, threads=1)
            index.save(str(temporary))
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
        with self._lock:
            self._index = index
            self._loaded_mtime_ns = self.path.stat().st_mtime_ns
            self._last_error = None
        return {
            "path": str(self.path),
            "vectors": int(len(key_array)),
            "dimensions": self.dimensions,
            "dtype": self.settings.dtype,
        }

    def status(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.path.is_file(),
            "loaded": self._index is not None,
            "last_error": self._last_error,
        }
