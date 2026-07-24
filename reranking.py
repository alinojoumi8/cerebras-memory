"""Optional local cross-encoder reranking with an explicit offline boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import threading
from typing import Any, Protocol, Sequence

from config import RerankerSettings


class RerankerUnavailable(RuntimeError):
    """Raised when a local reranker cannot be used without downloading data."""


class Reranker(Protocol):
    model_name: str

    def rerank(self, query: str, passages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def status(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RerankPassage:
    id: str
    text: str
    meta: dict[str, Any]


class FlashRankReranker:
    """Lazy FlashRank adapter that never downloads during an ordinary query."""

    def __init__(self, settings: RerankerSettings):
        self.settings = settings
        self.model_name = settings.model
        self.cache_dir = Path(settings.cache_dir)
        self.model_dir = self.cache_dir / self.model_name
        self._ranker: Any | None = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._last_error: str | None = None
        self._intra_op_threads: int | None = None

    def is_cached(self) -> bool:
        return (
            self.model_dir.is_dir()
            and any(self.model_dir.glob("*.onnx"))
            and (self.model_dir / "tokenizer.json").is_file()
        )

    def _load(self, *, allow_download: bool) -> Any:
        if self._ranker is not None:
            return self._ranker
        if not allow_download and not self.is_cached():
            raise RerankerUnavailable("reranker_model_not_cached")
        with self._lock:
            if self._ranker is not None:
                return self._ranker
            try:
                from flashrank import Ranker

                self.cache_dir.mkdir(parents=True, exist_ok=True)
                ranker = Ranker(
                    model_name=self.model_name,
                    cache_dir=str(self.cache_dir),
                    max_length=self.settings.max_length,
                    log_level="ERROR",
                )
                # FlashRank 0.2.10 creates an ONNX session with conservative
                # defaults.  On this 32-logical-core Windows host that misses
                # the warm latency gate by more than 2x. Recreate the already-
                # local session with bounded CPU parallelism; no model path or
                # provider changes, and no network access is introduced.
                session = getattr(ranker, "session", None)
                model_path = getattr(session, "_model_path", None)
                if model_path:
                    import onnxruntime as ort

                    logical_cpus = max(1, os.cpu_count() or 1)
                    threads = min(
                        logical_cpus,
                        self.settings.intra_op_threads,
                    )
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = threads
                    options.inter_op_num_threads = 1
                    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    ranker.session = ort.InferenceSession(
                        model_path,
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                    self._intra_op_threads = threads
                self._ranker = ranker
                self._last_error = None
                return self._ranker
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
                raise RerankerUnavailable(self._last_error) from exc

    def warm(self) -> dict[str, Any]:
        self._load(allow_download=True)
        return self.status()

    def rerank(self, query: str, passages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not passages:
            return []
        ranker = self._load(allow_download=False)
        from flashrank import RerankRequest

        try:
            with self._inference_lock:
                # FlashRank pads one request to its longest pair. Conversation
                # evidence mixes prose, code, and markup, so a single dense
                # passage could otherwise force all 40 context variants to
                # the 512-token ceiling. Length-bucketed micro-batches keep
                # every pair and the configured maximum while avoiding that
                # unnecessary padding. Cross-encoder scores are independent
                # per pair and therefore remain comparable across batches.
                indexed = list(enumerate(passages))
                tokenizer = getattr(ranker, "tokenizer", None)

                def token_length(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
                    position, passage = item
                    if tokenizer is None:
                        return (len(str(passage.get("text", ""))), position)
                    encoded = tokenizer.encode(query, str(passage.get("text", "")))
                    return (len(encoded.ids), position)

                indexed.sort(key=token_length)
                results: list[dict[str, Any]] = []
                batch_size = self.settings.batch_size
                for offset in range(0, len(indexed), batch_size):
                    batch = [item[1] for item in indexed[offset : offset + batch_size]]
                    ranked = ranker.rerank(RerankRequest(query=query, passages=batch))
                    results.extend(dict(item) for item in ranked)
                original_order = {
                    str(passage.get("id")): position for position, passage in enumerate(passages)
                }
                results.sort(
                    key=lambda item: (
                        -float(item.get("score", 0.0)),
                        original_order.get(str(item.get("id")), len(original_order)),
                    )
                )
            self._last_error = None
            return results
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"[:500]
            raise RerankerUnavailable(self._last_error) from exc

    def status(self) -> dict[str, Any]:
        if not self.settings.enabled:
            state = "disabled"
        elif self._last_error:
            state = "degraded"
        elif self._ranker is not None:
            state = "ready"
        elif self.is_cached():
            state = "cached"
        else:
            state = "model_not_cached"
        return {
            "enabled": self.settings.enabled,
            "status": state,
            "model": self.model_name,
            "cache_path": str(self.cache_dir),
            "model_cached": self.is_cached(),
            "loaded": self._ranker is not None,
            "candidate_documents": self.settings.candidate_documents,
            "max_length": self.settings.max_length,
            "batch_size": self.settings.batch_size,
            "intra_op_threads": (
                self._intra_op_threads
                if self._intra_op_threads is not None
                else self.settings.intra_op_threads
            ),
            "last_error": self._last_error,
        }


class DisabledReranker:
    model_name = "disabled"

    def rerank(self, query: str, passages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        raise RerankerUnavailable("reranker_disabled")

    def status(self) -> dict[str, Any]:
        return {"enabled": False, "status": "disabled", "model": self.model_name}
