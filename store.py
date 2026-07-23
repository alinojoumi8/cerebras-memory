"""Versioned SQLite storage and local hybrid retrieval."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import threading
import time
from typing import Any, Iterable, Sequence

import numpy as np

from chunking import chunk_text
from config import Settings, load_settings
from distillation import (
    AGENT_SOURCES,
    DistillationInvalid,
    DistillationUnavailable,
    Distiller,
    create_distiller,
    flatten_distillation,
    reconstruct_chunks,
    sanitize_role_contradictions,
    segment_dialogue,
    validate_distillation,
)
from embeddings import Embedder, FastEmbedder, HashingEmbedder
from models import IngestDocument
from redaction import redact_text
from reranking import DisabledReranker, FlashRankReranker, Reranker, RerankerUnavailable
from runlock import distillation_lock
from vector_index import UsearchVectorIndex, VectorIndexUnavailable


SCHEMA_VERSION = 2
_UTC = timezone.utc
_SCHEMA_INITIALIZATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class WriteResult:
    document_id: str
    status: str
    chunks: int


@dataclass
class _PreparedDocument:
    source: str
    source_key: str
    kind: str
    title: str
    text: str
    uri: str | None
    timestamp: str
    project: str | None
    metadata_json: str
    content_hash: str
    document_id: str
    existing: bool = False
    unchanged: bool = False
    pieces: list[str] = field(default_factory=list)
    vectors: list[np.ndarray] = field(default_factory=list)
    chunk_count: int = 0


@dataclass(frozen=True)
class _ExactVectorSnapshot:
    generation: int
    model: str
    dimensions: int
    keys: np.ndarray
    vectors: np.ndarray
    positions: dict[int, int]
    sources: np.ndarray
    projects: np.ndarray
    timestamps: np.ndarray


def _utc_now() -> datetime:
    return datetime.now(_UTC)


def _iso(value: datetime | str | None) -> str:
    if value is None:
        return _utc_now().isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        parsed = parse_timestamp(value)
        return parsed.isoformat().replace("+00:00", "Z")
    if value.tzinfo is None:
        value = value.replace(tzinfo=_UTC)
    return value.astimezone(_UTC).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif not value:
        return datetime.fromtimestamp(0, _UTC)
    else:
        text = str(value).strip()
        if text.isdigit():
            number = float(text)
            if number > 10_000_000_000:
                number /= 1000
            return datetime.fromtimestamp(number, _UTC)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return datetime.fromtimestamp(0, _UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_UTC)
    return parsed.astimezone(_UTC)


def stable_document_id(source: str, source_key: str) -> str:
    digest = hashlib.sha256(f"{source}\0{source_key}".encode("utf-8")).hexdigest()
    return f"doc_{digest[:32]}"


def stable_chunk_id(document_id: str, ordinal: int) -> str:
    digest = hashlib.sha256(f"{document_id}\0{ordinal}".encode("utf-8")).hexdigest()
    return f"chk_{digest[:32]}"


def _redact_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_json_value(item) for item in value]
    if isinstance(value, dict):
        return {redact_text(str(key)): _redact_json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value))


class KnowledgeStore:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        *,
        reranker: Reranker | None = None,
        distiller: Distiller | None = None,
    ):
        self.settings = settings or load_settings()
        self.settings.ensure_runtime_directories()
        self.database_path = Path(self.settings.database_path)
        if embedder is None:
            if os.environ.get("CEREBRAS_MEMORY_TEST_EMBEDDER") == "1":
                embedder = HashingEmbedder(dimensions=self.settings.embedding_dimensions)
            else:
                embedder = FastEmbedder(
                    self.settings.embedding_model,
                    self.settings.embedding_dimensions,
                    self.settings.model_cache_dir,
                )
        self.embedder = embedder
        if reranker is None:
            if (
                self.settings.reranker.enabled
                and os.environ.get("CEREBRAS_MEMORY_TEST_EMBEDDER") != "1"
            ):
                reranker = FlashRankReranker(self.settings.reranker)
            else:
                reranker = DisabledReranker()
        self.reranker = reranker
        self.distiller = distiller or create_distiller(self.settings.distillation)
        self.vector_index = UsearchVectorIndex(
            self.settings.vector_search,
            model=self.embedder.model_name,
            dimensions=self.embedder.dimensions,
        )
        self._exact_vector_lock = threading.Lock()
        self._exact_vector_snapshot: _ExactVectorSnapshot | None = None
        self._distillation_request_slots = threading.BoundedSemaphore(
            self.settings.distillation.max_concurrent_requests
        )
        self._distillation_checkpoint_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _initialize(self) -> None:
        with _SCHEMA_INITIALIZATION_LOCK:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                current = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if current > SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Database schema {current} is newer than supported schema {SCHEMA_VERSION}"
                    )
                if current < 1:
                    connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        source_key TEXT NOT NULL,
                        kind TEXT NOT NULL CHECK(kind IN ('derived', 'memory')),
                        title TEXT NOT NULL,
                        uri TEXT,
                        timestamp TEXT NOT NULL,
                        project TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        content_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(source, source_key)
                    );
                    CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
                    CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project);
                    CREATE INDEX IF NOT EXISTS idx_documents_timestamp ON documents(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents(kind);

                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                        id TEXT NOT NULL UNIQUE,
                        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        ordinal INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        embedding_model TEXT NOT NULL,
                        embedding_dimensions INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(document_id, ordinal)
                    );
                    CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);
                    CREATE INDEX IF NOT EXISTS idx_chunks_model ON chunks(embedding_model);

                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        title,
                        content,
                        content='chunks',
                        content_rowid='chunk_pk',
                        tokenize='unicode61 remove_diacritics 2'
                    );
                    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, title, content)
                        VALUES (new.chunk_pk, new.title, new.content);
                    END;
                    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, title, content)
                        VALUES ('delete', old.chunk_pk, old.title, old.content);
                    END;
                    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, title, content)
                        VALUES ('delete', old.chunk_pk, old.title, old.content);
                        INSERT INTO chunks_fts(rowid, title, content)
                        VALUES (new.chunk_pk, new.title, new.content);
                    END;

                    CREATE TABLE IF NOT EXISTS ingest_state (
                        source TEXT PRIMARY KEY,
                        watermark TEXT,
                        status TEXT NOT NULL,
                        last_started_at TEXT,
                        last_success_at TEXT,
                        last_failure_at TEXT,
                        scanned INTEGER NOT NULL DEFAULT 0,
                        imported INTEGER NOT NULL DEFAULT 0,
                        skipped INTEGER NOT NULL DEFAULT 0,
                        failures INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT
                    );
                    INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                    VALUES (1, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
                    PRAGMA user_version = 1;
                    COMMIT;
                    """
                    )
                    current = 1
                if current < 2:
                    connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS distillations (
                        distillation_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                        id TEXT NOT NULL UNIQUE,
                        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        unit_ordinal INTEGER NOT NULL,
                        input_hash TEXT NOT NULL,
                        start_ordinal INTEGER NOT NULL,
                        end_ordinal INTEGER NOT NULL,
                        summary_json TEXT NOT NULL,
                        search_text TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        embedding_model TEXT NOT NULL,
                        embedding_dimensions INTEGER NOT NULL,
                        distiller_model TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(document_id, input_hash, distiller_model, prompt_version)
                    );
                    CREATE INDEX IF NOT EXISTS idx_distillations_document
                        ON distillations(document_id, unit_ordinal);
                    CREATE INDEX IF NOT EXISTS idx_distillations_embedding_model
                        ON distillations(embedding_model, embedding_dimensions);

                    CREATE VIRTUAL TABLE IF NOT EXISTS distillations_fts USING fts5(
                        search_text,
                        content='distillations',
                        content_rowid='distillation_pk',
                        tokenize='unicode61 remove_diacritics 2'
                    );
                    CREATE TRIGGER IF NOT EXISTS distillations_ai AFTER INSERT ON distillations BEGIN
                        INSERT INTO distillations_fts(rowid, search_text)
                        VALUES (new.distillation_pk, new.search_text);
                    END;
                    CREATE TRIGGER IF NOT EXISTS distillations_ad AFTER DELETE ON distillations BEGIN
                        INSERT INTO distillations_fts(distillations_fts, rowid, search_text)
                        VALUES ('delete', old.distillation_pk, old.search_text);
                    END;
                    CREATE TRIGGER IF NOT EXISTS distillations_au AFTER UPDATE ON distillations BEGIN
                        INSERT INTO distillations_fts(distillations_fts, rowid, search_text)
                        VALUES ('delete', old.distillation_pk, old.search_text);
                        INSERT INTO distillations_fts(rowid, search_text)
                        VALUES (new.distillation_pk, new.search_text);
                    END;

                    CREATE TABLE IF NOT EXISTS distillation_state (
                        document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                        status TEXT NOT NULL,
                        model TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        units_total INTEGER NOT NULL DEFAULT 0,
                        units_ready INTEGER NOT NULL DEFAULT 0,
                        failures INTEGER NOT NULL DEFAULT 0,
                        last_attempt_at TEXT,
                        last_success_at TEXT,
                        last_error TEXT
                    );

                    CREATE TABLE IF NOT EXISTS distillation_unit_state (
                        unit_state_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        input_hash TEXT NOT NULL,
                        unit_ordinal INTEGER NOT NULL,
                        start_ordinal INTEGER NOT NULL,
                        end_ordinal INTEGER NOT NULL,
                        distiller_model TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('pending', 'ready', 'failed')),
                        attempts INTEGER NOT NULL DEFAULT 0,
                        last_attempt_at TEXT,
                        last_success_at TEXT,
                        last_error TEXT,
                        UNIQUE(document_id, input_hash, distiller_model, prompt_version)
                    );
                    CREATE INDEX IF NOT EXISTS idx_distillation_unit_state_status
                        ON distillation_unit_state(status, document_id);

                    CREATE TABLE IF NOT EXISTS distillation_pilot_documents (
                        document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                        source TEXT NOT NULL,
                        selected_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_distillation_pilot_source
                        ON distillation_pilot_documents(source, selected_at);

                    CREATE TABLE IF NOT EXISTS vector_index_state (
                        id INTEGER PRIMARY KEY CHECK(id = 1),
                        data_generation INTEGER NOT NULL DEFAULT 0,
                        built_generation INTEGER,
                        embedding_model TEXT,
                        embedding_dimensions INTEGER,
                        chunk_count INTEGER NOT NULL DEFAULT 0,
                        backend TEXT NOT NULL DEFAULT 'exact',
                        status TEXT NOT NULL DEFAULT 'exact',
                        exact_benchmark_ms REAL,
                        index_path TEXT,
                        built_at TEXT,
                        last_error TEXT,
                        updated_at TEXT NOT NULL
                    );
                    INSERT OR IGNORE INTO vector_index_state(
                        id, data_generation, chunk_count, backend, status, updated_at
                    ) VALUES (
                        1,
                        CASE WHEN EXISTS(SELECT 1 FROM chunks) THEN 1 ELSE 0 END,
                        (SELECT COUNT(*) FROM chunks),
                        'exact',
                        'exact',
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    );

                    INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                    VALUES (2, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                    )
                # Early v2 builds aggregated retry state at document level.
                # This additive, idempotent repair gives existing v2 databases
                # the required per-unit pending/failed state without changing
                # the schema version or any raw document/chunk identity.
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS distillation_unit_state (
                        unit_state_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        input_hash TEXT NOT NULL,
                        unit_ordinal INTEGER NOT NULL,
                        start_ordinal INTEGER NOT NULL,
                        end_ordinal INTEGER NOT NULL,
                        distiller_model TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('pending', 'ready', 'failed')),
                        attempts INTEGER NOT NULL DEFAULT 0,
                        last_attempt_at TEXT,
                        last_success_at TEXT,
                        last_error TEXT,
                        UNIQUE(document_id, input_hash, distiller_model, prompt_version)
                    );
                    CREATE INDEX IF NOT EXISTS idx_distillation_unit_state_status
                        ON distillation_unit_state(status, document_id);
                    CREATE TABLE IF NOT EXISTS distillation_pilot_documents (
                        document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                        source TEXT NOT NULL,
                        selected_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_distillation_pilot_source
                        ON distillation_pilot_documents(source, selected_at);
                    INSERT OR IGNORE INTO distillation_unit_state(
                        document_id, input_hash, unit_ordinal, start_ordinal, end_ordinal,
                        distiller_model, prompt_version, status, attempts,
                        last_attempt_at, last_success_at, last_error
                    )
                    SELECT document_id, input_hash, unit_ordinal, start_ordinal, end_ordinal,
                           distiller_model, prompt_version, 'ready', 0,
                           NULL, updated_at, NULL
                    FROM distillations;
                    """
                )

    def schema_version(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    @staticmethod
    def _bump_vector_generation(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            UPDATE vector_index_state
            SET data_generation = data_generation + 1,
                chunk_count = (SELECT COUNT(*) FROM chunks),
                status = CASE WHEN backend = 'hnsw' THEN 'stale' ELSE 'exact' END,
                updated_at = ?
            WHERE id = 1
            """,
            (_iso(None),),
        )

    def _document_needs_embedding(self, connection: sqlite3.Connection, document_id: str) -> bool:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN embedding_model = ? AND embedding_dimensions = ? THEN 1 ELSE 0 END) AS current
            FROM chunks WHERE document_id = ?
            """,
            (self.embedder.model_name, self.embedder.dimensions, document_id),
        ).fetchone()
        return int(row["total"] or 0) == 0 or int(row["current"] or 0) != int(row["total"] or 0)

    def _prepare_document(self, document: IngestDocument) -> _PreparedDocument:
        safe_source = redact_text(document.source.strip().casefold())
        safe_key = redact_text(document.source_key.strip())
        safe_title = redact_text(document.title.strip())[:1000] or "Untitled"
        safe_text = redact_text(document.text).strip()
        safe_uri = redact_text(document.uri) if document.uri else None
        safe_project = redact_text(document.project)[:500] if document.project else None
        safe_metadata = _redact_json_value(document.metadata)
        if not safe_text:
            raise ValueError("Cannot index an empty document")
        if document.kind not in {"derived", "memory"}:
            raise ValueError(f"Unsupported document kind: {document.kind}")

        document_id = stable_document_id(safe_source, safe_key)
        content_hash = hashlib.sha256(safe_text.encode("utf-8")).hexdigest()
        timestamp = _iso(document.normalized_timestamp())
        metadata_json = json.dumps(safe_metadata, ensure_ascii=False, sort_keys=True)
        return _PreparedDocument(
            source=safe_source,
            source_key=safe_key,
            kind=document.kind,
            title=safe_title,
            text=safe_text,
            uri=safe_uri,
            timestamp=timestamp,
            project=safe_project,
            metadata_json=metadata_json,
            content_hash=content_hash,
            document_id=document_id,
        )

    def upsert_documents(
        self,
        documents: Sequence[IngestDocument],
        *,
        force: bool = False,
    ) -> list[WriteResult]:
        """Redact, embed in bounded batches, and atomically replace documents.

        All embeddings are completed before the write transaction begins.  A
        model failure therefore cannot leave a partially written source pass.
        """

        prepared = [self._prepare_document(document) for document in documents]
        if not prepared:
            return []
        with self._connect() as connection:
            for item in prepared:
                existing = connection.execute(
                    "SELECT content_hash FROM documents WHERE source = ? AND source_key = ?",
                    (item.source, item.source_key),
                ).fetchone()
                item.existing = existing is not None
                item.unchanged = bool(
                    existing is not None
                    and existing["content_hash"] == item.content_hash
                    and not force
                    and not self._document_needs_embedding(connection, item.document_id)
                )
                if item.unchanged:
                    item.chunk_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                            (item.document_id,),
                        ).fetchone()[0]
                    )
                else:
                    item.pieces = chunk_text(
                        item.text,
                        target_size=self.settings.chunk_size,
                        overlap=self.settings.chunk_overlap,
                    )

        locations: list[tuple[_PreparedDocument, str]] = [
            (item, piece)
            for item in prepared
            if not item.unchanged
            for piece in item.pieces
        ]
        ingestion_embed = getattr(self.embedder, "embed_for_ingestion", self.embedder.embed)
        vectors = ingestion_embed([piece for _, piece in locations])
        if len(vectors) != len(locations):
            raise RuntimeError("Embedding backend returned an unexpected vector count")
        for (item, _), vector in zip(locations, vectors, strict=True):
            if np.asarray(vector).shape != (self.embedder.dimensions,):
                raise ValueError("Embedding backend returned an unexpected vector dimension")
            item.vectors.append(np.asarray(vector, dtype=np.float32))

        now = _iso(None)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for item in prepared:
                    if item.unchanged:
                        connection.execute(
                            """
                            UPDATE documents
                            SET title = ?, uri = ?, timestamp = ?, project = ?, metadata_json = ?, updated_at = ?
                            WHERE id = ?
                            """,
                            (
                                item.title,
                                item.uri,
                                item.timestamp,
                                item.project,
                                item.metadata_json,
                                now,
                                item.document_id,
                            ),
                        )
                        continue
                    connection.execute(
                        """
                        INSERT INTO documents(
                            id, source, source_key, kind, title, uri, timestamp, project,
                            metadata_json, content_hash, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(source, source_key) DO UPDATE SET
                            kind = excluded.kind,
                            title = excluded.title,
                            uri = excluded.uri,
                            timestamp = excluded.timestamp,
                            project = excluded.project,
                            metadata_json = excluded.metadata_json,
                            content_hash = excluded.content_hash,
                            updated_at = excluded.updated_at
                        """,
                        (
                            item.document_id,
                            item.source,
                            item.source_key,
                            item.kind,
                            item.title,
                            item.uri,
                            item.timestamp,
                            item.project,
                            item.metadata_json,
                            item.content_hash,
                            now,
                            now,
                        ),
                    )
                    # Preserve per-unit summaries for input-hash reuse after
                    # an appended transcript, but make them ineligible for
                    # retrieval until the current raw document is segmented
                    # and reconciled again.
                    connection.execute(
                        """
                        UPDATE distillation_state
                        SET status = 'pending', units_ready = 0,
                            last_error = 'raw_document_changed'
                        WHERE document_id = ?
                        """,
                        (item.document_id,),
                    )
                    connection.execute(
                        """
                        UPDATE distillation_unit_state
                        SET status = 'pending', last_error = 'raw_document_changed'
                        WHERE document_id = ?
                        """,
                        (item.document_id,),
                    )
                    connection.execute("DELETE FROM chunks WHERE document_id = ?", (item.document_id,))
                    for ordinal, (piece, vector) in enumerate(
                        zip(item.pieces, item.vectors, strict=True)
                    ):
                        chunk_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()
                        connection.execute(
                            """
                            INSERT INTO chunks(
                                id, document_id, ordinal, title, content, content_hash,
                                embedding, embedding_model, embedding_dimensions, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                stable_chunk_id(item.document_id, ordinal),
                                item.document_id,
                                ordinal,
                                item.title,
                                piece,
                                chunk_hash,
                                vector.tobytes(),
                                self.embedder.model_name,
                                self.embedder.dimensions,
                                now,
                            ),
                        )
                    item.chunk_count = len(item.pieces)
                if any(not item.unchanged for item in prepared):
                    self._bump_vector_generation(connection)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return [
            WriteResult(
                item.document_id,
                "unchanged" if item.unchanged else ("updated" if item.existing else "created"),
                item.chunk_count,
            )
            for item in prepared
        ]

    def upsert_document(self, document: IngestDocument, *, force: bool = False) -> WriteResult:
        return self.upsert_documents([document], force=force)[0]

    def save_memory(
        self,
        title: str,
        content: str,
        *,
        tags: Sequence[str] | None = None,
        project: str | None = None,
        confirmed_by_user: bool = False,
    ) -> dict[str, Any]:
        if not confirmed_by_user:
            raise PermissionError("kb_save_memory requires confirmed_by_user=true")
        safe_content = redact_text(content).strip()
        if not safe_content:
            raise ValueError("Memory content cannot be empty")
        digest = hashlib.sha256(safe_content.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            duplicate = connection.execute(
                "SELECT id FROM documents WHERE kind = 'memory' AND content_hash = ? LIMIT 1",
                (digest,),
            ).fetchone()
        if duplicate:
            return {
                "document_id": duplicate["id"],
                "status": "unchanged",
                "citation": f"cerebras-memory://document/{duplicate['id']}",
            }
        document = IngestDocument(
            source="memory",
            source_key=digest,
            title=title,
            text=safe_content,
            timestamp=_utc_now(),
            project=project,
            metadata={"tags": list(tags or []), "confirmed_by_user": True},
            kind="memory",
        )
        result = self.upsert_document(document)
        return {
            "document_id": result.document_id,
            "status": result.status,
            "chunks": result.chunks,
            "citation": f"cerebras-memory://document/{result.document_id}",
        }

    def forget_memory(self, memory_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT kind FROM documents WHERE id = ?", (memory_id,)
                ).fetchone()
                if row is None:
                    connection.rollback()
                    return False
                if row["kind"] != "memory":
                    raise ValueError("Administrative forget only accepts an explicitly saved memory ID")
                connection.execute("DELETE FROM documents WHERE id = ?", (memory_id,))
                self._bump_vector_generation(connection)
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise

    def reconcile_source(self, source: str, seen_keys: Iterable[str]) -> int:
        """Remove stale derived documents only after a successful source scan."""

        safe_source = redact_text(source.strip().casefold())
        safe_keys = [(redact_text(key),) for key in set(seen_keys)]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("CREATE TEMP TABLE IF NOT EXISTS scan_seen(source_key TEXT PRIMARY KEY)")
                connection.execute("DELETE FROM scan_seen")
                connection.executemany("INSERT OR IGNORE INTO scan_seen(source_key) VALUES (?)", safe_keys)
                cursor = connection.execute(
                    """
                    DELETE FROM documents
                    WHERE source = ? AND kind = 'derived'
                      AND NOT EXISTS (
                          SELECT 1 FROM scan_seen WHERE scan_seen.source_key = documents.source_key
                      )
                    """,
                    (safe_source,),
                )
                deleted = int(cursor.rowcount)
                if deleted:
                    self._bump_vector_generation(connection)
                connection.commit()
                return deleted
            except Exception:
                connection.rollback()
                raise

    def record_ingest_start(self, source: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_state(source, status, last_started_at)
                VALUES (?, 'running', ?)
                ON CONFLICT(source) DO UPDATE SET status = 'running', last_started_at = excluded.last_started_at
                """,
                (source, _iso(None)),
            )

    def record_ingest_success(
        self,
        source: str,
        *,
        watermark: str | None,
        scanned: int,
        imported: int,
        skipped: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_state(
                    source, watermark, status, last_success_at, scanned, imported, skipped, failures, last_error
                ) VALUES (?, ?, 'ok', ?, ?, ?, ?, 0, NULL)
                ON CONFLICT(source) DO UPDATE SET
                    watermark = excluded.watermark,
                    status = 'ok',
                    last_success_at = excluded.last_success_at,
                    scanned = excluded.scanned,
                    imported = excluded.imported,
                    skipped = excluded.skipped,
                    failures = 0,
                    last_error = NULL
                """,
                (source, watermark, _iso(None), scanned, imported, skipped),
            )

    def record_ingest_failure(self, source: str, error: str) -> None:
        safe_error = redact_text(error).replace("\r", " ").replace("\n", " ")[:1000]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_state(source, status, last_failure_at, failures, last_error)
                VALUES (?, 'failed', ?, 1, ?)
                ON CONFLICT(source) DO UPDATE SET
                    status = 'failed',
                    last_failure_at = excluded.last_failure_at,
                    failures = ingest_state.failures + 1,
                    last_error = excluded.last_error
                """,
                (source, _iso(None), safe_error),
            )

    def _known_projects(self) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT project, COUNT(*) AS count
                FROM documents
                WHERE project IS NOT NULL AND TRIM(project) <> ''
                GROUP BY project
                ORDER BY count DESC, project
                """
            ).fetchall()
        return {str(row["project"]).casefold(): str(row["project"]) for row in rows}

    def _project_for_path(self, value: str | Path, known: dict[str, str]) -> str | None:
        try:
            candidate = Path(value).resolve()
            root = self.settings.projects_root.resolve()
            relative = candidate.relative_to(root)
        except (OSError, ValueError):
            return None
        if not relative.parts:
            return None
        return known.get(relative.parts[0].casefold())

    def resolve_project_scope(
        self,
        *,
        project: str | None,
        global_search: bool,
        roots: Sequence[str | Path] | None = None,
        cwd: str | Path | None = None,
    ) -> dict[str, Any]:
        if project and global_search:
            raise ValueError("project cannot be combined with global_search=true")
        known = self._known_projects()
        if project:
            safe_project = redact_text(project).strip()
            return {
                "project": known.get(safe_project.casefold(), safe_project),
                "origin": "explicit",
            }
        if global_search:
            return {"project": None, "origin": "global_explicit"}

        root_projects = {
            candidate
            for value in roots or ()
            if (candidate := self._project_for_path(value, known)) is not None
        }
        if len(root_projects) == 1:
            return {"project": next(iter(root_projects)), "origin": "client_root"}
        if len(root_projects) > 1:
            return {"project": None, "origin": "global_ambiguous_roots"}

        inferred = self._project_for_path(cwd or Path.cwd(), known)
        if inferred:
            return {"project": inferred, "origin": "process_cwd"}
        return {"project": None, "origin": "global"}

    def _exact_vector_candidates(
        self,
        connection: sqlite3.Connection,
        query_vector: np.ndarray,
        *,
        filters: str,
        filter_params: Sequence[Any],
        limit: int,
        sources: Sequence[str] | None = None,
        project: str | None = None,
        since: str | datetime | None = None,
    ) -> list[tuple[float, int]]:
        snapshot = self._load_exact_vector_snapshot(connection)
        if not len(snapshot.keys):
            return []
        mask = np.ones(len(snapshot.keys), dtype=bool)
        if sources:
            normalized = np.asarray(
                [redact_text(source.casefold()) for source in sources], dtype=object
            )
            mask &= np.isin(snapshot.sources, normalized)
        if project:
            mask &= snapshot.projects == redact_text(project)
        if since:
            mask &= snapshot.timestamps >= _iso(since)
        indices = np.flatnonzero(mask)
        if not len(indices):
            return []
        scores = snapshot.vectors[indices] @ query_vector
        keys = snapshot.keys[indices]
        # Full lexicographic ordering preserves the prior deterministic
        # score/chunk-key tie behavior while still using one vectorized dot.
        order = np.lexsort((keys, scores))[-limit:][::-1]
        return [(float(scores[index]), int(keys[index])) for index in order]

    def _load_exact_vector_snapshot(
        self, connection: sqlite3.Connection
    ) -> _ExactVectorSnapshot:
        generation = int(
            connection.execute(
                "SELECT data_generation FROM vector_index_state WHERE id = 1"
            ).fetchone()[0]
        )
        cached = self._exact_vector_snapshot
        if (
            cached is not None
            and cached.generation == generation
            and cached.model == self.embedder.model_name
            and cached.dimensions == self.embedder.dimensions
        ):
            return cached
        with self._exact_vector_lock:
            cached = self._exact_vector_snapshot
            if (
                cached is not None
                and cached.generation == generation
                and cached.model == self.embedder.model_name
                and cached.dimensions == self.embedder.dimensions
            ):
                return cached
            connection.execute("BEGIN")
            try:
                generation = int(
                    connection.execute(
                        "SELECT data_generation FROM vector_index_state WHERE id = 1"
                    ).fetchone()[0]
                )
                rows = connection.execute(
                    """
                    SELECT c.chunk_pk, c.embedding, d.source, d.project, d.timestamp
                    FROM chunks c JOIN documents d ON d.id = c.document_id
                    WHERE c.embedding_model = ? AND c.embedding_dimensions = ?
                    ORDER BY c.chunk_pk
                    """,
                    (self.embedder.model_name, self.embedder.dimensions),
                ).fetchall()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            expected_bytes = self.embedder.dimensions * np.dtype(np.float32).itemsize
            valid = [row for row in rows if len(row["embedding"]) == expected_bytes]
            keys = np.asarray([int(row["chunk_pk"]) for row in valid], dtype=np.uint64)
            if valid:
                vectors = np.frombuffer(
                    b"".join(row["embedding"] for row in valid),
                    dtype=np.float32,
                ).reshape(len(valid), self.embedder.dimensions).copy()
            else:
                vectors = np.empty((0, self.embedder.dimensions), dtype=np.float32)
            snapshot = _ExactVectorSnapshot(
                generation=generation,
                model=self.embedder.model_name,
                dimensions=self.embedder.dimensions,
                keys=keys,
                vectors=vectors,
                positions={int(key): index for index, key in enumerate(keys)},
                sources=np.asarray([str(row["source"]) for row in valid], dtype=object),
                projects=np.asarray(
                    [str(row["project"]) if row["project"] is not None else None for row in valid],
                    dtype=object,
                ),
                timestamps=np.asarray([str(row["timestamp"]) for row in valid], dtype=object),
            )
            self._exact_vector_snapshot = snapshot
            return snapshot

    def vector_index_status(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM vector_index_state WHERE id = 1").fetchone()
        state = dict(row) if row else {}
        database_error = state.get("last_error")
        runtime_state = self.vector_index.status()
        state.update(runtime_state)
        state["database_last_error"] = database_error
        state["last_error"] = runtime_state.get("last_error") or database_error
        generation_current = bool(
            state
            and state.get("built_generation") == state.get("data_generation")
            and state.get("embedding_model") == self.embedder.model_name
            and int(state.get("embedding_dimensions") or 0) == self.embedder.dimensions
        )
        state["generation_current"] = generation_current
        state["active_backend"] = (
            "hnsw"
            if state.get("backend") == "hnsw"
            and state.get("status") == "ready"
            and generation_current
            and bool(state.get("exists"))
            and self.settings.vector_search.backend != "exact"
            else "exact"
        )
        state["ann_status"] = state.get("status", "missing")
        state["configured_backend"] = self.settings.vector_search.backend
        state["ann_min_chunks"] = self.settings.vector_search.ann_min_chunks
        state["ann_latency_threshold_ms"] = self.settings.vector_search.ann_latency_threshold_ms
        return state

    def benchmark_vector_search(self, *, runs: int = 3) -> dict[str, Any]:
        # This vector is fixed and content-free: benchmarking never embeds or
        # persists query text.  Normalization keeps cosine and dot-product
        # timings representative without carrying semantic content.
        query_vector = np.arange(1, self.embedder.dimensions + 1, dtype=np.float32)
        query_vector /= float(np.linalg.norm(query_vector))
        timings: list[float] = []
        count = 0
        for _ in range(max(1, runs)):
            started = time.perf_counter()
            with self._connect() as connection:
                self._exact_vector_candidates(
                    connection,
                    query_vector,
                    filters="",
                    filter_params=[],
                    limit=self.settings.candidate_limit,
                )
                count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM chunks
                        WHERE embedding_model = ? AND embedding_dimensions = ?
                        """,
                        (self.embedder.model_name, self.embedder.dimensions),
                    ).fetchone()[0]
                )
            timings.append((time.perf_counter() - started) * 1000.0)
        median_ms = float(statistics.median(timings))
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE vector_index_state
                SET exact_benchmark_ms = ?, chunk_count = ?, updated_at = ?
                WHERE id = 1
                """,
                (median_ms, count, _iso(None)),
            )
        return {"runs_ms": [round(item, 3) for item in timings], "median_ms": median_ms, "chunks": count}

    def rebuild_vector_index(self, *, force: bool = False) -> dict[str, Any]:
        with self._connect() as connection:
            state = connection.execute("SELECT * FROM vector_index_state WHERE id = 1").fetchone()
            generation = int(state["data_generation"])
            benchmark = float(state["exact_benchmark_ms"] or 0.0)
            rows = connection.execute(
                """
                SELECT chunk_pk, embedding FROM chunks
                WHERE embedding_model = ? AND embedding_dimensions = ?
                ORDER BY chunk_pk
                """,
                (self.embedder.model_name, self.embedder.dimensions),
            ).fetchall()
        count = len(rows)
        configured = self.settings.vector_search.backend
        eligible = (
            count >= self.settings.vector_search.ann_min_chunks
            or benchmark >= self.settings.vector_search.ann_latency_threshold_ms
        )
        if configured == "exact":
            eligible = False
        if not force and not eligible:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE vector_index_state
                    SET backend = 'exact', status = 'exact', chunk_count = ?, last_error = NULL,
                        updated_at = ? WHERE id = 1
                    """,
                    (count, _iso(None)),
                )
            return {"status": "exact", "built": False, "chunks": count, "eligible": False}

        keys = np.asarray([int(row["chunk_pk"]) for row in rows], dtype=np.uint64)
        vectors = np.asarray(
            [np.frombuffer(row["embedding"], dtype=np.float32).copy() for row in rows],
            dtype=np.float32,
        )
        try:
            result = self.vector_index.build(keys, vectors)
            with self._connect() as connection:
                current = int(
                    connection.execute(
                        "SELECT data_generation FROM vector_index_state WHERE id = 1"
                    ).fetchone()[0]
                )
                ready = current == generation
                connection.execute(
                    """
                    UPDATE vector_index_state
                    SET built_generation = ?, embedding_model = ?, embedding_dimensions = ?,
                        chunk_count = ?, backend = 'hnsw', status = ?, index_path = ?,
                        built_at = ?, last_error = NULL, updated_at = ?
                    WHERE id = 1
                    """,
                    (
                        generation,
                        self.embedder.model_name,
                        self.embedder.dimensions,
                        count,
                        "ready" if ready else "stale",
                        str(self.vector_index.path),
                        _iso(None),
                        _iso(None),
                    ),
                )
            return {"status": "ready" if ready else "stale", "built": True, **result}
        except Exception as exc:
            error = redact_text(f"{type(exc).__name__}: {exc}")[:500]
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE vector_index_state
                    SET backend = 'exact', status = 'failed', last_error = ?, updated_at = ?
                    WHERE id = 1
                    """,
                    (error, _iso(None)),
                )
            return {"status": "failed", "built": False, "error": error, "chunks": count}

    def maintain_vector_index(self) -> dict[str, Any]:
        benchmark = self.benchmark_vector_search(runs=3)
        result = self.rebuild_vector_index()
        return {"benchmark": benchmark, "index": result}

    def warm_reranker(self) -> dict[str, Any]:
        warm = getattr(self.reranker, "warm", None)
        if warm is None:
            raise RerankerUnavailable("reranker_disabled")
        return warm()

    @staticmethod
    def _stable_distillation_id(
        document_id: str,
        input_hash: str,
        model: str,
        prompt_version: str,
    ) -> str:
        digest = hashlib.sha256(
            f"{document_id}\0{input_hash}\0{model}\0{prompt_version}".encode("utf-8")
        ).hexdigest()
        return f"dst_{digest[:32]}"

    def _prepare_distillation_run(
        self,
        document_id: str,
        units: Sequence[Any],
        existing: dict[str, sqlite3.Row],
        *,
        force: bool,
    ) -> None:
        """Mark a document pending before local inference starts.

        A process interruption must not expose a mixture of old and new
        summary units. Individual completed units are checkpointed below, but
        retrieval continues to require the document-level state to be ready.
        """

        now = _iso(None)
        current_hashes = [unit.input_hash for unit in units]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                placeholders = ",".join("?" for _ in current_hashes)
                connection.execute(
                    f"""
                    DELETE FROM distillation_unit_state
                    WHERE document_id = ? AND (
                        distiller_model <> ? OR prompt_version <> ?
                        OR input_hash NOT IN ({placeholders})
                    )
                    """,
                    [
                        document_id,
                        self.settings.distillation.model,
                        self.settings.distillation.prompt_version,
                        *current_hashes,
                    ],
                )
                initial_ready = 0
                for unit in units:
                    cached = existing.get(unit.input_hash) if not force else None
                    status = "ready" if cached is not None else "pending"
                    initial_ready += int(status == "ready")
                    connection.execute(
                        """
                        INSERT INTO distillation_unit_state(
                            document_id, input_hash, unit_ordinal, start_ordinal, end_ordinal,
                            distiller_model, prompt_version, status, attempts,
                            last_attempt_at, last_success_at, last_error
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, NULL)
                        ON CONFLICT(document_id, input_hash, distiller_model, prompt_version)
                        DO UPDATE SET
                            unit_ordinal = excluded.unit_ordinal,
                            start_ordinal = excluded.start_ordinal,
                            end_ordinal = excluded.end_ordinal,
                            status = excluded.status,
                            last_success_at = CASE
                                WHEN excluded.status = 'ready'
                                THEN COALESCE(distillation_unit_state.last_success_at, excluded.last_success_at)
                                ELSE distillation_unit_state.last_success_at
                            END,
                            last_error = NULL
                        """,
                        (
                            document_id,
                            unit.input_hash,
                            unit.unit_ordinal,
                            unit.start_ordinal,
                            unit.end_ordinal,
                            self.settings.distillation.model,
                            self.settings.distillation.prompt_version,
                            status,
                            now if status == "ready" else None,
                        ),
                    )
                connection.execute(
                    """
                    INSERT INTO distillation_state(
                        document_id, status, model, prompt_version, units_total, units_ready,
                        failures, last_attempt_at, last_success_at, last_error
                    ) VALUES (?, 'pending', ?, ?, ?, ?, 0, ?, NULL, NULL)
                    ON CONFLICT(document_id) DO UPDATE SET
                        status = 'pending', model = excluded.model,
                        prompt_version = excluded.prompt_version,
                        units_total = excluded.units_total,
                        units_ready = excluded.units_ready,
                        failures = 0, last_attempt_at = excluded.last_attempt_at,
                        last_error = NULL
                    """,
                    (
                        document_id,
                        self.settings.distillation.model,
                        self.settings.distillation.prompt_version,
                        len(units),
                        initial_ready,
                        now,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _checkpoint_distillation_unit(
        self,
        document_id: str,
        unit: Any,
        structured: dict[str, Any],
        search_text: str,
        vector: np.ndarray,
        distillation_id: str,
    ) -> None:
        """Commit one completed unit so a later run can reuse it."""

        now = _iso(None)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO distillations(
                        id, document_id, unit_ordinal, input_hash, start_ordinal, end_ordinal,
                        summary_json, search_text, embedding, embedding_model,
                        embedding_dimensions, distiller_model, prompt_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        unit_ordinal = excluded.unit_ordinal,
                        start_ordinal = excluded.start_ordinal,
                        end_ordinal = excluded.end_ordinal,
                        summary_json = excluded.summary_json,
                        search_text = excluded.search_text,
                        embedding = excluded.embedding,
                        embedding_model = excluded.embedding_model,
                        embedding_dimensions = excluded.embedding_dimensions,
                        updated_at = excluded.updated_at
                    """,
                    (
                        distillation_id,
                        document_id,
                        unit.unit_ordinal,
                        unit.input_hash,
                        unit.start_ordinal,
                        unit.end_ordinal,
                        json.dumps(structured, ensure_ascii=False, sort_keys=True),
                        search_text,
                        np.asarray(vector, dtype=np.float32).tobytes(),
                        self.embedder.model_name,
                        self.embedder.dimensions,
                        self.settings.distillation.model,
                        self.settings.distillation.prompt_version,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    UPDATE distillation_unit_state
                    SET status = 'ready', attempts = attempts + 1,
                        last_attempt_at = ?, last_success_at = ?, last_error = NULL
                    WHERE document_id = ? AND input_hash = ?
                      AND distiller_model = ? AND prompt_version = ?
                    """,
                    (
                        now,
                        now,
                        document_id,
                        unit.input_hash,
                        self.settings.distillation.model,
                        self.settings.distillation.prompt_version,
                    ),
                )
                ready = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM distillation_unit_state
                        WHERE document_id = ? AND distiller_model = ?
                          AND prompt_version = ? AND status = 'ready'
                        """,
                        (
                            document_id,
                            self.settings.distillation.model,
                            self.settings.distillation.prompt_version,
                        ),
                    ).fetchone()[0]
                )
                connection.execute(
                    """
                    UPDATE distillation_state
                    SET units_ready = ?, last_attempt_at = ?, last_success_at = ?
                    WHERE document_id = ?
                    """,
                    (ready, now, now, document_id),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _qualifying_distillation_documents(
        self,
        *,
        source: str | None = None,
        pilot: bool = False,
        limit: int | None = None,
    ) -> list[str]:
        params: list[Any] = [*sorted(AGENT_SOURCES)]
        placeholders = ",".join("?" for _ in AGENT_SOURCES)
        source_clause = ""
        if source:
            source_clause = " AND d.source = ?"
            params.append(redact_text(source.casefold()))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT d.id, d.source, d.metadata_json, d.content_hash,
                       SUM(LENGTH(c.content)) AS characters
                FROM documents d JOIN chunks c ON c.document_id = d.id
                WHERE d.kind = 'derived' AND d.source IN ({placeholders}) {source_clause}
                GROUP BY d.id
                ORDER BY d.source, d.id
                """,
                params,
            ).fetchall()
        qualifying: list[sqlite3.Row] = []
        with self._connect() as connection:
            for row in rows:
                metadata = json.loads(row["metadata_json"] or "{}")
                messages = int(metadata.get("message_count") or 0)
                if messages >= self.settings.distillation.min_messages:
                    qualifying.append(row)
                    continue
                # Chunk overlap intentionally repeats content. Qualification
                # is based on redacted document characters, not the inflated
                # SUM(LENGTH(chunk)) value used only as a cheap prefilter.
                if int(row["characters"] or 0) < self.settings.distillation.min_characters:
                    continue
                document_chunks = connection.execute(
                    "SELECT ordinal, content FROM chunks WHERE document_id = ? ORDER BY ordinal",
                    (row["id"],),
                ).fetchall()
                body, _ = reconstruct_chunks(
                    [
                        (int(chunk["ordinal"]), str(chunk["content"]))
                        for chunk in document_chunks
                    ]
                )
                if len(body) >= self.settings.distillation.min_characters:
                    qualifying.append(row)
        if pilot:
            selected: list[sqlite3.Row] = []
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    for name in sorted(AGENT_SOURCES):
                        candidates = [row for row in qualifying if row["source"] == name]
                        if not candidates:
                            continue
                        by_id = {str(row["id"]): row for row in candidates}
                        existing_ids = [
                            str(row["document_id"])
                            for row in connection.execute(
                                """
                                SELECT document_id FROM distillation_pilot_documents
                                WHERE source = ? ORDER BY selected_at, document_id
                                """,
                                (name,),
                            )
                            if str(row["document_id"]) in by_id
                        ]
                        chosen = [by_id[document_id] for document_id in existing_ids]
                        remaining = [
                            row for row in candidates if str(row["id"]) not in existing_ids
                        ]
                        created = {
                            str(row["document_id"]): str(row["first_created"])
                            for row in connection.execute(
                                """
                                SELECT x.document_id, MIN(x.created_at) AS first_created
                                FROM distillations x JOIN documents d ON d.id = x.document_id
                                WHERE d.source = ? AND x.distiller_model = ?
                                  AND x.prompt_version = ?
                                GROUP BY x.document_id
                                """,
                                (
                                    name,
                                    self.settings.distillation.model,
                                    self.settings.distillation.prompt_version,
                                ),
                            )
                        }
                        attempts = {
                            str(row["document_id"]): str(row["last_attempt_at"] or "")
                            for row in connection.execute(
                                """
                                SELECT document_id, last_attempt_at FROM distillation_state
                                WHERE model = ? AND prompt_version = ?
                                """,
                                (
                                    self.settings.distillation.model,
                                    self.settings.distillation.prompt_version,
                                ),
                            )
                        }

                        def selection_key(row: sqlite3.Row) -> tuple[int, str, str]:
                            document_id = str(row["id"])
                            if document_id in created:
                                return (0, created[document_id], document_id)
                            if document_id in attempts:
                                return (1, attempts[document_id], document_id)
                            stable = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
                            return (2, stable, document_id)

                        remaining.sort(key=selection_key)
                        needed = max(
                            0, self.settings.distillation.pilot_per_source - len(chosen)
                        )
                        chosen.extend(remaining[:needed])
                        for row in chosen:
                            connection.execute(
                                """
                                INSERT OR IGNORE INTO distillation_pilot_documents(
                                    document_id, source, selected_at
                                ) VALUES (?, ?, ?)
                                """,
                                (row["id"], name, _iso(None)),
                            )
                        selected.extend(chosen)
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
            qualifying = selected
        identifiers = [str(row["id"]) for row in qualifying]
        return identifiers[:limit] if limit is not None else identifiers

    def distill_document(
        self,
        document_id: str,
        *,
        force: bool = False,
        force_input_hashes: set[str] | None = None,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            document = connection.execute(
                "SELECT source, metadata_json FROM documents WHERE id = ? AND kind = 'derived'",
                (document_id,),
            ).fetchone()
            chunks = connection.execute(
                "SELECT ordinal, content FROM chunks WHERE document_id = ? ORDER BY ordinal",
                (document_id,),
            ).fetchall()
            existing_rows = connection.execute(
                """
                SELECT * FROM distillations
                WHERE document_id = ? AND distiller_model = ? AND prompt_version = ?
                """,
                (
                    document_id,
                    self.settings.distillation.model,
                    self.settings.distillation.prompt_version,
                ),
            ).fetchall()
        if document is None or document["source"] not in AGENT_SOURCES:
            return {"document_id": document_id, "status": "ineligible", "units": 0}
        metadata = json.loads(document["metadata_json"] or "{}")
        body, spans = reconstruct_chunks(
            [(int(row["ordinal"]), str(row["content"])) for row in chunks]
        )
        if (
            int(metadata.get("message_count") or 0) < self.settings.distillation.min_messages
            and len(body) < self.settings.distillation.min_characters
        ):
            return {"document_id": document_id, "status": "ineligible", "units": 0}
        units = segment_dialogue(body, spans, self.settings.distillation)
        if not units:
            return {"document_id": document_id, "status": "no_dialogue", "units": 0}

        existing = {
            row["input_hash"]: row
            for row in existing_rows
            if row["embedding_model"] == self.embedder.model_name
            and int(row["embedding_dimensions"]) == self.embedder.dimensions
        }
        self._prepare_distillation_run(document_id, units, existing, force=force)
        ready_ids: set[str] = set()
        generated: list[tuple[Any, dict[str, Any], str, np.ndarray, str]] = []
        failures: list[str] = []
        unit_outcomes: list[tuple[Any, str, str | None, int]] = []
        pending_units: list[Any] = []
        for unit in units:
            cached = existing.get(unit.input_hash)
            selectively_forced = bool(
                force_input_hashes and unit.input_hash in force_input_hashes
            )
            if cached is not None and not force and not selectively_forced:
                ready_ids.add(str(cached["id"]))
                unit_outcomes.append((unit, "ready", None, 0))
                continue
            pending_units.append(unit)

        def checkpoint_generated(unit: Any, result: dict[str, Any]) -> None:
            distillation_id = self._stable_distillation_id(
                document_id,
                unit.input_hash,
                self.settings.distillation.model,
                self.settings.distillation.prompt_version,
            )
            # Revalidate and redact here even when an injected/local
            # distiller claims to return a validated object. The storage
            # boundary is the final authority. Embedding and SQLite writes
            # deliberately remain on this coordinator thread.
            structured = validate_distillation(result)
            raw_evidence = "\n".join(
                str(row["content"])
                for row in chunks
                if unit.start_ordinal <= int(row["ordinal"]) <= unit.end_ordinal
            )
            structured = sanitize_role_contradictions(structured, raw_evidence)
            search_text = flatten_distillation(structured)
            if not search_text:
                raise DistillationInvalid("distillation produced no searchable content")
            with self._distillation_checkpoint_lock:
                vector = self.embedder.embed([search_text])[0]
                self._checkpoint_distillation_unit(
                    document_id,
                    unit,
                    structured,
                    search_text,
                    vector,
                    distillation_id,
                )
            generated.append((unit, structured, search_text, vector, distillation_id))
            ready_ids.add(distillation_id)
            # The checkpoint already recorded the inference attempt.
            unit_outcomes.append((unit, "ready", None, 0))

        def record_failure(unit: Any, exc: Exception) -> None:
            error = redact_text(f"{type(exc).__name__}: {exc}")[:500]
            failures.append(error)
            unit_outcomes.append((unit, "failed", error, 1))

        def record_unavailable(unit: Any, exc: DistillationUnavailable) -> None:
            error = redact_text(str(exc))[:500]
            failures.append(error)
            unit_outcomes.append((unit, "failed", error, 1))

        def call_distiller(text: str) -> dict[str, Any]:
            with self._distillation_request_slots:
                return self.distiller.distill(text)

        concurrency = min(
            self.settings.distillation.max_concurrent_requests,
            max(1, len(pending_units)),
        )
        if concurrency == 1:
            unavailable = False
            for unit in pending_units:
                if unavailable:
                    error = "distiller_unavailable_after_prior_failure"
                    failures.append(error)
                    unit_outcomes.append((unit, "pending", error, 0))
                    continue
                try:
                    checkpoint_generated(unit, call_distiller(unit.text))
                except DistillationUnavailable as exc:
                    unavailable = True
                    record_unavailable(unit, exc)
                except (DistillationInvalid, ValueError, RuntimeError) as exc:
                    record_failure(unit, exc)
        elif pending_units:
            with ThreadPoolExecutor(
                max_workers=concurrency,
                thread_name_prefix="kb-distill",
            ) as executor:
                futures = {
                    executor.submit(call_distiller, unit.text): unit
                    for unit in pending_units
                }
                provider_unavailable = False
                for future in as_completed(futures):
                    unit = futures[future]
                    if future.cancelled():
                        error = "distiller_unavailable_after_prior_failure"
                        failures.append(error)
                        unit_outcomes.append((unit, "pending", error, 0))
                        continue
                    try:
                        checkpoint_generated(unit, future.result())
                    except DistillationUnavailable as exc:
                        provider_unavailable = True
                        record_unavailable(unit, exc)
                        for other in futures:
                            if other is not future and not other.done():
                                other.cancel()
                    except (DistillationInvalid, ValueError, RuntimeError) as exc:
                        record_failure(unit, exc)

                # A canceled queued request is retryable and was never billed
                # as an inference attempt. Running requests are allowed to
                # finish so their successful checkpoints are not discarded.
                if provider_unavailable:
                    for future, unit in futures.items():
                        if future.cancelled() and not any(
                            outcome[0] is unit for outcome in unit_outcomes
                        ):
                            error = "distiller_unavailable_after_prior_failure"
                            failures.append(error)
                            unit_outcomes.append((unit, "pending", error, 0))

        now = _iso(None)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for unit, structured, search_text, vector, distillation_id in generated:
                    connection.execute(
                        """
                        INSERT INTO distillations(
                            id, document_id, unit_ordinal, input_hash, start_ordinal, end_ordinal,
                            summary_json, search_text, embedding, embedding_model,
                            embedding_dimensions, distiller_model, prompt_version, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            unit_ordinal = excluded.unit_ordinal,
                            start_ordinal = excluded.start_ordinal,
                            end_ordinal = excluded.end_ordinal,
                            summary_json = excluded.summary_json,
                            search_text = excluded.search_text,
                            embedding = excluded.embedding,
                            embedding_model = excluded.embedding_model,
                            embedding_dimensions = excluded.embedding_dimensions,
                            updated_at = excluded.updated_at
                        """,
                        (
                            distillation_id,
                            document_id,
                            unit.unit_ordinal,
                            unit.input_hash,
                            unit.start_ordinal,
                            unit.end_ordinal,
                            json.dumps(structured, ensure_ascii=False, sort_keys=True),
                            search_text,
                            np.asarray(vector, dtype=np.float32).tobytes(),
                            self.embedder.model_name,
                            self.embedder.dimensions,
                            self.settings.distillation.model,
                            self.settings.distillation.prompt_version,
                            now,
                            now,
                        ),
                    )
                if ready_ids:
                    placeholders = ",".join("?" for _ in ready_ids)
                    connection.execute(
                        f"DELETE FROM distillations WHERE document_id = ? AND id NOT IN ({placeholders})",
                        [document_id, *sorted(ready_ids)],
                    )
                else:
                    connection.execute(
                        "DELETE FROM distillations WHERE document_id = ?", (document_id,)
                    )
                input_hashes = [unit.input_hash for unit in units]
                placeholders = ",".join("?" for _ in input_hashes)
                connection.execute(
                    f"""
                    DELETE FROM distillation_unit_state
                    WHERE document_id = ? AND (
                        distiller_model <> ? OR prompt_version <> ?
                        OR input_hash NOT IN ({placeholders})
                    )
                    """,
                    [
                        document_id,
                        self.settings.distillation.model,
                        self.settings.distillation.prompt_version,
                        *input_hashes,
                    ],
                )
                for unit, unit_status, unit_error, attempts in unit_outcomes:
                    connection.execute(
                        """
                        INSERT INTO distillation_unit_state(
                            document_id, input_hash, unit_ordinal, start_ordinal, end_ordinal,
                            distiller_model, prompt_version, status, attempts,
                            last_attempt_at, last_success_at, last_error
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(document_id, input_hash, distiller_model, prompt_version)
                        DO UPDATE SET
                            unit_ordinal = excluded.unit_ordinal,
                            start_ordinal = excluded.start_ordinal,
                            end_ordinal = excluded.end_ordinal,
                            status = excluded.status,
                            attempts = distillation_unit_state.attempts + excluded.attempts,
                            last_attempt_at = CASE
                                WHEN excluded.attempts > 0 THEN excluded.last_attempt_at
                                ELSE distillation_unit_state.last_attempt_at
                            END,
                            last_success_at = CASE
                                WHEN excluded.status = 'ready' THEN excluded.last_success_at
                                ELSE distillation_unit_state.last_success_at
                            END,
                            last_error = excluded.last_error
                        """,
                        (
                            document_id,
                            unit.input_hash,
                            unit.unit_ordinal,
                            unit.start_ordinal,
                            unit.end_ordinal,
                            self.settings.distillation.model,
                            self.settings.distillation.prompt_version,
                            unit_status,
                            attempts,
                            now if attempts else None,
                            now if unit_status == "ready" else None,
                            unit_error,
                        ),
                    )
                ready = len(ready_ids)
                total = len(units)
                status = "ready" if ready == total and not failures else ("partial" if ready else "failed")
                connection.execute(
                    """
                    INSERT INTO distillation_state(
                        document_id, status, model, prompt_version, units_total, units_ready,
                        failures, last_attempt_at, last_success_at, last_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        status = excluded.status,
                        model = excluded.model,
                        prompt_version = excluded.prompt_version,
                        units_total = excluded.units_total,
                        units_ready = excluded.units_ready,
                        failures = excluded.failures,
                        last_attempt_at = excluded.last_attempt_at,
                        last_success_at = excluded.last_success_at,
                        last_error = excluded.last_error
                    """,
                    (
                        document_id,
                        status,
                        self.settings.distillation.model,
                        self.settings.distillation.prompt_version,
                        total,
                        ready,
                        len(failures),
                        now,
                        now if status == "ready" else None,
                        failures[0] if failures else None,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {
            "document_id": document_id,
            "status": status,
            "units": len(units),
            "ready": len(ready_ids),
            "generated": len(generated),
            "failures": len(failures),
        }

    def distill_documents(
        self,
        *,
        pilot: bool = False,
        source: str | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        lock_path = self.database_path.parent / "distill.lock"
        with distillation_lock(lock_path):
            identifiers = self._qualifying_distillation_documents(
                source=source,
                pilot=pilot,
                limit=limit,
            )
            workers = min(
                self.settings.distillation.max_concurrent_requests,
                max(1, len(identifiers)),
            )
            if workers == 1:
                reports = [
                    self.distill_document(document_id, force=force)
                    for document_id in identifiers
                ]
            else:
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="kb-distill-document",
                ) as executor:
                    futures = {
                        executor.submit(
                            self.distill_document,
                            document_id,
                            force=force,
                        ): index
                        for index, document_id in enumerate(identifiers)
                    }
                    indexed_reports = {
                        futures[future]: future.result()
                        for future in as_completed(futures)
                    }
                reports = [indexed_reports[index] for index in range(len(identifiers))]
        return {
            "mode": "pilot" if pilot else "backfill",
            "documents": len(reports),
            "ready": sum(report["status"] == "ready" for report in reports),
            "partial": sum(report["status"] == "partial" for report in reports),
            "failed": sum(report["status"] == "failed" for report in reports),
            "units": sum(int(report.get("units", 0)) for report in reports),
            "generated": sum(int(report.get("generated", 0)) for report in reports),
            "reports": reports,
        }

    def distillation_status(self) -> dict[str, Any]:
        with self._connect() as connection:
            units = int(connection.execute("SELECT COUNT(*) FROM distillations").fetchone()[0])
            documents = int(
                connection.execute("SELECT COUNT(DISTINCT document_id) FROM distillations").fetchone()[0]
            )
            states = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM distillation_state GROUP BY status"
                )
            }
            failure = connection.execute(
                """
                SELECT last_error FROM distillation_state
                WHERE last_error IS NOT NULL ORDER BY last_attempt_at DESC LIMIT 1
                """
            ).fetchone()
            totals = connection.execute(
                """
                SELECT COALESCE(SUM(units_total), 0) AS units_total,
                       COALESCE(SUM(units_ready), 0) AS units_ready,
                       COALESCE(SUM(failures), 0) AS failures
                FROM distillation_state
                """
            ).fetchone()
            unit_states = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM distillation_unit_state GROUP BY status
                    """
                )
            }
            tracked_units = sum(unit_states.values())
            pilot_documents = int(
                connection.execute(
                    "SELECT COUNT(*) FROM distillation_pilot_documents"
                ).fetchone()[0]
            )
            pilot_units = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM distillations x
                    JOIN distillation_pilot_documents p ON p.document_id = x.document_id
                    WHERE x.distiller_model = ? AND x.prompt_version = ?
                    """,
                    (
                        self.settings.distillation.model,
                        self.settings.distillation.prompt_version,
                    ),
                ).fetchone()[0]
            )
            pilot_states = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT s.status, COUNT(*) AS count
                    FROM distillation_state s
                    JOIN distillation_pilot_documents p ON p.document_id = s.document_id
                    WHERE s.model = ? AND s.prompt_version = ?
                    GROUP BY s.status
                    """,
                    (
                        self.settings.distillation.model,
                        self.settings.distillation.prompt_version,
                    ),
                )
            }
        return {
            "mode": self.settings.distillation.mode,
            "search_enabled": self.settings.distillation.mode == "on",
            "provider": self.settings.distillation.provider,
            "model": self.settings.distillation.model,
            "prompt_version": self.settings.distillation.prompt_version,
            "endpoint": self.settings.distillation.endpoint,
            "credential_configured": (
                True
                if not self.settings.distillation.api_key_env
                else bool(os.environ.get(self.settings.distillation.api_key_env))
            ),
            "units": units,
            "documents": documents,
            "units_total": max(int(totals["units_total"]), tracked_units),
            "units_ready": unit_states.get("ready", int(totals["units_ready"])),
            "units_pending": max(
                unit_states.get("pending", 0) + unit_states.get("failed", 0),
                int(totals["units_total"]) - int(totals["units_ready"]),
            ),
            "unit_failures": unit_states.get("failed", int(totals["failures"])),
            "unit_states": unit_states,
            "pilot_documents": pilot_documents,
            "pilot_units": pilot_units,
            "pilot_states": pilot_states,
            "states": states,
            "last_error": failure["last_error"] if failure else None,
        }

    def evaluate_distillations(self, *, limit: int = 24) -> dict[str, Any]:
        limit = min(max(1, int(limit)), 100)
        pilot_join = (
            "JOIN distillation_pilot_documents p ON p.document_id = x.document_id"
            if self.settings.distillation.mode == "pilot"
            else ""
        )
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT x.document_id, x.summary_json
                FROM distillations x
                {pilot_join}
                JOIN distillation_state s
                  ON s.document_id = x.document_id AND s.status = 'ready'
                JOIN (
                    SELECT document_id, MIN(unit_ordinal) AS unit_ordinal
                    FROM distillations GROUP BY document_id
                ) first_unit
                  ON first_unit.document_id = x.document_id
                 AND first_unit.unit_ordinal = x.unit_ordinal
                ORDER BY x.document_id LIMIT ?
                """,
                (limit,),
            ).fetchall()
        cases: list[tuple[str, str]] = []
        schema_valid = True
        secret_free = True
        for row in rows:
            try:
                structured = json.loads(row["summary_json"])
                goal = str(structured.get("user_goal") or structured.get("summary") or "").strip()
                schema_valid = schema_valid and set(structured) == {
                    "user_goal", "summary", "outcome", "decisions", "artifacts",
                    "systems", "open_questions", "keywords",
                }
                secret_free = secret_free and redact_text(json.dumps(structured)) == json.dumps(structured)
                if goal:
                    cases.append((str(row["document_id"]), goal))
            except (TypeError, ValueError, json.JSONDecodeError):
                schema_valid = False
        baseline_ranks: list[int | None] = []
        augmented_ranks: list[int | None] = []
        case_reports: list[dict[str, Any]] = []
        for document_id, goal in cases:
            baseline = self.search_response(
                goal,
                limit=8,
                global_search=True,
                rerank=False,
                include_distillations=False,
            )["results"]
            augmented = self.search_response(
                goal,
                limit=8,
                global_search=True,
                rerank=False,
                include_distillations=True,
            )["results"]
            baseline_rank = next(
                (index for index, item in enumerate(baseline, start=1) if item["document_id"] == document_id),
                None,
            )
            augmented_rank = next(
                (index for index, item in enumerate(augmented, start=1) if item["document_id"] == document_id),
                None,
            )
            baseline_ranks.append(baseline_rank)
            augmented_ranks.append(augmented_rank)
            case_reports.append(
                {
                    "document_id": document_id,
                    "baseline_rank": baseline_rank,
                    "augmented_rank": augmented_rank,
                }
            )

        def metrics(ranks: Sequence[int | None]) -> dict[str, float]:
            count = len(ranks)
            recall = sum(rank is not None and rank <= 8 for rank in ranks) / count if count else 0.0
            mrr = sum((1.0 / rank) if rank else 0.0 for rank in ranks) / count if count else 0.0
            return {"recall_at_8": round(recall, 6), "mrr_at_8": round(mrr, 6)}

        baseline_metrics = metrics(baseline_ranks)
        augmented_metrics = metrics(augmented_ranks)
        baseline_mrr = baseline_metrics["mrr_at_8"]
        improvement = (
            (augmented_metrics["mrr_at_8"] - baseline_mrr) / baseline_mrr
            if baseline_mrr > 0
            else (1.0 if augmented_metrics["mrr_at_8"] > 0 else 0.0)
        )
        automated_gate = bool(
            cases
            and schema_valid
            and secret_free
            and augmented_metrics["recall_at_8"] >= baseline_metrics["recall_at_8"]
            and improvement >= 0.05
        )
        return {
            "cases": len(cases),
            "schema_valid": schema_valid,
            "secret_free": secret_free,
            "baseline": baseline_metrics,
            "augmented": augmented_metrics,
            "mrr_relative_improvement": round(improvement, 6),
            "automated_gate_passed": automated_gate,
            "manual_traceability_audit_required": True,
            "promotion_ready": False,
            "case_results": case_reports,
        }

    @staticmethod
    def _fts_query(query: str) -> str | None:
        tokens = re.findall(r"[\w'-]+", query.casefold(), flags=re.UNICODE)
        if not tokens:
            return None
        unique = list(dict.fromkeys(tokens))[:24]
        return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique)

    @staticmethod
    def _filter_sql(
        *, sources: Sequence[str] | None, project: str | None, since: str | datetime | None
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if sources:
            normalized = [redact_text(source.casefold()) for source in sources]
            placeholders = ",".join("?" for _ in normalized)
            clauses.append(f"d.source IN ({placeholders})")
            params.extend(normalized)
        if project:
            clauses.append("d.project = ?")
            params.append(redact_text(project))
        if since:
            clauses.append("d.timestamp >= ?")
            params.append(_iso(since))
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    @staticmethod
    def _snippet(content: str, query: str, length: int = 700) -> str:
        if len(content) <= length:
            return content
        terms = re.findall(r"[\w'-]+", query.casefold())
        lowered = content.casefold()
        positions = [lowered.find(term) for term in terms]
        position = min((item for item in positions if item >= 0), default=0)
        start = max(0, position - length // 3)
        end = min(len(content), start + length)
        prefix = "…" if start else ""
        suffix = "…" if end < len(content) else ""
        return f"{prefix}{content[start:end].strip()}{suffix}"

    def _vector_candidates(
        self,
        connection: sqlite3.Connection,
        query_vector: np.ndarray,
        *,
        filters: str,
        filter_params: Sequence[Any],
        limit: int,
        sources: Sequence[str] | None = None,
        project: str | None = None,
        since: str | datetime | None = None,
    ) -> tuple[list[tuple[float, int]], dict[str, Any]]:
        state = connection.execute("SELECT * FROM vector_index_state WHERE id = 1").fetchone()
        unfiltered = not filters
        ready = bool(
            state
            and state["backend"] == "hnsw"
            and state["status"] == "ready"
            and state["built_generation"] == state["data_generation"]
            and state["embedding_model"] == self.embedder.model_name
            and int(state["embedding_dimensions"] or 0) == self.embedder.dimensions
            and self.settings.vector_search.backend != "exact"
            and unfiltered
        )
        if ready:
            try:
                return self.vector_index.search(query_vector, limit), {
                    "backend": "hnsw",
                    "status": "ready",
                    "degraded_reason": None,
                }
            except VectorIndexUnavailable as exc:
                degraded = redact_text(str(exc))[:500]
        else:
            degraded = None
        return self._exact_vector_candidates(
            connection,
            query_vector,
            filters=filters,
            filter_params=filter_params,
            limit=limit,
            sources=sources,
            project=project,
            since=since,
        ), {
            "backend": "exact",
            "status": "fallback" if degraded else "ready",
            "degraded_reason": degraded,
        }

    def _distillation_candidates(
        self,
        connection: sqlite3.Connection,
        safe_query: str,
        query_vector: np.ndarray,
        *,
        filters: str,
        filter_params: Sequence[Any],
        limit: int,
    ) -> tuple[list[int], dict[int, str]]:
        fts_query = self._fts_query(safe_query)
        lexical_rank: dict[int, int] = {}
        vector_rank: dict[int, int] = {}
        rows_by_pk: dict[int, sqlite3.Row] = {}
        pilot_join = (
            "JOIN distillation_pilot_documents p ON p.document_id = x.document_id"
            if self.settings.distillation.mode == "pilot"
            else ""
        )
        if fts_query:
            rows = connection.execute(
                f"""
                SELECT x.*, bm25(distillations_fts) AS lexical_score
                FROM distillations_fts
                JOIN distillations x ON x.distillation_pk = distillations_fts.rowid
                {pilot_join}
                JOIN documents d ON d.id = x.document_id
                JOIN distillation_state s
                  ON s.document_id = x.document_id AND s.status = 'ready'
                WHERE distillations_fts MATCH ? {filters}
                ORDER BY lexical_score ASC LIMIT ?
                """,
                [fts_query, *filter_params, limit],
            ).fetchall()
            for rank, row in enumerate(rows, start=1):
                key = int(row["distillation_pk"])
                rows_by_pk[key] = row
                lexical_rank[key] = rank

        heap: list[tuple[float, int]] = []
        vector_rows = connection.execute(
            f"""
            SELECT x.* FROM distillations x
            {pilot_join}
            JOIN documents d ON d.id = x.document_id
            JOIN distillation_state s
              ON s.document_id = x.document_id AND s.status = 'ready'
            WHERE x.embedding_model = ? AND x.embedding_dimensions = ? {filters}
            """,
            [self.embedder.model_name, self.embedder.dimensions, *filter_params],
        )
        for row in vector_rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.shape != (self.embedder.dimensions,):
                continue
            candidate = (float(np.dot(query_vector, vector)), int(row["distillation_pk"]))
            if len(heap) < limit:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
            rows_by_pk[int(row["distillation_pk"])] = row
        for rank, (_, key) in enumerate(sorted(heap, reverse=True), start=1):
            vector_rank[key] = rank

        ranked_units: list[tuple[float, int]] = []
        for key in set(lexical_rank) | set(vector_rank):
            score = 0.0
            if key in lexical_rank:
                score += 1.0 / (self.settings.rrf_k + lexical_rank[key])
            if key in vector_rank:
                score += 1.0 / (self.settings.rrf_k + vector_rank[key])
            ranked_units.append((score, key))
        ranked_units.sort(reverse=True)

        chosen_by_document: dict[str, sqlite3.Row] = {}
        for _, key in ranked_units:
            row = rows_by_pk[key]
            chosen_by_document.setdefault(str(row["document_id"]), row)
            if len(chosen_by_document) >= limit:
                break

        anchors: list[int] = []
        matched: dict[int, str] = {}
        for row in chosen_by_document.values():
            raw_rows = connection.execute(
                """
                SELECT chunk_pk, embedding FROM chunks
                WHERE document_id = ? AND ordinal BETWEEN ? AND ?
                  AND embedding_model = ? AND embedding_dimensions = ?
                """,
                (
                    row["document_id"],
                    int(row["start_ordinal"]),
                    int(row["end_ordinal"]),
                    self.embedder.model_name,
                    self.embedder.dimensions,
                ),
            ).fetchall()
            if not raw_rows:
                continue
            best = max(
                raw_rows,
                key=lambda item: float(
                    np.dot(query_vector, np.frombuffer(item["embedding"], dtype=np.float32))
                ),
            )
            chunk_key = int(best["chunk_pk"])
            anchors.append(chunk_key)
            matched[chunk_key] = str(row["id"])
        return anchors, matched

    def search_response(
        self,
        query: str,
        *,
        limit: int = 8,
        sources: Sequence[str] | None = None,
        project: str | None = None,
        since: str | datetime | None = None,
        global_search: bool = False,
        rerank: bool | None = None,
        roots: Sequence[str | Path] | None = None,
        cwd: str | Path | None = None,
        include_distillations: bool | None = None,
    ) -> dict[str, Any]:
        safe_query = redact_text(query).strip()
        if not safe_query:
            raise ValueError("Search query cannot be empty")
        # Reranking and context expansion deliberately advance no more than the
        # configured top-20 document stage.
        limit = min(max(1, int(limit)), 20)
        candidate_limit = self.settings.candidate_limit
        scope = self.resolve_project_scope(
            project=project,
            global_search=global_search,
            roots=roots,
            cwd=cwd,
        )
        applied_project = scope["project"]
        filters, filter_params = self._filter_sql(
            sources=sources,
            project=applied_project,
            since=since,
        )
        detail_columns = """
            c.chunk_pk, c.id AS chunk_id, c.document_id, c.ordinal, c.content,
            d.title, d.source, d.project, d.timestamp, d.uri, d.metadata_json
        """
        query_embed = getattr(self.embedder, "embed_query", None)
        query_vector = query_embed(safe_query) if query_embed else self.embedder.embed([safe_query])[0]
        records: dict[int, sqlite3.Row] = {}
        lexical_rank: dict[int, int] = {}
        vector_rank: dict[int, int] = {}
        distillation_rank: dict[int, int] = {}
        distillation_match: dict[int, str] = {}

        with self._connect() as connection:
            fts_query = self._fts_query(safe_query)
            if fts_query:
                lexical_rows = connection.execute(
                    f"""
                    SELECT {detail_columns}, bm25(chunks_fts, 2.0, 1.0) AS lexical_score
                    FROM chunks_fts
                    JOIN chunks c ON c.chunk_pk = chunks_fts.rowid
                    JOIN documents d ON d.id = c.document_id
                    WHERE chunks_fts MATCH ? {filters}
                    ORDER BY lexical_score ASC LIMIT ?
                    """,
                    [fts_query, *filter_params, candidate_limit],
                ).fetchall()
                for rank_number, row in enumerate(lexical_rows, start=1):
                    key = int(row["chunk_pk"])
                    records[key] = row
                    lexical_rank[key] = rank_number

            vector_candidates, vector_meta = self._vector_candidates(
                connection,
                query_vector,
                filters=filters,
                filter_params=filter_params,
                limit=candidate_limit,
                sources=sources,
                project=applied_project,
                since=since,
            )
            for rank_number, (_, key) in enumerate(vector_candidates, start=1):
                vector_rank[key] = rank_number

            use_distillations = (
                self.settings.distillation.mode == "on"
                if include_distillations is None
                else bool(include_distillations)
            )
            if use_distillations:
                distillation_keys, distillation_match = self._distillation_candidates(
                    connection,
                    safe_query,
                    query_vector,
                    filters=filters,
                    filter_params=filter_params,
                    limit=candidate_limit,
                )
                distillation_rank = {
                    key: rank_number for rank_number, key in enumerate(distillation_keys, start=1)
                }

            missing_keys = (
                set(vector_rank) | set(distillation_rank)
            ) - set(records)
            if missing_keys:
                placeholders = ",".join("?" for _ in missing_keys)
                rows = connection.execute(
                    f"""
                    SELECT {detail_columns} FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.chunk_pk IN ({placeholders})
                    """,
                    sorted(missing_keys),
                ).fetchall()
                records.update({int(row["chunk_pk"]): row for row in rows})

        now = _utc_now()
        chunk_scores: dict[int, float] = {}
        best_by_document: dict[str, tuple[float, int, sqlite3.Row]] = {}
        for key, row in records.items():
            reciprocal = 0.0
            if key in lexical_rank:
                reciprocal += 1.0 / (self.settings.rrf_k + lexical_rank[key])
            if key in vector_rank:
                reciprocal += 1.0 / (self.settings.rrf_k + vector_rank[key])
            if key in distillation_rank:
                reciprocal += 1.0 / (self.settings.rrf_k + distillation_rank[key])
            age_days = max(
                0.0,
                (now - parse_timestamp(row["timestamp"])).total_seconds() / 86_400,
            )
            retrieval_score = reciprocal * (0.7 + 0.3 * math.exp(-age_days / 180.0))
            chunk_scores[key] = retrieval_score
            document_id = str(row["document_id"])
            candidate = (retrieval_score, -key, row)
            if document_id not in best_by_document or candidate[:2] > best_by_document[document_id][:2]:
                best_by_document[document_id] = candidate

        document_limit = min(
            20, max(limit, self.settings.reranker.candidate_documents)
        )
        document_candidates = sorted(
            best_by_document.values(),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )[:document_limit]

        variants: dict[str, dict[str, Any]] = {}
        passages: list[dict[str, Any]] = []
        with self._connect() as connection:
            neighbor_rows_by_document: dict[str, list[sqlite3.Row]] = {}
            if document_candidates:
                clauses: list[str] = []
                neighbor_params: list[Any] = []
                for _, _, anchor in document_candidates:
                    anchor_ordinal = int(anchor["ordinal"])
                    clauses.append("(c.document_id = ? AND c.ordinal BETWEEN ? AND ?)")
                    neighbor_params.extend(
                        [
                            anchor["document_id"],
                            max(0, anchor_ordinal - 1),
                            anchor_ordinal + 1,
                        ]
                    )
                neighbor_rows = connection.execute(
                    f"""
                    SELECT {detail_columns} FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE {' OR '.join(clauses)}
                    ORDER BY c.document_id, c.ordinal
                    """,
                    neighbor_params,
                ).fetchall()
                for row in neighbor_rows:
                    neighbor_rows_by_document.setdefault(
                        str(row["document_id"]), []
                    ).append(row)
            for retrieval_score, _, anchor in document_candidates:
                neighbors = neighbor_rows_by_document.get(str(anchor["document_id"]), [])
                anchor_ordinal = int(anchor["ordinal"])
                previous = next(
                    (row for row in neighbors if int(row["ordinal"]) == anchor_ordinal - 1),
                    None,
                )
                following = next(
                    (row for row in neighbors if int(row["ordinal"]) == anchor_ordinal + 1),
                    None,
                )
                choices: list[tuple[str, list[sqlite3.Row]]] = []
                if previous is not None:
                    choices.append(("previous", [previous, anchor]))
                if following is not None:
                    choices.append(("next", [anchor, following]))
                if not choices:
                    choices.append(("anchor", [anchor]))
                for direction, context_rows in choices:
                    variant_id = f"{anchor['document_id']}:{direction}"
                    snippets = [self._snippet(str(row["content"]), safe_query) for row in context_rows]
                    rerank_budget = max(100, 700 // len(context_rows))
                    rerank_snippets = [
                        self._snippet(str(row["content"]), safe_query, length=rerank_budget)
                        for row in context_rows
                    ]
                    variants[variant_id] = {
                        "document_id": str(anchor["document_id"]),
                        "anchor": anchor,
                        "rows": context_rows,
                        "retrieval_score": retrieval_score,
                    }
                    passages.append(
                        {
                            "id": variant_id,
                            # One balanced, query-centred ~700-character view
                            # of the two-chunk variant keeps both sides visible
                            # to the cross-encoder without padding all 40 pairs
                            # to its 512-token ceiling. Returned evidence keeps
                            # the full per-chunk 700-character snippets below.
                            "text": "\n\n".join(rerank_snippets),
                            "meta": {"document_id": str(anchor["document_id"])},
                        }
                    )

        should_rerank = self.settings.reranker.enabled if rerank is None else bool(rerank)
        reranker_applied = False
        reranker_error: str | None = None
        selected: dict[str, tuple[dict[str, Any], float | None]] = {}
        if should_rerank and passages:
            try:
                ranked = self.reranker.rerank(safe_query, passages)
                for item in ranked:
                    variant = variants.get(str(item.get("id")))
                    if variant is None:
                        continue
                    document_id = variant["document_id"]
                    score = float(item.get("score", 0.0))
                    current = selected.get(document_id)
                    if current is None or score > float(current[1] or 0.0):
                        selected[document_id] = (variant, score)
                reranker_applied = bool(selected)
            except RerankerUnavailable as exc:
                reranker_error = redact_text(str(exc))[:500]

        if not reranker_applied:
            selected = {}
            for retrieval_score, _, anchor in document_candidates:
                choices = [
                    variant for variant in variants.values()
                    if variant["document_id"] == anchor["document_id"]
                ]
                if not choices:
                    continue
                def fallback_key(variant: dict[str, Any]) -> tuple[float, int]:
                    other_scores = [
                        chunk_scores.get(int(row["chunk_pk"]), -1.0)
                        for row in variant["rows"]
                        if int(row["ordinal"]) != int(anchor["ordinal"])
                    ]
                    has_next = any(int(row["ordinal"]) > int(anchor["ordinal"]) for row in variant["rows"])
                    return (max(other_scores, default=-1.0), int(has_next))
                selected[str(anchor["document_id"])] = (max(choices, key=fallback_key), None)

        ordered = sorted(
            selected.values(),
            key=lambda item: (
                -(
                    float(item[1])
                    if reranker_applied and item[1] is not None
                    else float(item[0]["retrieval_score"])
                ),
                -float(item[0]["retrieval_score"]),
                str(item[0]["document_id"]),
            ),
        )
        results: list[dict[str, Any]] = []
        for variant, rerank_score in ordered[:limit]:
            anchor = variant["anchor"]
            anchor_key = int(anchor["chunk_pk"])
            context_chunks = [
                {
                    "chunk_id": row["chunk_id"],
                    "ordinal": int(row["ordinal"]),
                    "snippet": self._snippet(str(row["content"]), safe_query),
                    "citation": (
                        f"cerebras-memory://document/{row['document_id']}?chunk={row['chunk_id']}"
                    ),
                    "anchor": int(row["ordinal"]) == int(anchor["ordinal"]),
                    "content_trust": "untrusted_evidence",
                }
                for row in sorted(variant["rows"], key=lambda item: int(item["ordinal"]))
            ]
            matched_via = [
                name for name, mapping in (
                    ("lexical", lexical_rank),
                    ("vector", vector_rank),
                    ("distillation", distillation_rank),
                ) if anchor_key in mapping
            ]
            retrieval_score = float(variant["retrieval_score"])
            final_score = float(rerank_score) if reranker_applied and rerank_score is not None else retrieval_score
            results.append(
                {
                    "snippet": "\n\n".join(chunk["snippet"] for chunk in context_chunks),
                    "score": round(final_score, 8),
                    "retrieval_score": round(retrieval_score, 8),
                    "rerank_score": round(float(rerank_score), 8) if rerank_score is not None else None,
                    "score_stage": "reranker" if reranker_applied else "rrf",
                    "lexical_rank": lexical_rank.get(anchor_key),
                    "vector_rank": vector_rank.get(anchor_key),
                    "distillation_rank": distillation_rank.get(anchor_key),
                    "matched_via": matched_via,
                    "distillation_id": distillation_match.get(anchor_key),
                    "citation": (
                        f"cerebras-memory://document/{anchor['document_id']}?chunk={anchor['chunk_id']}"
                    ),
                    "source": anchor["source"],
                    "client": anchor["source"],
                    "project": anchor["project"],
                    "timestamp": anchor["timestamp"],
                    "title": anchor["title"],
                    "document_id": anchor["document_id"],
                    "chunk_id": anchor["chunk_id"],
                    "chunk_ordinal": int(anchor["ordinal"]),
                    "context_chunks": context_chunks,
                    "uri": anchor["uri"],
                    "metadata": json.loads(anchor["metadata_json"] or "{}"),
                    "content_trust": "untrusted_evidence",
                }
            )
        return {
            "results": results,
            "scope": scope,
            "retrieval": {
                "candidate_documents": len(document_candidates),
                "document_deduplication": True,
                "max_context_chunks": 2,
                "vector": vector_meta,
                "reranker": {
                    "requested": should_rerank,
                    "applied": reranker_applied,
                    "status": "applied" if reranker_applied else (
                        "fallback" if should_rerank else "disabled"
                    ),
                    "model": getattr(self.reranker, "model_name", None),
                    "degraded_reason": reranker_error,
                },
                "distillation": {
                    "enabled": use_distillations,
                    "candidates": len(distillation_rank),
                    "evidence": "raw_chunks_only",
                },
            },
        }

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        sources: Sequence[str] | None = None,
        project: str | None = None,
        since: str | datetime | None = None,
        global_search: bool = False,
        rerank: bool | None = None,
        roots: Sequence[str | Path] | None = None,
        cwd: str | Path | None = None,
        include_distillations: bool | None = None,
    ) -> list[dict[str, Any]]:
        return self.search_response(
            query,
            limit=limit,
            sources=sources,
            project=project,
            since=since,
            global_search=global_search,
            rerank=rerank,
            roots=roots,
            cwd=cwd,
            include_distillations=include_distillations,
        )["results"]

    def get_document(self, document_id: str, *, offset: int = 0, limit: int = 10) -> dict[str, Any] | None:
        offset = max(0, int(offset))
        limit = min(max(1, int(limit)), 50)
        with self._connect() as connection:
            document = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if document is None:
                return None
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,)
                ).fetchone()[0]
            )
            chunks = connection.execute(
                """
                SELECT id, ordinal, content, content_hash
                FROM chunks WHERE document_id = ? ORDER BY ordinal LIMIT ? OFFSET ?
                """,
                (document_id, limit, offset),
            ).fetchall()
        return {
            "document": {
                "document_id": document["id"],
                "source": document["source"],
                "kind": document["kind"],
                "title": document["title"],
                "uri": document["uri"],
                "timestamp": document["timestamp"],
                "project": document["project"],
                "metadata": json.loads(document["metadata_json"] or "{}"),
                "citation": f"cerebras-memory://document/{document['id']}",
                "content_trust": "untrusted_evidence",
            },
            "chunks": [
                {
                    "chunk_id": row["id"],
                    "ordinal": int(row["ordinal"]),
                    "content": row["content"],
                    "content_hash": row["content_hash"],
                    "citation": (
                        f"cerebras-memory://document/{document_id}?chunk={row['id']}"
                    ),
                    "content_trust": "untrusted_evidence",
                }
                for row in chunks
            ],
            "pagination": {
                "offset": offset,
                "limit": limit,
                "returned": len(chunks),
                "total_chunks": total,
                "next_offset": offset + len(chunks) if offset + len(chunks) < total else None,
            },
        }

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            document_count = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            chunk_count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            memory_count = int(
                connection.execute("SELECT COUNT(*) FROM documents WHERE kind = 'memory'").fetchone()[0]
            )
            by_source = {
                row["source"]: int(row["count"])
                for row in connection.execute(
                    "SELECT source, COUNT(*) AS count FROM documents GROUP BY source ORDER BY source"
                )
            }
            current_embeddings = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM chunks
                    WHERE embedding_model = ? AND embedding_dimensions = ?
                    """,
                    (self.embedder.model_name, self.embedder.dimensions),
                ).fetchone()[0]
            )
            state_rows = connection.execute(
                "SELECT * FROM ingest_state ORDER BY source"
            ).fetchall()
            latest_refresh = connection.execute(
                "SELECT MAX(last_success_at) FROM ingest_state"
            ).fetchone()[0]
        source_state = {
            row["source"]: {
                "watermark": row["watermark"],
                "status": row["status"],
                "last_started_at": row["last_started_at"],
                "last_success_at": row["last_success_at"],
                "last_failure_at": row["last_failure_at"],
                "scanned": int(row["scanned"]),
                "imported": int(row["imported"]),
                "skipped": int(row["skipped"]),
                "failures": int(row["failures"]),
                "last_error": row["last_error"],
            }
            for row in state_rows
        }
        production_backend = isinstance(self.embedder, FastEmbedder)
        model_cached = (
            any(self.settings.model_cache_dir.rglob("*.onnx"))
            if production_backend and self.settings.model_cache_dir.exists()
            else not production_backend
        )
        if not model_cached:
            embedding_status = "model_not_cached"
        elif max(0, chunk_count - current_embeddings):
            embedding_status = "reembed_required"
        else:
            embedding_status = "ready"
        reranker_status = self.reranker.status()
        vector_status = self.vector_index_status()
        distillation_status = self.distillation_status()
        return {
            "schema_version": self.schema_version(),
            "database_path": str(self.database_path),
            "documents": document_count,
            "chunks": chunk_count,
            "saved_memories": memory_count,
            "documents_by_source": by_source,
            "embedding": {
                "backend": "fastembed" if production_backend else "test",
                "model": self.embedder.model_name,
                "dimensions": self.embedder.dimensions,
                "status": embedding_status,
                "model_cached": model_cached,
                "indexed_chunks": current_embeddings,
                "pending_reembed": max(0, chunk_count - current_embeddings),
                "cache_path": str(self.settings.model_cache_dir),
                "cache_exists": self.settings.model_cache_dir.exists(),
            },
            "reranker": reranker_status,
            "vector_search": vector_status,
            "distillation": distillation_status,
            "last_refresh": latest_refresh,
            "refresh_in_progress": any(
                state["status"] == "running" for state in source_state.values()
            ),
            "sources": source_state,
            "failures": {
                source: state["last_error"]
                for source, state in source_state.items()
                if state["status"] == "failed"
            },
        }


# Original scaffold imported ``Store``. Keep the name while exposing the new API.
Store = KnowledgeStore
