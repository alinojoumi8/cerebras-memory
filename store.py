"""Versioned SQLite storage and local hybrid retrieval."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import sqlite3
import statistics
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid
from urllib.parse import urlparse

import numpy as np

from chunking import Chunk, chunk_document, chunk_text
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


SCHEMA_VERSION = 5

class ReconcileFloorNotMet(RuntimeError):
    """A scan returned implausibly few keys, so deletion was refused."""


# Below this many indexed documents a source is too small for the ratio floor to
# distinguish a broken scan from ordinary cleanup. Every real source in a working
# deployment sits far above it.
_RECONCILE_FLOOR_MIN_DOCUMENTS = 10


_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)

# Content-free tokens. These carry almost no retrieval signal but dominate a
# naive query tokenization: they appear in nearly every chunk, so letting them
# anchor a snippet window or consume the lexical candidate budget crowds out the
# rare terms that actually discriminate.
_STOPWORDS = frozenset(
    """
    a about after all also am an and any are as at be because been before being
    but by can cannot could did do does doing done down during each few for from
    further had has have having he her here hers him his how i if in into is it
    its just me more most my no nor not of off on once only or other our out over
    own same she should so some such than that the their them then there these
    they this those through to too under until up very was we were what when
    where which while who whom why will with would you your
    """.split()
)


# Total characters of passage text handed to the cross-encoder per variant,
# split across that variant's chunks. Sized against the reranker's 512-token
# ceiling at roughly four characters per token, leaving room for the query.
_RERANK_PASSAGE_BUDGET = 700

# Maximum distinct terms in a single FTS5 MATCH expression.
_MAX_FTS_TERMS = 24

# Bind-parameter window for IN (...) lookups, well under SQLite's limit.
_SQL_PARAM_BATCH = 500

# Bump whenever chunk boundaries or the embedded text change. Documents whose
# chunks carry a different version are re-chunked on an ordinary incremental
# refresh, so a chunking change no longer requires a full rebuild.
CHUNKER_VERSION = "heading-breadcrumb-v1"

# A refresh heartbeats every 60 s. Five missed beats means the owner is gone, so
# its lease can be reclaimed without waiting out the full expiry.
_LEASE_HEARTBEAT_GRACE_SECONDS = 300

# Sentinel model name the deterministic test embedder is bound to.
_TEST_EMBEDDING_MODEL = "test/hash-v1"


# Extensions whose structure is worth chunking on rather than through.
_MARKDOWN_SUFFIXES = frozenset({".md", ".mdx", ".markdown"})


def _is_markdown(item: Any) -> bool:
    """Whether a prepared document should use the heading-aware chunker.

    Only project documents qualify. Agent transcripts are stored as
    ``USER [ts] / ASSISTANT [ts]`` blocks whose ``#`` lines are dialogue content,
    not structure, and re-chunking them on headings would break the ordinal
    ranges that distillation maps evidence back through.
    """

    if getattr(item, "source", "") != "projects":
        return False
    extension = ""
    try:
        metadata = json.loads(getattr(item, "metadata_json", "") or "{}")
    except (TypeError, ValueError):
        metadata = {}
    if isinstance(metadata, Mapping):
        extension = str(metadata.get("extension") or "").casefold()
    if not extension:
        extension = Path(str(getattr(item, "title", ""))).suffix.casefold()
    return extension in _MARKDOWN_SUFFIXES


def _embedding_text(item: Any, chunk: Chunk) -> str:
    """Text handed to the embedder: breadcrumb first, then the chunk body."""

    if not chunk.breadcrumb:
        return chunk.text
    return f"{getattr(item, 'title', '')} › {chunk.breadcrumb}\n\n{chunk.text}"


def _query_tokens(query: str) -> list[str]:
    """Ordered, unique, case-folded word tokens for a query."""

    return list(dict.fromkeys(_WORD_RE.findall(query.casefold())))
_UTC = timezone.utc
_SCHEMA_INITIALIZATION_LOCK = threading.Lock()
_DEFAULT_REFRESH_LEASE_SECONDS = 30 * 60


@dataclass(frozen=True)
class RefreshLease:
    run_id: str
    token: str
    expires_at: str


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
    pieces: list[Chunk] = field(default_factory=list)
    vectors: list[np.ndarray] = field(default_factory=list)
    chunk_count: int = 0


@dataclass(frozen=True)
class _ExactVectorSnapshot:
    generation: int
    model: str
    dimensions: int
    keys: np.ndarray
    vectors: np.ndarray
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
            # Deliberate: an unparseable timestamp sorts as maximally old rather
            # than failing ingestion or search.  Note the ranking consequence --
            # the recency factor in search turns epoch 0 into a permanent x0.7
            # multiplier, so a source that systematically emits bad timestamps
            # is quietly penalised rather than rejected.  Importers are expected
            # to normalise before reaching here.
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


def _stable_receipt_id(
    artifact_type: str,
    artifact_id: str,
    content_hash: str,
    producer: str,
    producer_version: str,
) -> str:
    material = "\0".join(
        (artifact_type, artifact_id, content_hash, producer, producer_version)
    )
    return f"rcpt_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


_EXECUTABLE_INSTRUCTION_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:"
    r"ignore\s+(?:all\s+)?previous|system\s*:|developer\s*:|"
    r"run\s+(?:this|the following|powershell|cmd)|execute\s+(?:this|the following)|"
    r"(?:sudo|powershell|cmd(?:\.exe)?|bash|sh)\s+[-/]|"
    r"curl\s+https?://|invoke-expression\b"
    r")"
)


def _content_taints(content: str, *, externally_processed: bool = False) -> list[str]:
    taints = ["untrusted_evidence"]
    if _EXECUTABLE_INSTRUCTION_RE.search(content):
        taints.append("executable_instruction")
    if externally_processed:
        taints.append("externally_processed")
    return taints


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
        ingest_mode: bool = False,
    ):
        # ingest_mode selects the batch thread budget instead of the serving cap.
        self.settings = settings or load_settings()
        self.settings.ensure_runtime_directories()
        self.database_path = Path(self.settings.database_path)
        if embedder is None:
            if os.environ.get("CEREBRAS_MEMORY_TEST_EMBEDDER") == "1":
                # This hook swaps semantic embeddings for a bag-of-words hash and
                # disables the reranker. Retrieval quality collapses silently, so
                # require the configuration to opt in as well: an ambient
                # environment variable alone must not be able to downgrade a real
                # deployment, and MCP clients control the server's environment.
                if self.settings.embedding_model != _TEST_EMBEDDING_MODEL:
                    raise RuntimeError(
                        "CEREBRAS_MEMORY_TEST_EMBEDDER=1 requires "
                        f'embedding_model="{_TEST_EMBEDDING_MODEL}", but this '
                        f'configuration uses "{self.settings.embedding_model}". '
                        "Refusing to silently downgrade retrieval quality."
                    )
                embedder = HashingEmbedder(dimensions=self.settings.embedding_dimensions)
            else:
                embedder = FastEmbedder(
                    self.settings.embedding_model,
                    self.settings.embedding_dimensions,
                    self.settings.model_cache_dir,
                    query_prefix=self.settings.embedding_query_prefix,
                    threads=(
                        self.settings.ingest_embedding_threads
                        if ingest_mode
                        else self.settings.embedding_threads
                    ),
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
                # The version this process found on disk, before any migration
                # below advances it.  Used to decide whether the legacy repair
                # pass still has anything to do.
                opened_version = current
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
                        embedding_input_hash TEXT,
                        chunker_version TEXT,
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
                        status TEXT NOT NULL CHECK(
                            status IN ('pending', 'ready', 'failed', 'policy_blocked')
                        ),
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
                    current = 2
                if current < 3:
                    connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS refresh_runs (
                        run_id TEXT PRIMARY KEY,
                        mode TEXT NOT NULL,
                        owner_pid INTEGER NOT NULL,
                        owner_host TEXT NOT NULL,
                        lease_token_hash TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(
                            status IN ('running', 'succeeded', 'failed', 'abandoned')
                        ),
                        current_source TEXT,
                        started_at TEXT NOT NULL,
                        heartbeat_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        completed_at TEXT,
                        summary_json TEXT,
                        last_error TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_refresh_runs_status
                        ON refresh_runs(status, started_at);

                    CREATE TABLE IF NOT EXISTS refresh_lease (
                        id INTEGER PRIMARY KEY CHECK(id = 1),
                        run_id TEXT NOT NULL REFERENCES refresh_runs(run_id) ON DELETE CASCADE,
                        lease_token_hash TEXT NOT NULL,
                        acquired_at TEXT NOT NULL,
                        heartbeat_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS outbound_distillation_audit (
                        audit_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT NOT NULL UNIQUE,
                        document_id TEXT,
                        unit_input_hash TEXT NOT NULL,
                        source TEXT NOT NULL,
                        project TEXT,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        endpoint_host TEXT NOT NULL,
                        character_count INTEGER NOT NULL,
                        decision TEXT NOT NULL,
                        status TEXT NOT NULL,
                        error_code TEXT,
                        created_at TEXT NOT NULL,
                        completed_at TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_outbound_audit_document
                        ON outbound_distillation_audit(document_id, created_at);
                    CREATE INDEX IF NOT EXISTS idx_outbound_audit_status
                        ON outbound_distillation_audit(decision, status, created_at);

                    CREATE TABLE IF NOT EXISTS provenance_receipts (
                        id TEXT PRIMARY KEY,
                        artifact_type TEXT NOT NULL,
                        artifact_id TEXT NOT NULL,
                        document_id TEXT,
                        source TEXT,
                        project TEXT,
                        content_hash TEXT NOT NULL,
                        producer TEXT NOT NULL,
                        producer_version TEXT NOT NULL,
                        taints_json TEXT NOT NULL DEFAULT '[]',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        superseded_at TEXT,
                        UNIQUE(
                            artifact_type, artifact_id, content_hash,
                            producer, producer_version
                        )
                    );
                    CREATE INDEX IF NOT EXISTS idx_provenance_artifact
                        ON provenance_receipts(artifact_type, artifact_id, created_at);
                    CREATE INDEX IF NOT EXISTS idx_provenance_document
                        ON provenance_receipts(document_id, artifact_type);

                    CREATE TABLE IF NOT EXISTS provenance_edges (
                        parent_receipt_id TEXT NOT NULL
                            REFERENCES provenance_receipts(id) ON DELETE CASCADE,
                        child_receipt_id TEXT NOT NULL
                            REFERENCES provenance_receipts(id) ON DELETE CASCADE,
                        relation TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(parent_receipt_id, child_receipt_id, relation)
                    );

                    CREATE TABLE IF NOT EXISTS deletion_manifests (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        source_key_hash TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        chunk_count INTEGER NOT NULL,
                        distillation_count INTEGER NOT NULL,
                        manifest_json TEXT NOT NULL,
                        manifest_hash TEXT NOT NULL,
                        deleted_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_deletion_manifests_document
                        ON deletion_manifests(document_id, deleted_at);

                    CREATE TABLE IF NOT EXISTS canary_runs (
                        run_id TEXT PRIMARY KEY,
                        suite_version TEXT NOT NULL,
                        suite_hash TEXT NOT NULL,
                        status TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        completed_at TEXT,
                        cases_total INTEGER NOT NULL DEFAULT 0,
                        cases_passed INTEGER NOT NULL DEFAULT 0,
                        p95_latency_ms REAL,
                        report_json TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE TABLE IF NOT EXISTS canary_case_results (
                        run_id TEXT NOT NULL REFERENCES canary_runs(run_id) ON DELETE CASCADE,
                        case_id TEXT NOT NULL,
                        passed INTEGER NOT NULL,
                        latency_ms REAL NOT NULL,
                        result_json TEXT NOT NULL,
                        PRIMARY KEY(run_id, case_id)
                    );
                    CREATE TABLE IF NOT EXISTS quality_gate_state (
                        id INTEGER PRIMARY KEY CHECK(id = 1),
                        status TEXT NOT NULL,
                        suite_version TEXT,
                        last_run_id TEXT,
                        updated_at TEXT NOT NULL,
                        detail_json TEXT NOT NULL DEFAULT '{}'
                    );
                    INSERT OR IGNORE INTO quality_gate_state(
                        id, status, updated_at, detail_json
                    ) VALUES (
                        1, 'not_run',
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        '{}'
                    );

                    INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                    VALUES (3, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
                    PRAGMA user_version = 3;
                    COMMIT;
                    """
                    )
                    # Keep the tracking variable in step with user_version, as
                    # v1 and v2 do.  Nothing below reads it today, so the
                    # omission was harmless -- but a v2 database would reach the
                    # next migration added here still claiming to be v2.
                    current = 3
                if current < 4:
                    # Units belonging to a policy-blocked document were recorded
                    # as 'pending', which reads as retryable work. The
                    # blocked-decision cache means they are never retried, so the
                    # pipeline could not converge and stats reported outstanding
                    # work that would never finish. Add a terminal status and
                    # reclassify the existing rows. SQLite cannot alter a CHECK
                    # constraint, so the table is rebuilt in place.
                    connection.executescript(
                        """
                    BEGIN IMMEDIATE;
                    CREATE TABLE distillation_unit_state_v4 (
                        unit_state_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        input_hash TEXT NOT NULL,
                        unit_ordinal INTEGER NOT NULL,
                        start_ordinal INTEGER NOT NULL,
                        end_ordinal INTEGER NOT NULL,
                        distiller_model TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(
                            status IN ('pending', 'ready', 'failed', 'policy_blocked')
                        ),
                        attempts INTEGER NOT NULL DEFAULT 0,
                        last_attempt_at TEXT,
                        last_success_at TEXT,
                        last_error TEXT,
                        UNIQUE(document_id, input_hash, distiller_model, prompt_version)
                    );
                    INSERT INTO distillation_unit_state_v4(
                        unit_state_pk, document_id, input_hash, unit_ordinal,
                        start_ordinal, end_ordinal, distiller_model, prompt_version,
                        status, attempts, last_attempt_at, last_success_at, last_error
                    )
                    SELECT
                        u.unit_state_pk, u.document_id, u.input_hash, u.unit_ordinal,
                        u.start_ordinal, u.end_ordinal, u.distiller_model, u.prompt_version,
                        CASE
                            WHEN u.status = 'pending' AND s.status = 'blocked'
                                THEN 'policy_blocked'
                            ELSE u.status
                        END,
                        u.attempts, u.last_attempt_at, u.last_success_at, u.last_error
                    FROM distillation_unit_state u
                    LEFT JOIN distillation_state s ON s.document_id = u.document_id;

                    DROP TABLE distillation_unit_state;
                    ALTER TABLE distillation_unit_state_v4
                        RENAME TO distillation_unit_state;
                    CREATE INDEX IF NOT EXISTS idx_distillation_unit_state_status
                        ON distillation_unit_state(status, document_id);

                    INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                    VALUES (4, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
                    PRAGMA user_version = 4;
                    COMMIT;
                    """
                    )
                if current < 5:
                    # Two columns that make re-indexing cheap.
                    #
                    # embedding_input_hash lets a rebuild reuse an existing vector
                    # instead of recomputing it. It hashes the *embedding input*,
                    # not the chunk text, because _embedding_text prepends the
                    # heading breadcrumb. The backfill below is exact rather than
                    # approximate: no chunk written before this migration carried a
                    # breadcrumb, and for those _embedding_text returns the bare
                    # chunk text, so the input hash is precisely content_hash.
                    #
                    # chunker_version lets a chunking change invalidate only the
                    # documents it affects, the same way an embedding-model change
                    # already does, instead of forcing a full rebuild. Existing
                    # rows stay NULL so they are re-chunked exactly once.
                    # A freshly created database already has both columns from the
                    # v1 table definition, so the ALTERs must be conditional -
                    # SQLite has no ADD COLUMN IF NOT EXISTS.
                    existing_columns = {
                        str(row["name"])
                        for row in connection.execute("PRAGMA table_info(chunks)")
                    }
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        if "embedding_input_hash" not in existing_columns:
                            connection.execute(
                                "ALTER TABLE chunks ADD COLUMN embedding_input_hash TEXT"
                            )
                        if "chunker_version" not in existing_columns:
                            connection.execute(
                                "ALTER TABLE chunks ADD COLUMN chunker_version TEXT"
                            )
                        connection.execute(
                            "UPDATE chunks SET embedding_input_hash = content_hash "
                            "WHERE embedding_input_hash IS NULL"
                        )
                        connection.execute(
                            """
                            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_input
                            ON chunks(
                                embedding_input_hash, embedding_model, embedding_dimensions
                            )
                            """
                        )
                        connection.execute(
                            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) "
                            "VALUES (5, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
                        )
                        connection.execute("PRAGMA user_version = 5")
                        connection.commit()
                    except Exception:
                        connection.rollback()
                        raise
                # Early v2 builds aggregated retry state at document level.
                # This additive, idempotent repair gives existing v2 databases
                # the required per-unit pending/failed state without changing
                # the schema version or any raw document/chunk identity.
                #
                # Only databases opened below the current schema can still need
                # it: the v2 migration creates both tables outright, so for an
                # up-to-date database this was a full scan of every distillation
                # row against a UNIQUE index on every KnowledgeStore
                # construction -- and MCP builds one store per client.
                if opened_version < SCHEMA_VERSION:
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
                        status TEXT NOT NULL CHECK(
                            status IN ('pending', 'ready', 'failed', 'policy_blocked')
                        ),
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
                self._backfill_provenance(connection)

    @staticmethod
    def _insert_provenance_receipt(
        connection: sqlite3.Connection,
        *,
        artifact_type: str,
        artifact_id: str,
        document_id: str | None,
        source: str | None,
        project: str | None,
        content_hash: str,
        producer: str,
        producer_version: str,
        taints: Sequence[str],
        metadata: dict[str, Any] | None = None,
        parent_receipt_ids: Sequence[str] = (),
    ) -> str:
        receipt_id = _stable_receipt_id(
            artifact_type,
            artifact_id,
            content_hash,
            producer,
            producer_version,
        )
        now = _iso(None)
        connection.execute(
            """
            UPDATE provenance_receipts
            SET superseded_at = COALESCE(superseded_at, ?)
            WHERE artifact_type = ? AND artifact_id = ? AND id <> ?
              AND superseded_at IS NULL
            """,
            (now, artifact_type, artifact_id, receipt_id),
        )
        # ON CONFLICT rather than INSERT OR IGNORE: the receipt id is derived
        # from the content hash, so an artifact whose content returns to a value
        # it previously held (a file edited then reverted, a transcript
        # truncated back) collides with its own superseded row.  Ignoring the
        # conflict would leave that artifact with *zero* active receipts --
        # search would report provenance: null and fall back to default taints,
        # and _backfill_provenance's parity check could never be satisfied
        # again, re-running a full corpus backfill on every store construction.
        # Reactivating the existing row is the correct reading: same artifact,
        # same content, same producer, so the same receipt is valid once more.
        # created_at is left alone -- the receipt's identity has not changed.
        connection.execute(
            """
            INSERT INTO provenance_receipts(
                id, artifact_type, artifact_id, document_id, source, project,
                content_hash, producer, producer_version, taints_json,
                metadata_json, created_at, superseded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(id) DO UPDATE SET
                superseded_at = NULL,
                document_id = excluded.document_id,
                source = excluded.source,
                project = excluded.project,
                taints_json = excluded.taints_json,
                metadata_json = excluded.metadata_json
            """,
            (
                receipt_id,
                artifact_type,
                artifact_id,
                document_id,
                source,
                project,
                content_hash,
                producer,
                producer_version,
                json.dumps(sorted(set(taints)), ensure_ascii=False),
                json.dumps(
                    _redact_json_value(metadata or {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                now,
            ),
        )
        for parent_receipt_id in parent_receipt_ids:
            connection.execute(
                """
                INSERT OR IGNORE INTO provenance_edges(
                    parent_receipt_id, child_receipt_id, relation, created_at
                ) VALUES (?, ?, 'derived_from', ?)
                """,
                (parent_receipt_id, receipt_id, now),
            )
        return receipt_id

    def _backfill_provenance(self, connection: sqlite3.Connection) -> None:
        self._supersede_orphaned_provenance(connection)
        expected = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        expected += int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        expected += int(connection.execute("SELECT COUNT(*) FROM distillations").fetchone()[0])
        present = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM provenance_receipts
                WHERE artifact_type IN ('document', 'chunk', 'distillation')
                  AND superseded_at IS NULL
                """
            ).fetchone()[0]
        )
        if present >= expected:
            return

        connection.execute("BEGIN IMMEDIATE")
        try:
            document_receipts: dict[str, str] = {}
            document_rows = connection.execute(
                """
                SELECT id, source, project, content_hash, metadata_json
                FROM documents ORDER BY id
                """
            ).fetchall()
            for row in document_rows:
                document_id = str(row["id"])
                document_receipts[document_id] = self._insert_provenance_receipt(
                    connection,
                    artifact_type="document",
                    artifact_id=document_id,
                    document_id=document_id,
                    source=str(row["source"]),
                    project=row["project"],
                    content_hash=str(row["content_hash"]),
                    producer=f"ingest:{row['source']}",
                    producer_version="schema-v3",
                    taints=["untrusted_evidence"],
                    metadata={"backfilled": True},
                )

            for row in connection.execute(
                """
                SELECT c.id, c.document_id, c.content_hash, c.content,
                       d.source, d.project
                FROM chunks c JOIN documents d ON d.id = c.document_id
                ORDER BY c.document_id, c.ordinal
                """
            ):
                document_id = str(row["document_id"])
                self._insert_provenance_receipt(
                    connection,
                    artifact_type="chunk",
                    artifact_id=str(row["id"]),
                    document_id=document_id,
                    source=str(row["source"]),
                    project=row["project"],
                    content_hash=str(row["content_hash"]),
                    producer="paragraph_chunker",
                    producer_version=(
                        f"{self.settings.chunk_size}:{self.settings.chunk_overlap}:schema-v3"
                    ),
                    taints=_content_taints(str(row["content"])),
                    metadata={"backfilled": True},
                    parent_receipt_ids=[document_receipts[document_id]],
                )

            for row in connection.execute(
                """
                SELECT x.id, x.document_id, x.input_hash, x.search_text,
                       x.distiller_model, x.prompt_version, d.source, d.project
                FROM distillations x JOIN documents d ON d.id = x.document_id
                ORDER BY x.document_id, x.unit_ordinal
                """
            ):
                document_id = str(row["document_id"])
                self._insert_provenance_receipt(
                    connection,
                    artifact_type="distillation",
                    artifact_id=str(row["id"]),
                    document_id=document_id,
                    source=str(row["source"]),
                    project=row["project"],
                    content_hash=hashlib.sha256(
                        str(row["search_text"]).encode("utf-8")
                    ).hexdigest(),
                    producer=f"distiller:{row['distiller_model']}",
                    producer_version=str(row["prompt_version"]),
                    taints=_content_taints(
                        str(row["search_text"]),
                        externally_processed=True,
                    ),
                    metadata={
                        "input_hash": str(row["input_hash"]),
                        "backfilled": True,
                    },
                    parent_receipt_ids=[document_receipts[document_id]],
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @staticmethod
    def _supersede_orphaned_provenance(connection: sqlite3.Connection) -> int:
        cursor = connection.execute(
            """
            UPDATE provenance_receipts
            SET superseded_at = ?
            WHERE superseded_at IS NULL AND (
                (
                    artifact_type = 'document'
                    AND NOT EXISTS (
                        SELECT 1 FROM documents d WHERE d.id = artifact_id
                    )
                )
                OR (
                    artifact_type = 'chunk'
                    AND NOT EXISTS (
                        SELECT 1 FROM chunks c WHERE c.id = artifact_id
                    )
                )
                OR (
                    artifact_type = 'distillation'
                    AND NOT EXISTS (
                        SELECT 1 FROM distillations x WHERE x.id = artifact_id
                    )
                )
            )
            """,
            (_iso(None),),
        )
        return int(cursor.rowcount)

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
        """Whether a document's stored chunks are stale.

        A chunker change counts as staleness just like an embedding-model change:
        without this, altering how documents are split required a `--full`
        rebuild of the entire corpus, because content hashes were unchanged and
        every document therefore looked up to date.
        """

        row = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(
                       CASE WHEN embedding_model = ? AND embedding_dimensions = ?
                                 AND chunker_version = ?
                            THEN 1 ELSE 0 END
                   ) AS current
            FROM chunks WHERE document_id = ?
            """,
            (
                self.embedder.model_name,
                self.embedder.dimensions,
                CHUNKER_VERSION,
                document_id,
            ),
        ).fetchone()
        return int(row["total"] or 0) == 0 or int(row["current"] or 0) != int(row["total"] or 0)

    def _reusable_embeddings(
        self,
        connection: sqlite3.Connection,
        input_hashes: Sequence[str],
    ) -> dict[str, np.ndarray]:
        """Existing vectors for embedding inputs we have already encoded.

        Re-encoding text that has not changed is the single largest avoidable
        cost in a rebuild: a chunking change that only affects markdown still
        recomputed every conversation chunk. Old rows are still present here
        because embeddings are computed before ``BEGIN IMMEDIATE``, so a document
        can reuse its own previous vectors, and identical text shared across
        files is reused for free.
        """

        if not input_hashes:
            return {}
        found: dict[str, np.ndarray] = {}
        unique = list(dict.fromkeys(input_hashes))
        for start in range(0, len(unique), _SQL_PARAM_BATCH):
            window = unique[start : start + _SQL_PARAM_BATCH]
            placeholders = ",".join("?" for _ in window)
            rows = connection.execute(
                f"""
                SELECT embedding_input_hash, embedding
                FROM chunks
                WHERE embedding_input_hash IN ({placeholders})
                  AND embedding_model = ? AND embedding_dimensions = ?
                GROUP BY embedding_input_hash
                """,
                (*window, self.embedder.model_name, self.embedder.dimensions),
            ).fetchall()
            for row in rows:
                vector = np.frombuffer(row["embedding"], dtype=np.float32)
                if vector.shape == (self.embedder.dimensions,):
                    found[str(row["embedding_input_hash"])] = np.ascontiguousarray(vector)
        return found

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
        """Redact, chunk, embed and atomically replace documents, in batches.

        Work is committed every ``ingest_batch_documents`` documents rather than
        once per source. A full rebuild previously held an entire source in
        memory - 3.7 GB and ~100 minutes for `projects` - showed no progress the
        whole time, and lost everything on a crash or on the scheduled task's
        six-hour limit. Batching bounds the memory, makes progress observable,
        and makes an interrupted rebuild resumable: each batch is durable, and
        the next run re-derives only what is still stale.

        Per-document atomicity is unchanged - a document is fully written or not
        at all - and ingest.py still reconciles only after a completely
        successful pass, so a partial pass can never trigger deletions.
        """

        prepared = [self._prepare_document(document) for document in documents]
        if not prepared:
            return []
        self._classify_documents(prepared, force=force)

        batch_size = max(1, self.settings.ingest_batch_documents)
        results: list[WriteResult] = []
        totals = {"total": 0, "reused": 0, "embedded": 0}
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            self._embed_prepared(batch)
            for key in totals:
                totals[key] += int(self.last_embedding_reuse.get(key, 0))
            results.extend(self._write_prepared(batch))
        self.last_embedding_reuse = totals
        return results

    def _classify_documents(
        self,
        prepared: Sequence[_PreparedDocument],
        *,
        force: bool,
    ) -> None:
        """Decide which documents are stale and chunk the ones that are."""

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
                    item.pieces = chunk_document(
                        item.text,
                        target_size=self.settings.chunk_size,
                        overlap=self.settings.chunk_overlap,
                        markdown=_is_markdown(item),
                    )


    def _embed_prepared(self, prepared: Sequence[_PreparedDocument]) -> None:
        """Attach vectors to every stale document in this batch."""

        locations: list[tuple[_PreparedDocument, Chunk]] = [
            (item, piece)
            for item in prepared
            if not item.unchanged
            for piece in item.pieces
        ]
        ingestion_embed = getattr(self.embedder, "embed_for_ingestion", self.embedder.embed)
        # The breadcrumb is embedded with the chunk but never stored as its
        # content: a fragment reading "we decided to keep the 30-second timeout"
        # is otherwise encoded with no signal about which project, file or
        # section it belongs to, and no amount of neighbour expansion at query
        # time can put that back. Returned evidence stays the raw chunk, so
        # citations and provenance are unaffected.
        embedding_texts = [_embedding_text(item, piece) for item, piece in locations]
        input_hashes = [
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in embedding_texts
        ]
        with self._connect() as connection:
            reusable = self._reusable_embeddings(connection, input_hashes)
        pending_indexes = [
            index for index, digest in enumerate(input_hashes) if digest not in reusable
        ]
        fresh = (
            ingestion_embed([embedding_texts[index] for index in pending_indexes])
            if pending_indexes
            else []
        )
        vectors: list[np.ndarray] = [None] * len(locations)  # type: ignore[list-item]
        for position, index in enumerate(pending_indexes):
            vectors[index] = fresh[position]
            # Later chunks in this same pass can reuse what we just computed.
            reusable.setdefault(input_hashes[index], fresh[position])
        for index, digest in enumerate(input_hashes):
            if vectors[index] is None:
                vectors[index] = reusable[digest]
        self.last_embedding_reuse = {
            "total": len(locations),
            "reused": len(locations) - len(pending_indexes),
            "embedded": len(pending_indexes),
        }
        if len(vectors) != len(locations):
            raise RuntimeError("Embedding backend returned an unexpected vector count")
        for (item, _), vector in zip(locations, vectors, strict=True):
            if np.asarray(vector).shape != (self.embedder.dimensions,):
                raise ValueError("Embedding backend returned an unexpected vector dimension")
            item.vectors.append(np.asarray(vector, dtype=np.float32))


    def _write_prepared(self, prepared: Sequence[_PreparedDocument]) -> list[WriteResult]:
        """Commit one batch of prepared documents."""

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
                        self._insert_provenance_receipt(
                            connection,
                            artifact_type="document",
                            artifact_id=item.document_id,
                            document_id=item.document_id,
                            source=item.source,
                            project=item.project,
                            content_hash=item.content_hash,
                            producer=f"ingest:{item.source}",
                            producer_version="schema-v3",
                            taints=_content_taints(item.text),
                            metadata={
                                "kind": item.kind,
                                "timestamp": item.timestamp,
                                "unchanged_refresh": True,
                            },
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
                    document_receipt_id = self._insert_provenance_receipt(
                        connection,
                        artifact_type="document",
                        artifact_id=item.document_id,
                        document_id=item.document_id,
                        source=item.source,
                        project=item.project,
                        content_hash=item.content_hash,
                        producer=f"ingest:{item.source}",
                        producer_version="schema-v3",
                        taints=_content_taints(item.text),
                        metadata={
                            "kind": item.kind,
                            "timestamp": item.timestamp,
                        },
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
                    for ordinal, (chunk, vector) in enumerate(
                        zip(item.pieces, item.vectors, strict=True)
                    ):
                        piece = chunk.text
                        # ``title`` is FTS-indexed at weight 2.0, so the
                        # breadcrumb becomes lexically searchable too.
                        chunk_title = (
                            f"{item.title} › {chunk.breadcrumb}"
                            if chunk.breadcrumb
                            else item.title
                        )
                        chunk_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()
                        chunk_id = stable_chunk_id(item.document_id, ordinal)
                        connection.execute(
                            """
                            INSERT INTO chunks(
                                id, document_id, ordinal, title, content, content_hash,
                                embedding, embedding_model, embedding_dimensions,
                                embedding_input_hash, chunker_version, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                chunk_id,
                                item.document_id,
                                ordinal,
                                chunk_title,
                                piece,
                                chunk_hash,
                                vector.tobytes(),
                                self.embedder.model_name,
                                self.embedder.dimensions,
                                hashlib.sha256(
                                    _embedding_text(item, chunk).encode("utf-8")
                                ).hexdigest(),
                                CHUNKER_VERSION,
                                now,
                            ),
                        )
                        self._insert_provenance_receipt(
                            connection,
                            artifact_type="chunk",
                            artifact_id=chunk_id,
                            document_id=item.document_id,
                            source=item.source,
                            project=item.project,
                            content_hash=chunk_hash,
                            producer=(
                                "heading_chunker" if chunk.breadcrumb else "paragraph_chunker"
                            ),
                            producer_version=(
                                f"{self.settings.chunk_size}:"
                                f"{self.settings.chunk_overlap}:schema-v3"
                            ),
                            taints=_content_taints(piece),
                            metadata={"ordinal": ordinal},
                            parent_receipt_ids=[document_receipt_id],
                        )
                    item.chunk_count = len(item.pieces)
                self._supersede_orphaned_provenance(connection)
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

    def _record_deletion_manifest(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        reason: str,
    ) -> str:
        document_id = str(row["id"])
        chunk_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
        )
        distillation_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM distillations WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
        )
        deleted_at = _iso(None)
        manifest = {
            "document_id": document_id,
            "source": str(row["source"]),
            "source_key_hash": hashlib.sha256(
                str(row["source_key"]).encode("utf-8")
            ).hexdigest(),
            "content_hash": str(row["content_hash"]),
            "reason": redact_text(reason)[:100],
            "chunk_count": chunk_count,
            "distillation_count": distillation_count,
            "deleted_at": deleted_at,
        }
        manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
        manifest_id = f"del_{uuid.uuid4().hex}"
        connection.execute(
            """
            INSERT INTO deletion_manifests(
                id, document_id, source, source_key_hash, content_hash, reason,
                chunk_count, distillation_count, manifest_json, manifest_hash,
                deleted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest_id,
                document_id,
                str(row["source"]),
                manifest["source_key_hash"],
                str(row["content_hash"]),
                manifest["reason"],
                chunk_count,
                distillation_count,
                manifest_json,
                manifest_hash,
                deleted_at,
            ),
        )
        connection.execute(
            """
            UPDATE provenance_receipts
            SET superseded_at = COALESCE(superseded_at, ?)
            WHERE document_id = ? AND superseded_at IS NULL
            """,
            (deleted_at, document_id),
        )
        self._insert_provenance_receipt(
            connection,
            artifact_type="deletion_manifest",
            artifact_id=manifest_id,
            document_id=document_id,
            source=str(row["source"]),
            project=row["project"] if "project" in row.keys() else None,
            content_hash=manifest_hash,
            producer="administrative_delete",
            producer_version="schema-v3",
            taints=[],
            metadata=manifest,
        )
        return manifest_id

    def forget_memory(self, memory_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT id, source, source_key, kind, project, content_hash
                    FROM documents WHERE id = ?
                    """,
                    (memory_id,),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    return False
                if row["kind"] != "memory":
                    raise ValueError("Administrative forget only accepts an explicitly saved memory ID")
                self._record_deletion_manifest(
                    connection,
                    row,
                    reason="explicit_memory_forget",
                )
                connection.execute("DELETE FROM documents WHERE id = ?", (memory_id,))
                self._bump_vector_generation(connection)
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise

    def reconcile_source(
        self,
        source: str,
        seen_keys: Iterable[str],
        *,
        reason: str = "source_absent",
        allow_large: bool = False,
    ) -> int:
        """Remove stale derived documents only after a successful source scan.

        Guarded by a sanity floor. Several scan paths fail *open*: a truncated
        Hermes export still exits 0, and an upstream format change that drops the
        timestamp field makes every message look older than the cutoff. Either
        one yields a near-empty scan that looks indistinguishable from "the user
        deleted everything", and reconciliation would then delete the entire
        source. Refusing to reconcile an implausibly small scan turns silent
        mass deletion into a visible, recoverable failure.
        """

        safe_source = redact_text(source.strip().casefold())
        safe_keys = [(redact_text(key),) for key in set(seen_keys)]
        floor_ratio = self.settings.reconcile_min_ratio
        with self._connect() as connection:
            indexed = int(
                connection.execute(
                    "SELECT COUNT(*) FROM documents WHERE source = ? AND kind = 'derived'",
                    (safe_source,),
                ).fetchone()[0]
            )
            # The floor targets mass deletion. Applying it to a source holding a
            # handful of documents would block ordinary cleanup without
            # protecting anything worth protecting.
            if (
                not allow_large
                and indexed >= _RECONCILE_FLOOR_MIN_DOCUMENTS
                and len(safe_keys) < indexed * floor_ratio
            ):
                raise ReconcileFloorNotMet(
                    f"Refusing to reconcile {safe_source}: the scan reported "
                    f"{len(safe_keys)} keys against {indexed} indexed documents, "
                    f"below the {floor_ratio:.0%} floor. Nothing was deleted."
                )
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("CREATE TEMP TABLE IF NOT EXISTS scan_seen(source_key TEXT PRIMARY KEY)")
                connection.execute("DELETE FROM scan_seen")
                connection.executemany("INSERT OR IGNORE INTO scan_seen(source_key) VALUES (?)", safe_keys)
                stale_rows = connection.execute(
                    """
                    SELECT id, source, source_key, kind, project, content_hash
                    FROM documents
                    WHERE source = ? AND kind = 'derived'
                      AND NOT EXISTS (
                          SELECT 1 FROM scan_seen WHERE scan_seen.source_key = documents.source_key
                      )
                    ORDER BY id
                    """,
                    (safe_source,),
                ).fetchall()
                for row in stale_rows:
                    # A flat "source_reconciliation" made every deletion look
                    # alike, so an audit could not tell a genuinely removed file
                    # from one that merely grew past the size limit or stopped
                    # decoding. Record which it was.
                    self._record_deletion_manifest(
                        connection,
                        row,
                        reason=f"source_reconciliation:{reason}",
                    )
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

    @staticmethod
    def _expire_stale_refreshes(
        connection: sqlite3.Connection,
        *,
        now: str,
    ) -> int:
        # A lease is orphaned when it has outlived its expiry *or* when its
        # heartbeat has gone quiet for far longer than the beat interval. Keying
        # only on expiry meant that killing a refresh locked the database for the
        # remainder of a 30-minute lease even though the owning process was
        # provably gone seconds later. Mutual exclusion does not depend on this:
        # `runlock.ingestion_lock` is an OS file lock the kernel releases when the
        # process dies, so a second refresh still cannot start while one is truly
        # alive. The lease is bookkeeping on top of that.
        silent_before = _iso(
            datetime.now(timezone.utc) - timedelta(seconds=_LEASE_HEARTBEAT_GRACE_SECONDS)
        )
        stale = connection.execute(
            """
            SELECT r.run_id
            FROM refresh_runs r
            JOIN refresh_lease l ON l.run_id = r.run_id
            WHERE r.status = 'running'
              AND (l.expires_at <= ? OR l.heartbeat_at <= ?)
            """,
            (now, silent_before),
        ).fetchall()
        run_ids = [str(row["run_id"]) for row in stale]
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            connection.execute(
                f"""
                UPDATE refresh_runs
                SET status = 'abandoned', completed_at = ?,
                    last_error = 'refresh_lease_expired'
                WHERE run_id IN ({placeholders}) AND status = 'running'
                """,
                [now, *run_ids],
            )
            connection.execute(
                f"DELETE FROM refresh_lease WHERE run_id IN ({placeholders})",
                run_ids,
            )
        active = connection.execute(
            """
            SELECT 1 FROM refresh_lease l
            JOIN refresh_runs r ON r.run_id = l.run_id
            WHERE r.status = 'running' AND l.expires_at > ?
            LIMIT 1
            """,
            (now,),
        ).fetchone()
        orphaned_sources = 0
        if active is None:
            cursor = connection.execute(
                """
                UPDATE ingest_state
                SET status = 'abandoned', last_failure_at = ?,
                    failures = failures + 1,
                    last_error = ?
                WHERE status = 'running'
                """,
                (
                    now,
                    (
                        "refresh_lease_expired"
                        if run_ids
                        else "orphaned_running_state_without_lease"
                    ),
                ),
            )
            orphaned_sources = int(cursor.rowcount)
        return len(run_ids) or int(orphaned_sources > 0)

    def recover_stale_refreshes(self) -> int:
        now = _iso(None)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                recovered = self._expire_stale_refreshes(connection, now=now)
                connection.commit()
                return recovered
            except Exception:
                connection.rollback()
                raise

    def start_refresh_run(
        self,
        mode: str,
        *,
        lease_seconds: int = _DEFAULT_REFRESH_LEASE_SECONDS,
    ) -> RefreshLease:
        safe_mode = redact_text(mode).strip()[:40] or "incremental"
        duration = max(60, int(lease_seconds))
        now_dt = _utc_now()
        now = _iso(now_dt)
        expires_at = _iso(now_dt + timedelta(seconds=duration))
        run_id = f"run_{uuid.uuid4().hex}"
        token = secrets.token_urlsafe(32)
        token_hash = _token_hash(token)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._expire_stale_refreshes(connection, now=now)
                active = connection.execute(
                    """
                    SELECT r.run_id, r.owner_pid, r.owner_host, l.expires_at
                    FROM refresh_lease l JOIN refresh_runs r ON r.run_id = l.run_id
                    WHERE l.id = 1
                    """
                ).fetchone()
                if active is not None:
                    raise RuntimeError(
                        "Another Cerebras Memory refresh owns the database lease "
                        f"(run_id={active['run_id']}, pid={active['owner_pid']}, "
                        f"host={active['owner_host']}, expires_at={active['expires_at']})"
                    )
                connection.execute(
                    """
                    INSERT INTO refresh_runs(
                        run_id, mode, owner_pid, owner_host, lease_token_hash,
                        status, started_at, heartbeat_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)
                    """,
                    (
                        run_id,
                        safe_mode,
                        os.getpid(),
                        socket.gethostname()[:255],
                        token_hash,
                        now,
                        now,
                        expires_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO refresh_lease(
                        id, run_id, lease_token_hash, acquired_at,
                        heartbeat_at, expires_at
                    ) VALUES (1, ?, ?, ?, ?, ?)
                    """,
                    (run_id, token_hash, now, now, expires_at),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return RefreshLease(run_id=run_id, token=token, expires_at=expires_at)

    def heartbeat_refresh_run(
        self,
        lease: RefreshLease,
        *,
        current_source: str | None = None,
        lease_seconds: int = _DEFAULT_REFRESH_LEASE_SECONDS,
    ) -> str:
        now_dt = _utc_now()
        now = _iso(now_dt)
        expires_at = _iso(now_dt + timedelta(seconds=max(60, int(lease_seconds))))
        token_hash = _token_hash(lease.token)
        safe_source = redact_text(current_source)[:100] if current_source else None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE refresh_lease
                    SET heartbeat_at = ?, expires_at = ?
                    WHERE id = 1 AND run_id = ? AND lease_token_hash = ?
                    """,
                    (now, expires_at, lease.run_id, token_hash),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("Refresh database lease is no longer owned by this run")
                connection.execute(
                    """
                    UPDATE refresh_runs
                    SET heartbeat_at = ?, expires_at = ?, current_source = ?
                    WHERE run_id = ? AND lease_token_hash = ? AND status = 'running'
                    """,
                    (now, expires_at, safe_source, lease.run_id, token_hash),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return expires_at

    def finish_refresh_run(
        self,
        lease: RefreshLease,
        *,
        succeeded: bool,
        report: dict[str, object] | None = None,
        error: str | None = None,
    ) -> bool:
        now = _iso(None)
        token_hash = _token_hash(lease.token)
        status = "succeeded" if succeeded else "failed"
        safe_error = (
            redact_text(error).replace("\r", " ").replace("\n", " ")[:1000]
            if error
            else None
        )
        summary_json = json.dumps(
            _redact_json_value(report or {}),
            ensure_ascii=False,
            sort_keys=True,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE refresh_runs
                    SET status = ?, completed_at = ?, heartbeat_at = ?,
                        summary_json = ?, last_error = ?
                    WHERE run_id = ? AND lease_token_hash = ? AND status = 'running'
                    """,
                    (
                        status,
                        now,
                        now,
                        summary_json,
                        safe_error,
                        lease.run_id,
                        token_hash,
                    ),
                )
                connection.execute(
                    """
                    DELETE FROM refresh_lease
                    WHERE id = 1 AND run_id = ? AND lease_token_hash = ?
                    """,
                    (lease.run_id, token_hash),
                )
                if not succeeded:
                    connection.execute(
                        """
                        UPDATE ingest_state
                        SET status = 'failed', last_failure_at = ?,
                            failures = failures + 1,
                            last_error = COALESCE(?, 'refresh_failed')
                        WHERE status = 'running'
                        """,
                        (now, safe_error),
                    )
                connection.commit()
                return cursor.rowcount == 1
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
        # The first run builds the ~100 MB snapshot; later runs hit the cache.
        # Taking a plain median over [cold, warm, warm] silently reported the
        # warm figure while looking like it summarised all three. Separate them:
        # the warm number drives activation because a long-lived MCP worker pays
        # it on every query, while the cold number is the once-per-generation
        # cost worth watching after each refresh invalidates the snapshot.
        cold_ms = float(timings[0])
        warm_samples = timings[1:] or timings
        median_ms = float(statistics.median(warm_samples))
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE vector_index_state
                SET exact_benchmark_ms = ?, chunk_count = ?, updated_at = ?
                WHERE id = 1
                """,
                (median_ms, count, _iso(None)),
            )
        return {
            "runs_ms": [round(item, 3) for item in timings],
            "cold_ms": round(cold_ms, 3),
            "warm_median_ms": round(median_ms, 3),
            "median_ms": median_ms,
            "chunks": count,
        }

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
        # Structural limit worth stating plainly: the ANN path requires an
        # unfiltered query (`unfiltered = not filters` gates it in
        # `_vector_candidates`), and the MCP server passes client roots on every
        # call, so any correctly-scoped search resolves to a project filter and
        # takes the exact path. The sidecar can therefore only ever serve
        # unscoped global search - the case that needs it least. Exact search
        # also pre-filters before the matmul, so a scoped query already scans
        # only its own project. Keep the thresholds high and treat an active
        # sidecar as the exception, not the goal.
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
                document_row = connection.execute(
                    "SELECT source, project FROM documents WHERE id = ?",
                    (document_id,),
                ).fetchone()
                parent_rows = connection.execute(
                    """
                    SELECT id FROM provenance_receipts
                    WHERE superseded_at IS NULL AND (
                        (artifact_type = 'document' AND artifact_id = ?)
                        OR (
                            artifact_type = 'chunk' AND document_id = ?
                            AND artifact_id IN (
                                SELECT id FROM chunks
                                WHERE document_id = ? AND ordinal BETWEEN ? AND ?
                            )
                        )
                    )
                    ORDER BY artifact_type, artifact_id
                    """,
                    (
                        document_id,
                        document_id,
                        document_id,
                        unit.start_ordinal,
                        unit.end_ordinal,
                    ),
                ).fetchall()
                taints = _content_taints(
                    search_text,
                    externally_processed=(
                        self.settings.distillation.provider == "deepseek"
                    ),
                )
                taints.append("generated_summary")
                self._insert_provenance_receipt(
                    connection,
                    artifact_type="distillation",
                    artifact_id=distillation_id,
                    document_id=document_id,
                    source=(
                        str(document_row["source"]) if document_row is not None else None
                    ),
                    project=document_row["project"] if document_row is not None else None,
                    content_hash=hashlib.sha256(
                        search_text.encode("utf-8")
                    ).hexdigest(),
                    producer=f"distiller:{self.settings.distillation.model}",
                    producer_version=self.settings.distillation.prompt_version,
                    taints=taints,
                    metadata={
                        "input_hash": unit.input_hash,
                        "start_ordinal": unit.start_ordinal,
                        "end_ordinal": unit.end_ordinal,
                        "provider": self.settings.distillation.provider,
                    },
                    parent_receipt_ids=[str(row["id"]) for row in parent_rows],
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

    @staticmethod
    def _project_matches(value: str | None, patterns: Sequence[str]) -> bool:
        if not value:
            return False
        folded = value.casefold()
        return any(folded == pattern.casefold() for pattern in patterns)

    def _remote_distillation_decision(
        self,
        document: sqlite3.Row,
    ) -> tuple[bool, str]:
        settings = self.settings.distillation
        if settings.provider != "deepseek":
            return True, "local_provider"
        if not settings.remote_policy_enabled:
            return True, "policy_disabled"

        source = str(document["source"]).casefold()
        project = str(document["project"]).strip() if document["project"] else None
        if source not in set(settings.remote_allow_sources):
            return False, "source_not_allowed"
        if self._project_matches(project, settings.remote_deny_projects):
            return False, "project_denied"
        if settings.remote_allow_projects and not self._project_matches(
            project,
            settings.remote_allow_projects,
        ):
            return False, "project_not_allowlisted"
        if settings.block_unscoped_remote and not project:
            return False, "unscoped_document"

        labels = " ".join(
            str(document[name] or "")
            for name in ("project", "title", "uri", "source_key")
        ).casefold()
        for term in settings.sensitive_project_terms:
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", labels):
                return False, f"sensitive_project_term:{term}"
        return True, "policy_allowed"

    def _audit_distillation_request(
        self,
        *,
        document_id: str,
        unit: Any,
        source: str,
        project: str | None,
        decision: str,
        status: str,
        error_code: str | None = None,
    ) -> str | None:
        if not self.settings.distillation.audit_remote_requests:
            return None
        if self.settings.distillation.provider != "deepseek":
            return None
        request_id = f"out_{uuid.uuid4().hex}"
        endpoint_host = urlparse(self.settings.distillation.endpoint).hostname or "unknown"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO outbound_distillation_audit(
                    request_id, document_id, unit_input_hash, source, project,
                    provider, model, endpoint_host, character_count,
                    decision, status, error_code, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    document_id,
                    unit.input_hash,
                    source,
                    project,
                    self.settings.distillation.provider,
                    self.settings.distillation.model,
                    endpoint_host,
                    len(unit.text),
                    decision,
                    status,
                    redact_text(error_code)[:100] if error_code else None,
                    _iso(None),
                    _iso(None) if status != "pending" else None,
                ),
            )
        return request_id

    def _complete_distillation_audit(
        self,
        request_id: str | None,
        *,
        status: str,
        error_code: str | None = None,
    ) -> None:
        if not request_id:
            return
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE outbound_distillation_audit
                SET status = ?, error_code = ?, completed_at = ?
                WHERE request_id = ?
                """,
                (
                    status,
                    redact_text(error_code)[:100] if error_code else None,
                    _iso(None),
                    request_id,
                ),
            )

    def _mark_distillation_blocked(
        self,
        document_id: str,
        *,
        units: Sequence[Any],
        reason: str,
        source: str,
        project: str | None,
    ) -> dict[str, Any]:
        """Record a terminal policy block for a document and all of its units.

        The document row is terminal (``blocked``) but the unit rows used to be
        set to ``pending``, which reads as retryable work. Because the
        blocked-decision cache short-circuits these documents, those units were
        never retried and never completed: the pipeline could not converge and
        ``kb_stats`` reported outstanding work that would never finish. 612 units
        sat in that state. ``policy_blocked`` is terminal and countable.
        """

        now = _iso(None)
        safe_reason = redact_text(reason)[:200]
        for unit in units:
            self._audit_distillation_request(
                document_id=document_id,
                unit=unit,
                source=source,
                project=project,
                decision="blocked",
                status="blocked",
                error_code=safe_reason,
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO distillation_state(
                        document_id, status, model, prompt_version, units_total,
                        units_ready, failures, last_attempt_at, last_success_at, last_error
                    ) VALUES (?, 'blocked', ?, ?, ?, 0, 0, ?, NULL, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        status = 'blocked', model = excluded.model,
                        prompt_version = excluded.prompt_version,
                        units_total = excluded.units_total, units_ready = 0,
                        failures = 0, last_attempt_at = excluded.last_attempt_at,
                        last_success_at = NULL, last_error = excluded.last_error
                    """,
                    (
                        document_id,
                        self.settings.distillation.model,
                        self.settings.distillation.prompt_version,
                        len(units),
                        now,
                        safe_reason,
                    ),
                )
                connection.execute(
                    """
                    UPDATE distillation_unit_state
                    SET status = 'policy_blocked', last_error = ?
                    WHERE document_id = ?
                    """,
                    (safe_reason, document_id),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {
            "document_id": document_id,
            "status": "blocked",
            "reason": safe_reason,
            "units": len(units),
            "ready": 0,
            "generated": 0,
            "failures": 0,
        }

    def _finalize_distillation_document(
        self,
        document_id: str,
        *,
        units: Sequence[Any],
        generated: Sequence[tuple[Any, dict[str, Any], str, np.ndarray, str]],
        ready_ids: set[str],
        unit_outcomes: Sequence[tuple[Any, str, str | None, int]],
        failures: Sequence[str],
    ) -> str:
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
                        f"""
                        DELETE FROM distillations
                        WHERE document_id = ? AND id NOT IN ({placeholders})
                        """,
                        [document_id, *sorted(ready_ids)],
                    )
                else:
                    connection.execute(
                        "DELETE FROM distillations WHERE document_id = ?",
                        (document_id,),
                    )
                connection.execute(
                    """
                    UPDATE provenance_receipts
                    SET superseded_at = ?
                    WHERE artifact_type = 'distillation'
                      AND document_id = ? AND superseded_at IS NULL
                      AND NOT EXISTS (
                          SELECT 1 FROM distillations x
                          WHERE x.id = provenance_receipts.artifact_id
                      )
                    """,
                    (now, document_id),
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
                status = (
                    "ready"
                    if ready == total and not failures
                    else ("partial" if ready else "failed")
                )
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
                return status
            except Exception:
                connection.rollback()
                raise

    def distill_document(
        self,
        document_id: str,
        *,
        force: bool = False,
        force_input_hashes: set[str] | None = None,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            document = connection.execute(
                """
                SELECT source, source_key, title, uri, project, metadata_json
                FROM documents WHERE id = ? AND kind = 'derived'
                """,
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
            existing_state = connection.execute(
                """
                SELECT status, model, prompt_version, units_total, units_ready,
                       last_error
                FROM distillation_state WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
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
        allowed, policy_reason = self._remote_distillation_decision(document)
        if not allowed:
            if (
                existing_state is not None
                and existing_state["status"] == "blocked"
                and existing_state["model"] == self.settings.distillation.model
                and existing_state["prompt_version"]
                == self.settings.distillation.prompt_version
                and existing_state["last_error"] == policy_reason
                and int(existing_state["units_total"]) == len(units)
            ):
                return {
                    "document_id": document_id,
                    "status": "blocked",
                    "reason": policy_reason,
                    "units": len(units),
                    "ready": 0,
                    "generated": 0,
                    "failures": 0,
                    "cached": True,
                }
            return self._mark_distillation_blocked(
                document_id,
                units=units,
                reason=policy_reason,
                source=str(document["source"]),
                project=str(document["project"]) if document["project"] else None,
            )

        existing = {
            row["input_hash"]: row
            for row in existing_rows
            if row["embedding_model"] == self.embedder.model_name
            and int(row["embedding_dimensions"]) == self.embedder.dimensions
        }
        if (
            not force
            and not force_input_hashes
            and existing_state is not None
            and existing_state["status"] == "ready"
            and existing_state["model"] == self.settings.distillation.model
            and existing_state["prompt_version"]
            == self.settings.distillation.prompt_version
            and int(existing_state["units_total"]) == len(units)
            and int(existing_state["units_ready"]) == len(units)
            and all(unit.input_hash in existing for unit in units)
        ):
            return {
                "document_id": document_id,
                "status": "ready",
                "units": len(units),
                "ready": len(units),
                "generated": 0,
                "failures": 0,
                "cached": True,
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

        def call_distiller(unit: Any) -> dict[str, Any]:
            request_id = self._audit_distillation_request(
                document_id=document_id,
                unit=unit,
                source=str(document["source"]),
                project=str(document["project"]) if document["project"] else None,
                decision="allowed",
                status="pending",
            )
            try:
                with self._distillation_request_slots:
                    result = self.distiller.distill(unit.text)
                self._complete_distillation_audit(request_id, status="succeeded")
                return result
            except Exception as exc:
                self._complete_distillation_audit(
                    request_id,
                    status="failed",
                    error_code=type(exc).__name__,
                )
                raise

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
                    checkpoint_generated(unit, call_distiller(unit))
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
                    executor.submit(call_distiller, unit): unit
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

        status = self._finalize_distillation_document(
            document_id,
            units=units,
            generated=generated,
            ready_ids=ready_ids,
            unit_outcomes=unit_outcomes,
            failures=failures,
        )
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
            "blocked": sum(report["status"] == "blocked" for report in reports),
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
            blocked_units = int(
                connection.execute(
                    """
                    SELECT COALESCE(SUM(units_total), 0)
                    FROM distillation_state WHERE status = 'blocked'
                    """
                ).fetchone()[0]
            )
            nonblocked_gap = int(
                connection.execute(
                    """
                    SELECT COALESCE(SUM(units_total - units_ready), 0)
                    FROM distillation_state WHERE status <> 'blocked'
                    """
                ).fetchone()[0]
            )
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
            retryable_pending = max(
                0,
                unit_states.get("pending", 0) - blocked_units,
            ) + unit_states.get("failed", 0)
            effective_unit_states = dict(unit_states)
            if blocked_units:
                effective_unit_states["blocked"] = blocked_units
                adjusted_pending = max(
                    0,
                    effective_unit_states.get("pending", 0) - blocked_units,
                )
                if adjusted_pending:
                    effective_unit_states["pending"] = adjusted_pending
                else:
                    effective_unit_states.pop("pending", None)
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
            outbound_status = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM outbound_distillation_audit GROUP BY status
                    """
                )
            }
            outbound_decisions = {
                str(row["decision"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT decision, COUNT(*) AS count
                    FROM outbound_distillation_audit GROUP BY decision
                    """
                )
            }
            last_outbound = connection.execute(
                """
                SELECT request_id, document_id, source, project, provider, model,
                       endpoint_host, character_count, decision, status,
                       error_code, created_at, completed_at
                FROM outbound_distillation_audit
                ORDER BY audit_pk DESC LIMIT 1
                """
            ).fetchone()
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
            "units_pending": max(retryable_pending, nonblocked_gap),
            "units_blocked": blocked_units,
            "unit_failures": unit_states.get("failed", int(totals["failures"])),
            "unit_states": effective_unit_states,
            "pilot_documents": pilot_documents,
            "pilot_units": pilot_units,
            "pilot_states": pilot_states,
            "states": states,
            "remote_policy": {
                "enabled": self.settings.distillation.remote_policy_enabled,
                "allow_sources": list(self.settings.distillation.remote_allow_sources),
                "allow_projects": list(self.settings.distillation.remote_allow_projects),
                "deny_projects": list(self.settings.distillation.remote_deny_projects),
                "sensitive_project_terms": list(
                    self.settings.distillation.sensitive_project_terms
                ),
                "block_unscoped": self.settings.distillation.block_unscoped_remote,
                "audit_requests": self.settings.distillation.audit_remote_requests,
            },
            "outbound_audit": {
                "status_counts": outbound_status,
                "decision_counts": outbound_decisions,
                "last_request": dict(last_outbound) if last_outbound else None,
            },
            "last_error": failure["last_error"] if failure else None,
        }

    @staticmethod
    def _label_relevance(case: Mapping[str, Any]) -> dict[str, float]:
        """Map ``document_id -> gain`` for a labelled evaluation case."""

        relevant = case.get("relevant")
        if isinstance(relevant, list) and relevant:
            graded: dict[str, float] = {}
            for item in relevant:
                if isinstance(item, Mapping):
                    document_id = str(item.get("document_id") or "").strip()
                    gain = float(item.get("gain", 1.0))
                else:
                    document_id, gain = str(item).strip(), 1.0
                if document_id and gain > 0:
                    graded[document_id] = gain
            if graded:
                return graded
        expected = case.get("expected_document_id")
        return {str(expected): 1.0} if expected else {}

    def _held_out_distillation_quality(self, label_path: Path) -> dict[str, Any]:
        """Measure the distillation channel against human-written queries.

        The self-retrieval check below asks whether a distillation can find its
        own source document using a query generated *from* that document.  That
        is a traceability check, not a quality measure: a model that copies
        distinctive strings out of the dialogue scores near-perfectly by
        construction.  Only held-out human labels can say whether the channel
        helps a real question.
        """

        if not label_path.exists():
            return {"available": False, "reason": "label_set_missing"}
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            cases = payload.get("cases")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {"available": False, "reason": redact_text(str(exc))[:200]}
        if not isinstance(cases, list) or not cases:
            return {"available": False, "reason": "label_set_empty"}

        def rank(results: Sequence[Mapping[str, Any]], relevance: Mapping[str, float]) -> int | None:
            return next(
                (
                    index
                    for index, item in enumerate(results, start=1)
                    if str(item["document_id"]) in relevance
                ),
                None,
            )

        without: list[int | None] = []
        with_channel: list[int | None] = []
        for case in cases:
            if not isinstance(case, dict):
                continue
            relevance = self._label_relevance(case)
            query = str(case.get("query") or "").strip()
            if not relevance or not query:
                continue
            sources = [str(case["source"])] if case.get("source") else None
            # Reranking is held off on both sides so the comparison isolates the
            # retrieval channel rather than cross-encoder noise.
            without.append(
                rank(
                    self.search_response(
                        query, limit=8, sources=sources, global_search=True,
                        rerank=False, include_distillations=False,
                    )["results"],
                    relevance,
                )
            )
            with_channel.append(
                rank(
                    self.search_response(
                        query, limit=8, sources=sources, global_search=True,
                        rerank=False, include_distillations=True,
                    )["results"],
                    relevance,
                )
            )

        def metrics(ranks: Sequence[int | None]) -> dict[str, float]:
            count = len(ranks)
            if not count:
                return {"recall_at_8": 0.0, "mrr_at_8": 0.0}
            return {
                "recall_at_8": round(
                    sum(item is not None and item <= 8 for item in ranks) / count, 6
                ),
                "mrr_at_8": round(
                    sum((1.0 / item) if item else 0.0 for item in ranks) / count, 6
                ),
            }

        if not without:
            return {"available": False, "reason": "no_usable_labelled_cases"}
        off = metrics(without)
        on = metrics(with_channel)
        return {
            "available": True,
            "label_set": str(label_path.resolve()),
            "cases": len(without),
            "distillations_off": off,
            "distillations_on": on,
            "recall_delta": round(on["recall_at_8"] - off["recall_at_8"], 6),
            "mrr_delta": round(on["mrr_at_8"] - off["mrr_at_8"], 6),
            "not_harmful": bool(
                on["recall_at_8"] >= off["recall_at_8"] and on["mrr_at_8"] >= off["mrr_at_8"]
            ),
        }

    def evaluate_distillations(
        self,
        *,
        limit: int = 24,
        label_path: Path | None = None,
    ) -> dict[str, Any]:
        limit = min(max(1, int(limit)), 100)
        if label_path is None:
            label_path = self.settings.canary_suite_path.parent / "search-quality-baseline.json"
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
        held_out = self._held_out_distillation_quality(label_path)
        # The gate is driven by held-out human labels. The self-retrieval numbers
        # are reported for traceability but cannot pass the gate on their own:
        # their queries are generated from the documents they must retrieve, so a
        # high score there is guaranteed by construction rather than earned.
        automated_gate = bool(
            cases
            and schema_valid
            and secret_free
            and held_out.get("available")
            and held_out.get("not_harmful")
        )
        return {
            "cases": len(cases),
            "schema_valid": schema_valid,
            "secret_free": secret_free,
            "self_retrieval": {
                "note": (
                    "Traceability only: queries are derived from the documents "
                    "they must retrieve, so these numbers are not a quality measure."
                ),
                "baseline": baseline_metrics,
                "augmented": augmented_metrics,
                "mrr_relative_improvement": round(improvement, 6),
            },
            "held_out_quality": held_out,
            "automated_gate_passed": automated_gate,
            "manual_traceability_audit_required": True,
            "promotion_ready": False,
            "case_results": case_reports,
        }

    @staticmethod
    def _fts_query(query: str, *, rarity: Mapping[str, int] | None = None) -> str | None:
        """Build the FTS5 MATCH expression for a query.

        Two changes over a plain OR of every token. Content-free words are
        dropped, because BM25 fixes their *ranking* contribution but not their
        consumption of the fixed candidate budget: a document matching only
        "what"/"did"/"about" can still crowd a genuinely relevant one out of the
        top 50 before scoring ever happens. And when the term cap bites, terms
        are dropped by falling corpus rarity, so the cap sheds common words
        rather than truncating whatever happened to be typed last.
        """

        tokens = _query_tokens(query)
        if not tokens:
            return None
        # Keep stopwords only if the query is nothing but stopwords.
        content = [token for token in tokens if token not in _STOPWORDS] or tokens
        if len(content) > _MAX_FTS_TERMS:
            if rarity:
                # Rarer terms first; unknown terms are treated as maximally rare
                # because they cannot have inflated the candidate set.
                content.sort(key=lambda token: rarity.get(token, 0))
            content = content[:_MAX_FTS_TERMS]
        return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in content)

    def _fts_query_for(
        self,
        connection: sqlite3.Connection,
        query: str,
    ) -> str | None:
        """``_fts_query`` with corpus rarity supplied only when the cap bites."""

        tokens = _query_tokens(query)
        content = [token for token in tokens if token not in _STOPWORDS] or tokens
        rarity = (
            self._term_document_frequency(connection, content)
            if len(content) > _MAX_FTS_TERMS
            else None
        )
        return self._fts_query(query, rarity=rarity)

    def _term_document_frequency(
        self,
        connection: sqlite3.Connection,
        terms: Sequence[str],
    ) -> dict[str, int]:
        """Corpus document frequency per term, via FTS5's vocabulary table.

        ``fts5vocab`` is created in the temp schema so this needs no migration
        and no extra stored state.
        """

        if not terms:
            return {}
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS temp.chunks_vocab "
                "USING fts5vocab(main, chunks_fts, 'row')"
            )
            placeholders = ",".join("?" for _ in terms)
            rows = connection.execute(
                f"SELECT term, doc FROM temp.chunks_vocab WHERE term IN ({placeholders})",
                tuple(terms),
            ).fetchall()
        except sqlite3.Error:
            # Rarity ordering is an optimisation; losing it must not fail search.
            return {}
        return {str(row["term"]): int(row["doc"]) for row in rows}

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
    def _snippet(
        content: str,
        query: str,
        length: int = 700,
        *,
        terms: Sequence[str] | None = None,
    ) -> str:
        """Return the densest query-relevant window of ``content``.

        The previous implementation anchored on ``min(content.find(term))``:
        the *earliest* occurrence of *any* term, matched as a raw substring.
        Short tokens such as ``i``, ``a``, ``do`` or ``the`` occur inside some
        word within the first few characters of nearly any chunk, so the window
        collapsed to the head of the chunk for 82-100% of real chunks. Because
        this text is what the cross-encoder scores, the most expensive stage of
        retrieval was usually reading the wrong part of the document.

        Selection is now word-boundary based, ignores content-free tokens, and
        scores candidate windows by how many *distinct* terms they cover.
        """

        if len(content) <= length:
            return content
        selected = [
            term
            for term in (terms if terms is not None else _query_tokens(query))
            if term not in _STOPWORDS and len(term) > 1
        ]
        if not selected:
            # Nothing content-bearing to centre on; the head is as good as any
            # other window, but say so explicitly rather than by accident.
            return content[:length].strip() + "…"

        wanted = set(selected)
        lowered = content.casefold()
        hits = [
            (match.start(), match.group(0))
            for match in _WORD_RE.finditer(lowered)
            if match.group(0) in wanted
        ]
        if not hits:
            return content[:length].strip() + "…"

        # Slide a window anchored on each hit and keep the one covering the most
        # distinct terms, breaking ties on total hits and then earliest position.
        best_left, best_right = 0, 1
        best_key = (-1, 1)
        right = 0
        for left, (position, _term) in enumerate(hits):
            while right < len(hits) and hits[right][0] < position + length:
                right += 1
            distinct = len({term for _pos, term in hits[left:right]})
            last_position, last_term = hits[right - 1]
            span = (last_position + len(last_term)) - position
            # Most distinct terms wins; ties go to the tightest span. Preferring
            # more *hits* instead would favour a window padded with repeats of
            # one term, pushing the rarer terms past the window edge.
            key = (distinct, -span)
            if key > best_key:
                best_key = key
                best_left, best_right = left, right

        # Centre on the matched *span*, not on its left edge: backing off from
        # the first hit would push the window left and drop the later terms that
        # made this window the best one.
        span_start = hits[best_left][0]
        last_position, last_term = hits[best_right - 1]
        span_end = last_position + len(last_term)
        centre = (span_start + span_end) // 2
        start = max(0, centre - length // 2)
        start = min(start, max(0, len(content) - length))
        end = min(len(content), start + length)
        prefix = "…" if start else ""
        suffix = "…" if end < len(content) else ""
        return f"{prefix}{content[start:end].strip()}{suffix}"

    @staticmethod
    def _rerank_passage(content: str, query: str, length: int) -> str:
        """Build the passage text handed to the cross-encoder.

        This deliberately does *not* use the query-centred window from
        ``_snippet``, despite that window being objectively better at containing
        the query terms (99.5% of eligible chunks versus 67.8%). Measured on the
        24-case gold set, switching this call site to the centred window moved
        MRR@8 from 0.7604 to 0.6386 and nDCG@10 from 0.8184 to 0.7291. A pure
        head-of-chunk passage was worse still at 0.5980.

        The reranker scores one anchor per *document*, so what it needs is a
        document-identity signal, and the opening region of a chunk carries the
        section heading that the chunk text itself never records.

        The obvious counter-hypothesis has been tested and rejected: once heading
        breadcrumbs were added to the embedded text, the centred window was
        re-measured on the same gold set and still lost badly - MRR@8 0.6577
        against 0.7865 for this head-biased passage, with recall@8 dropping from
        1.0 to 0.9583. Breadcrumbs did not rescue it. Do not switch this call
        site to ``_snippet`` without new evidence; two independent measurements
        now say it is worse.
        """

        if len(content) <= length:
            return content
        terms = _query_tokens(query)
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
        fts_query = self._fts_query_for(connection, safe_query)
        lexical_rank: dict[int, int] = {}
        vector_rank: dict[int, int] = {}
        rows_by_pk: dict[int, sqlite3.Row] = {}
        # Only these columns are read downstream.  Selecting ``x.*`` would drag
        # the embedding blob, summary_json and search_text of every candidate
        # through the query for nothing.
        unit_columns = (
            "x.distillation_pk, x.id, x.document_id, x.start_ordinal, x.end_ordinal"
        )
        pilot_join = (
            "JOIN distillation_pilot_documents p ON p.document_id = x.document_id"
            if self.settings.distillation.mode == "pilot"
            else ""
        )
        if fts_query:
            rows = connection.execute(
                f"""
                SELECT {unit_columns}, bm25(distillations_fts) AS lexical_score
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
        # This scan is unbounded -- it visits every ready distillation matching
        # the current model.  Stream only the key and the vector, and retain
        # nothing beyond the bounded heap: materialising each visited row here
        # made peak memory scale with corpus size on every search.
        vector_rows = connection.execute(
            f"""
            SELECT x.distillation_pk, x.embedding FROM distillations x
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
        for rank, (_, key) in enumerate(sorted(heap, reverse=True), start=1):
            vector_rank[key] = rank

        # Fetch the surviving vector winners' metadata once, in bounded batches.
        missing = sorted(set(vector_rank) - set(rows_by_pk))
        for start in range(0, len(missing), _SQL_PARAM_BATCH):
            window = missing[start : start + _SQL_PARAM_BATCH]
            placeholders = ",".join("?" for _ in window)
            for row in connection.execute(
                f"""
                SELECT {unit_columns} FROM distillations x
                WHERE x.distillation_pk IN ({placeholders})
                """,
                window,
            ):
                rows_by_pk[int(row["distillation_pk"])] = row

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
            # A concurrent refresh can delete a unit between the scan above and
            # the metadata fetch, so tolerate a missing row rather than failing
            # the whole search.
            row = rows_by_pk.get(key)
            if row is None:
                continue
            chosen_by_document.setdefault(str(row["document_id"]), row)
            if len(chosen_by_document) >= limit:
                break

        anchors: list[int] = []
        matched: dict[int, str] = {}
        expected_bytes = self.embedder.dimensions * 4
        # One indexed lookup per chosen document.  Folding these into a single
        # statement with OR'd (document_id, ordinal) clauses was measured and
        # rejected: SQLite abandons idx_chunks_document for idx_chunks_model,
        # scanning every chunk of the active model and sorting the result --
        # 277ms against 3.1ms for the fifty separate seeks it replaced.
        for row in chosen_by_document.values():
            candidates = [
                candidate
                for candidate in connection.execute(
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
                )
                if len(candidate["embedding"]) == expected_bytes
            ]
            if not candidates:
                continue
            # One matmul over the range instead of re-running frombuffer and dot
            # inside a comparator for every pairwise comparison.
            matrix = np.frombuffer(
                b"".join(candidate["embedding"] for candidate in candidates),
                dtype=np.float32,
            ).reshape(len(candidates), self.embedder.dimensions)
            best = candidates[int(np.argmax(matrix @ query_vector))]
            chunk_key = int(best["chunk_pk"])
            anchors.append(chunk_key)
            matched[chunk_key] = str(row["id"])
        return anchors, matched

    def _active_provenance(
        self,
        artifact_ids: Iterable[str],
    ) -> dict[str, dict[str, Any]]:
        identifiers = sorted({str(value) for value in artifact_ids if value})
        if not identifiers:
            return {}
        placeholders = ",".join("?" for _ in identifiers)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, artifact_id, artifact_type, taints_json, producer,
                       producer_version, created_at
                FROM provenance_receipts
                WHERE superseded_at IS NULL
                  AND artifact_id IN ({placeholders})
                ORDER BY created_at, id
                """,
                identifiers,
            ).fetchall()
        return {
            str(row["artifact_id"]): {
                "receipt_id": str(row["id"]),
                "artifact_type": str(row["artifact_type"]),
                "producer": str(row["producer"]),
                "producer_version": str(row["producer_version"]),
                "taints": json.loads(row["taints_json"] or "[]"),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        }

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
            fts_query = self._fts_query_for(connection, safe_query)
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

            # Up to two candidate_limit-sized rank sets land here.  The
            # configured cap keeps this well inside SQLITE_LIMIT_VARIABLE_NUMBER
            # (32,766), so batching is defence in depth rather than a live fix:
            # it keeps the bound local to this query instead of resting on a
            # clamp in config.py, matching _reusable_embeddings.
            missing_keys = sorted(
                (set(vector_rank) | set(distillation_rank)) - set(records)
            )
            for start in range(0, len(missing_keys), _SQL_PARAM_BATCH):
                window = missing_keys[start : start + _SQL_PARAM_BATCH]
                placeholders = ",".join("?" for _ in window)
                rows = connection.execute(
                    f"""
                    SELECT {detail_columns} FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.chunk_pk IN ({placeholders})
                    """,
                    window,
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
                    # Returned evidence recomputes its own snippets below, once
                    # the winning variant is known; computing them here for every
                    # candidate variant was pure waste on the latency-sensitive
                    # path (~80 discarded densest-window scans per query).
                    #
                    # Widening this budget was measured and rejected: raising it
                    # to 1400 characters cost roughly 2x warm p95 (1400ms ->
                    # 2700ms) without improving MRR@8 or nDCG@10 over the 700
                    # baseline. The cross-encoder's 512-token ceiling means the
                    # extra text is largely truncated anyway.
                    rerank_budget = max(100, _RERANK_PASSAGE_BUDGET // len(context_rows))
                    rerank_snippets = [
                        self._rerank_passage(
                            str(row["content"]), safe_query, rerank_budget
                        )
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
        top_variants = ordered[:limit]
        artifact_ids = {
            str(row["chunk_id"])
            for variant, _ in top_variants
            for row in variant["rows"]
        }
        artifact_ids.update(
            str(variant["document_id"]) for variant, _ in top_variants
        )
        artifact_ids.update(
            str(distillation_match[int(variant["anchor"]["chunk_pk"])])
            for variant, _ in top_variants
            if int(variant["anchor"]["chunk_pk"]) in distillation_match
            and distillation_match[int(variant["anchor"]["chunk_pk"])]
        )
        receipt_by_artifact = self._active_provenance(artifact_ids)
        results: list[dict[str, Any]] = []
        for variant, rerank_score in top_variants:
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
                    "provenance": receipt_by_artifact.get(str(row["chunk_id"])),
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
            distillation_id = distillation_match.get(anchor_key)
            anchor_receipt = receipt_by_artifact.get(str(anchor["chunk_id"]))
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
                    "distillation_id": distillation_id,
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
                    "taints": (
                        list(anchor_receipt.get("taints", []))
                        if anchor_receipt
                        else ["untrusted_evidence"]
                    ),
                    "provenance": {
                        "document": receipt_by_artifact.get(
                            str(anchor["document_id"])
                        ),
                        "anchor_chunk": anchor_receipt,
                        "distillation": (
                            receipt_by_artifact.get(str(distillation_id))
                            if distillation_id
                            else None
                        ),
                    },
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
        provenance = self._active_provenance(
            [document_id, *[str(row["id"]) for row in chunks]]
        )
        document_receipt = provenance.get(document_id)
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
                "taints": (
                    list(document_receipt.get("taints", []))
                    if document_receipt
                    else ["untrusted_evidence"]
                ),
                "provenance": document_receipt,
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
                    "taints": list(
                        provenance.get(str(row["id"]), {}).get(
                            "taints",
                            ["untrusted_evidence"],
                        )
                    ),
                    "provenance": provenance.get(str(row["id"])),
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

    def record_canary_result(self, result: dict[str, Any]) -> str:
        run_id = f"canary_{uuid.uuid4().hex}"
        status = "passed" if bool(result.get("gate_passed")) else "failed"
        safe_report = {
            "suite_path": redact_text(str(result.get("suite_path") or "")),
            "suite_version": str(result.get("suite_version") or ""),
            "suite_hash": str(result.get("suite_hash") or ""),
            "started_at": str(result.get("started_at") or ""),
            "completed_at": str(result.get("completed_at") or ""),
            "cases_total": int(result.get("cases_total") or 0),
            "cases_passed": int(result.get("cases_passed") or 0),
            "p95_latency_ms": float(result.get("p95_latency_ms") or 0.0),
            "gate_passed": bool(result.get("gate_passed")),
            "results": result.get("results") if isinstance(result.get("results"), list) else [],
        }
        report_json = json.dumps(
            _redact_json_value(safe_report),
            ensure_ascii=False,
            sort_keys=True,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO canary_runs(
                        run_id, suite_version, suite_hash, status, started_at,
                        completed_at, cases_total, cases_passed, p95_latency_ms,
                        report_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        safe_report["suite_version"],
                        safe_report["suite_hash"],
                        status,
                        safe_report["started_at"] or _iso(None),
                        safe_report["completed_at"] or _iso(None),
                        safe_report["cases_total"],
                        safe_report["cases_passed"],
                        safe_report["p95_latency_ms"],
                        report_json,
                    ),
                )
                for case in safe_report["results"]:
                    if not isinstance(case, dict):
                        continue
                    case_id = redact_text(str(case.get("case_id") or ""))[:200]
                    if not case_id:
                        continue
                    connection.execute(
                        """
                        INSERT INTO canary_case_results(
                            run_id, case_id, passed, latency_ms, result_json
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            case_id,
                            int(bool(case.get("passed"))),
                            float(case.get("latency_ms") or 0.0),
                            json.dumps(
                                _redact_json_value(case),
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        ),
                    )
                connection.execute(
                    """
                    UPDATE quality_gate_state
                    SET status = ?, suite_version = ?, last_run_id = ?,
                        updated_at = ?, detail_json = ?
                    WHERE id = 1
                    """,
                    (
                        status,
                        safe_report["suite_version"],
                        run_id,
                        _iso(None),
                        json.dumps(
                            {
                                "cases_total": safe_report["cases_total"],
                                "cases_passed": safe_report["cases_passed"],
                                "p95_latency_ms": safe_report["p95_latency_ms"],
                            },
                            sort_keys=True,
                        ),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return run_id

    def canary_status(self) -> dict[str, Any]:
        with self._connect() as connection:
            gate = connection.execute(
                "SELECT * FROM quality_gate_state WHERE id = 1"
            ).fetchone()
            latest = connection.execute(
                """
                SELECT run_id, suite_version, suite_hash, status, started_at,
                       completed_at, cases_total, cases_passed, p95_latency_ms
                FROM canary_runs ORDER BY started_at DESC LIMIT 1
                """
            ).fetchone()
        return {
            "suite_path": str(self.settings.canary_suite_path),
            "suite_exists": self.settings.canary_suite_path.exists(),
            "run_after_refresh": self.settings.canary_run_after_refresh,
            "latency_threshold_ms": self.settings.canary_latency_threshold_ms,
            "gate": dict(gate) if gate is not None else None,
            "latest": dict(latest) if latest is not None else None,
        }

    def stats(self, *, recover: bool = True) -> dict[str, Any]:
        # Recovery writes. Callers that advertise themselves as read-only must
        # pass recover=False; the refresh path performs recovery explicitly.
        recovered_refreshes = self.recover_stale_refreshes() if recover else 0
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
            active_refresh = connection.execute(
                """
                SELECT r.run_id, r.mode, r.owner_pid, r.owner_host, r.status,
                       r.current_source, r.started_at, r.heartbeat_at, r.expires_at
                FROM refresh_lease l JOIN refresh_runs r ON r.run_id = l.run_id
                WHERE l.id = 1 AND r.status = 'running' AND l.expires_at > ?
                """,
                (_iso(None),),
            ).fetchone()
            latest_run = connection.execute(
                """
                SELECT run_id, mode, owner_pid, owner_host, status, current_source,
                       started_at, heartbeat_at, expires_at, completed_at, last_error
                FROM refresh_runs ORDER BY started_at DESC LIMIT 1
                """
            ).fetchone()
            provenance_counts = {
                str(row["artifact_type"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT artifact_type, COUNT(*) AS count
                    FROM provenance_receipts
                    WHERE superseded_at IS NULL
                    GROUP BY artifact_type
                    """
                )
            }
            deletion_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM deletion_manifests"
                ).fetchone()[0]
            )
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
        canary_status = self.canary_status()
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
            "quality_gate": canary_status,
            "last_refresh": latest_refresh,
            "refresh_in_progress": active_refresh is not None,
            "refresh": {
                "active": dict(active_refresh) if active_refresh is not None else None,
                "latest": dict(latest_run) if latest_run is not None else None,
                "stale_runs_recovered": recovered_refreshes,
            },
            "provenance": {
                "active_receipts": provenance_counts,
                "deletion_manifests": deletion_count,
            },
            "sources": source_state,
            "failures": {
                source: state["last_error"]
                for source, state in source_state.items()
                if state["status"] == "failed"
            },
        }


# Original scaffold imported ``Store``. Keep the name while exposing the new API.
Store = KnowledgeStore
