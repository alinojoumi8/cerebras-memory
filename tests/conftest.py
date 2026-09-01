from __future__ import annotations

from pathlib import Path

import pytest

from config import DistillationSettings, RerankerSettings, Settings, VectorSearchSettings
from embeddings import HashingEmbedder
from store import KnowledgeStore


@pytest.fixture
def settings_factory(tmp_path: Path):
    def create(**overrides) -> Settings:
        values = {
            "project_root": Path(__file__).resolve().parents[1],
            "config_path": tmp_path / "config.json",
            "database_path": tmp_path / "data" / "memory.sqlite3",
            "model_cache_dir": tmp_path / "data" / "models",
            "embedding_model": "test/hash-v1",
            "embedding_dimensions": 32,
            "embedding_query_prefix": "",
            "projects_root": tmp_path / "projects",
            "transcript_days": 90,
            "chunk_size": 400,
            "chunk_overlap": 50,
            "candidate_limit": 50,
            "rrf_k": 60,
            "max_file_bytes": 1_048_576,
            "reconcile_min_ratio": 0.5,
            "embedding_threads": 2,
            "ingest_embedding_threads": 2,
            "ingest_batch_documents": 250,
            "enabled_sources": frozenset({"projects"}),
            "claude_roots": (tmp_path / "claude",),
            "codex_roots": (tmp_path / "codex",),
            "grok_roots": (tmp_path / "grok",),
            "hermes_command": "hermes",
            "reranker": RerankerSettings(
                enabled=False,
                model="test/reranker",
                cache_dir=tmp_path / "data" / "models" / "reranker",
                candidate_documents=20,
                max_length=128,
                batch_size=8,
                intra_op_threads=2,
            ),
            "vector_search": VectorSearchSettings(
                backend="exact",
                index_dir=tmp_path / "data" / "vector-index",
                ann_min_chunks=100_000,
                ann_latency_threshold_ms=750.0,
                connectivity=16,
                expansion_add=128,
                expansion_search=64,
                dtype="f16",
            ),
            "distillation": DistillationSettings(
                mode="off",
                provider="ollama",
                endpoint="http://127.0.0.1:11434",
                model="test/distiller",
                api_key_env="",
                max_output_tokens=1_536,
                max_concurrent_requests=1,
                prompt_version="test-v1",
                timeout_seconds=1.0,
                min_messages=8,
                min_characters=12_000,
                max_messages_per_unit=12,
                max_characters_per_unit=24_000,
                gap_minutes=30,
                pilot_per_source=10,
                remote_policy_enabled=True,
                remote_allow_sources=("hermes", "claude", "codex", "grok"),
                remote_allow_projects=(),
                remote_deny_projects=(),
                sensitive_project_terms=(
                    "legal",
                    "case",
                    "defence",
                    "defense",
                    "court",
                    "osc",
                    "disclosure",
                    "privileged",
                    "counsel",
                ),
                block_unscoped_remote=False,
                audit_remote_requests=True,
            ),
            "canary_suite_path": tmp_path / "evaluation" / "canary-suite.json",
            "canary_run_after_refresh": False,
            "canary_latency_threshold_ms": 1_500.0,
        }
        values.update(overrides)
        return Settings(**values)

    return create


@pytest.fixture
def store(settings_factory) -> KnowledgeStore:
    settings = settings_factory()
    return KnowledgeStore(settings, HashingEmbedder(dimensions=settings.embedding_dimensions))
