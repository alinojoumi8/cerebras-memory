"""Configuration for the local Cerebras Memory service.

Configuration is deliberately small and file based.  Every path is resolved to
an absolute path before it is used so the MCP server behaves identically no
matter which client launched it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
_ALLOWED_SECRET_ENV_KEYS = frozenset({"DEEPSEEK_API_KEY", "NVIDIA_API_KEY"})


@dataclass(frozen=True)
class RerankerSettings:
    enabled: bool
    model: str
    cache_dir: Path
    candidate_documents: int
    max_length: int
    batch_size: int
    intra_op_threads: int


@dataclass(frozen=True)
class VectorSearchSettings:
    backend: str
    index_dir: Path
    ann_min_chunks: int
    ann_latency_threshold_ms: float
    connectivity: int
    expansion_add: int
    expansion_search: int
    dtype: str


@dataclass(frozen=True)
class DistillationSettings:
    mode: str
    provider: str
    endpoint: str
    model: str
    api_key_env: str
    max_output_tokens: int
    max_concurrent_requests: int
    prompt_version: str
    timeout_seconds: float
    min_messages: int
    min_characters: int
    max_messages_per_unit: int
    max_characters_per_unit: int
    gap_minutes: int
    pilot_per_source: int
    remote_policy_enabled: bool
    remote_allow_sources: tuple[str, ...]
    remote_allow_projects: tuple[str, ...]
    remote_deny_projects: tuple[str, ...]
    sensitive_project_terms: tuple[str, ...]
    block_unscoped_remote: bool
    audit_remote_requests: bool


@dataclass(frozen=True)
class Settings:
    project_root: Path
    config_path: Path
    database_path: Path
    model_cache_dir: Path
    embedding_model: str
    embedding_dimensions: int
    projects_root: Path
    transcript_days: int
    chunk_size: int
    chunk_overlap: int
    candidate_limit: int
    rrf_k: int
    max_file_bytes: int
    enabled_sources: frozenset[str]
    claude_roots: tuple[Path, ...]
    codex_roots: tuple[Path, ...]
    grok_roots: tuple[Path, ...]
    hermes_command: str
    reranker: RerankerSettings
    vector_search: VectorSearchSettings
    distillation: DistillationSettings
    canary_suite_path: Path
    canary_run_after_refresh: bool
    canary_latency_threshold_ms: float

    def ensure_runtime_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.reranker.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_search.index_dir.mkdir(parents=True, exist_ok=True)


def _expand_path(value: str | Path, *, base: Path) -> Path:
    text = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _deep_update(target: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _load_secret_env(path: Path = DEFAULT_ENV_PATH) -> None:
    """Load only explicitly allowed API keys without overriding the process env."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or key not in _ALLOWED_SECRET_ENV_KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if value:
            os.environ.setdefault(key, value)


def _defaults() -> dict[str, Any]:
    user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
    codex_home = Path(os.environ.get("CODEX_HOME", user_profile / ".codex"))
    return {
        "database_path": "data/cerebras_memory.sqlite3",
        "model_cache_dir": "data/models",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimensions": 384,
        "projects_root": str(user_profile / "Documents" / "myprojects"),
        "transcript_days": 90,
        "chunk_size": 1800,
        "chunk_overlap": 200,
        "candidate_limit": 50,
        "rrf_k": 60,
        "max_file_bytes": 1_048_576,
        "sources": {
            "hermes": True,
            "claude": True,
            "codex": True,
            "grok": True,
            "projects": True,
        },
        "agent_roots": {
            "claude": [str(user_profile / ".claude" / "projects")],
            "codex": [
                str(codex_home / "sessions"),
                str(codex_home / "archived_sessions"),
            ],
            "grok": [
                str(user_profile / ".grok" / "sessions"),
                str(user_profile / ".grok" / "projects"),
            ],
        },
        "hermes_command": "hermes",
        "reranker": {
            "enabled": True,
            "model": "ms-marco-MiniLM-L-12-v2",
            "cache_dir": "data/models/reranker",
            "candidate_documents": 20,
            "max_length": 512,
            "batch_size": 8,
            "intra_op_threads": 24,
        },
        "vector_search": {
            "backend": "auto",
            "index_dir": "data/vector-index",
            "ann_min_chunks": 100_000,
            "ann_latency_threshold_ms": 750.0,
            "connectivity": 16,
            "expansion_add": 128,
            "expansion_search": 64,
            "dtype": "f16",
        },
        "distillation": {
            "mode": "pilot",
            "provider": "ollama",
            "endpoint": "http://127.0.0.1:11434",
            "model": "qwen3.5:4b",
            "api_key_env": "",
            "max_output_tokens": 1_536,
            "max_concurrent_requests": 1,
            "prompt_version": "agent-dialogue-v2",
            "timeout_seconds": 120.0,
            "min_messages": 8,
            "min_characters": 12_000,
            "max_messages_per_unit": 12,
            "max_characters_per_unit": 24_000,
            "gap_minutes": 30,
            "pilot_per_source": 10,
            "remote_policy": {
                "enabled": True,
                "allow_sources": ["hermes", "claude", "codex", "grok"],
                "allow_projects": [],
                "deny_projects": [],
                "sensitive_project_terms": [
                    "legal",
                    "case",
                    "defence",
                    "defense",
                    "court",
                    "osc",
                    "disclosure",
                    "privileged",
                    "counsel",
                ],
                "block_unscoped": False,
                "audit_requests": True,
            },
        },
        "quality": {
            "canary_suite": "evaluation/canary-suite.json",
            "run_after_refresh": True,
            "latency_threshold_ms": 1_500.0,
        },
    }


def load_settings(path: str | Path | None = None) -> Settings:
    _load_secret_env()
    configured_path = path or os.environ.get("CEREBRAS_MEMORY_CONFIG") or DEFAULT_CONFIG_PATH
    config_path = _expand_path(configured_path, base=PROJECT_ROOT)
    raw = _defaults()
    if config_path.exists():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration must be a JSON object: {config_path}")
        _deep_update(raw, loaded)

    # Environment overrides are intentionally narrow so a client cannot switch
    # the service to a remote backend through ambient configuration.
    if os.environ.get("CEREBRAS_MEMORY_DB"):
        raw["database_path"] = os.environ["CEREBRAS_MEMORY_DB"]
    if os.environ.get("CEREBRAS_MEMORY_MODEL_CACHE"):
        raw["model_cache_dir"] = os.environ["CEREBRAS_MEMORY_MODEL_CACHE"]

    source_flags = raw.get("sources", {})
    enabled_sources = frozenset(
        name for name in ("hermes", "claude", "codex", "grok", "projects")
        if bool(source_flags.get(name, False))
    )
    roots = raw.get("agent_roots", {})
    reranker = raw.get("reranker", {})
    vector_search = raw.get("vector_search", {})
    distillation = raw.get("distillation", {})
    quality = raw.get("quality", {})
    if not isinstance(quality, dict):
        raise ValueError("quality must be an object")
    remote_policy = distillation.get("remote_policy", {})
    if not isinstance(remote_policy, dict):
        raise ValueError("distillation.remote_policy must be an object")
    vector_backend = str(vector_search.get("backend", "auto")).casefold()
    if vector_backend not in {"auto", "exact", "hnsw"}:
        raise ValueError("vector_search.backend must be auto, exact, or hnsw")
    distillation_mode = str(distillation.get("mode", "pilot")).casefold()
    if distillation_mode not in {"off", "pilot", "on"}:
        raise ValueError("distillation.mode must be off, pilot, or on")
    distillation_provider = str(distillation.get("provider", "ollama")).casefold()
    if distillation_provider not in {"ollama", "deepseek"}:
        raise ValueError("distillation.provider must be ollama or deepseek")
    endpoint = str(distillation.get("endpoint", "http://127.0.0.1:11434")).rstrip("/")
    if distillation_provider == "ollama":
        if endpoint not in {"http://127.0.0.1:11434", "http://localhost:11434"}:
            raise ValueError("Ollama distillation must use the local loopback endpoint")
    elif endpoint != "https://api.deepseek.com/beta":
        raise ValueError("DeepSeek strict distillation must use https://api.deepseek.com/beta")
    api_key_env = str(distillation.get("api_key_env", "")).strip()
    if distillation_provider == "deepseek" and api_key_env != "DEEPSEEK_API_KEY":
        raise ValueError("DeepSeek distillation must use DEEPSEEK_API_KEY")

    return Settings(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        database_path=_expand_path(raw["database_path"], base=PROJECT_ROOT),
        model_cache_dir=_expand_path(raw["model_cache_dir"], base=PROJECT_ROOT),
        embedding_model=str(raw["embedding_model"]),
        embedding_dimensions=int(raw["embedding_dimensions"]),
        projects_root=_expand_path(raw["projects_root"], base=PROJECT_ROOT),
        transcript_days=max(1, int(raw["transcript_days"])),
        chunk_size=max(400, int(raw["chunk_size"])),
        chunk_overlap=max(0, int(raw["chunk_overlap"])),
        candidate_limit=max(10, int(raw["candidate_limit"])),
        rrf_k=max(1, int(raw["rrf_k"])),
        max_file_bytes=max(1024, int(raw["max_file_bytes"])),
        enabled_sources=enabled_sources,
        claude_roots=tuple(_expand_path(p, base=PROJECT_ROOT) for p in roots.get("claude", [])),
        codex_roots=tuple(_expand_path(p, base=PROJECT_ROOT) for p in roots.get("codex", [])),
        grok_roots=tuple(_expand_path(p, base=PROJECT_ROOT) for p in roots.get("grok", [])),
        hermes_command=str(raw.get("hermes_command", "hermes")),
        reranker=RerankerSettings(
            enabled=bool(reranker.get("enabled", True)),
            model=str(reranker.get("model", "ms-marco-MiniLM-L-12-v2")),
            cache_dir=_expand_path(reranker.get("cache_dir", "data/models/reranker"), base=PROJECT_ROOT),
            candidate_documents=max(1, min(20, int(reranker.get("candidate_documents", 20)))),
            max_length=max(64, min(512, int(reranker.get("max_length", 512)))),
            batch_size=max(1, min(64, int(reranker.get("batch_size", 8)))),
            intra_op_threads=max(
                1,
                min(24, int(reranker.get("intra_op_threads", 24))),
            ),
        ),
        vector_search=VectorSearchSettings(
            backend=vector_backend,
            index_dir=_expand_path(vector_search.get("index_dir", "data/vector-index"), base=PROJECT_ROOT),
            ann_min_chunks=max(1_000, int(vector_search.get("ann_min_chunks", 100_000))),
            ann_latency_threshold_ms=max(
                1.0, float(vector_search.get("ann_latency_threshold_ms", 750.0))
            ),
            connectivity=max(2, int(vector_search.get("connectivity", 16))),
            expansion_add=max(8, int(vector_search.get("expansion_add", 128))),
            expansion_search=max(8, int(vector_search.get("expansion_search", 64))),
            dtype=str(vector_search.get("dtype", "f16")),
        ),
        distillation=DistillationSettings(
            mode=distillation_mode,
            provider=distillation_provider,
            endpoint=endpoint,
            model=str(distillation.get("model", "qwen3.5:4b")),
            api_key_env=api_key_env,
            max_output_tokens=max(256, int(distillation.get("max_output_tokens", 1_536))),
            max_concurrent_requests=max(
                1, min(16, int(distillation.get("max_concurrent_requests", 1)))
            ),
            prompt_version=str(distillation.get("prompt_version", "agent-dialogue-v2")),
            timeout_seconds=max(1.0, float(distillation.get("timeout_seconds", 120.0))),
            min_messages=max(1, int(distillation.get("min_messages", 8))),
            min_characters=max(1_000, int(distillation.get("min_characters", 12_000))),
            max_messages_per_unit=max(
                1, int(distillation.get("max_messages_per_unit", 12))
            ),
            max_characters_per_unit=max(
                1_000, int(distillation.get("max_characters_per_unit", 24_000))
            ),
            gap_minutes=max(1, int(distillation.get("gap_minutes", 30))),
            pilot_per_source=max(1, int(distillation.get("pilot_per_source", 10))),
            remote_policy_enabled=bool(remote_policy.get("enabled", True)),
            remote_allow_sources=tuple(
                str(value).strip().casefold()
                for value in remote_policy.get(
                    "allow_sources",
                    ["hermes", "claude", "codex", "grok"],
                )
                if str(value).strip()
            ),
            remote_allow_projects=tuple(
                str(value).strip()
                for value in remote_policy.get("allow_projects", [])
                if str(value).strip()
            ),
            remote_deny_projects=tuple(
                str(value).strip()
                for value in remote_policy.get("deny_projects", [])
                if str(value).strip()
            ),
            sensitive_project_terms=tuple(
                str(value).strip().casefold()
                for value in remote_policy.get(
                    "sensitive_project_terms",
                    [
                        "legal",
                        "case",
                        "defence",
                        "defense",
                        "court",
                        "osc",
                        "disclosure",
                        "privileged",
                        "counsel",
                    ],
                )
                if str(value).strip()
            ),
            block_unscoped_remote=bool(remote_policy.get("block_unscoped", False)),
            audit_remote_requests=bool(remote_policy.get("audit_requests", True)),
        ),
        canary_suite_path=_expand_path(
            quality.get("canary_suite", "evaluation/canary-suite.json"),
            base=PROJECT_ROOT,
        ),
        canary_run_after_refresh=bool(quality.get("run_after_refresh", True)),
        canary_latency_threshold_ms=max(
            1.0,
            float(quality.get("latency_threshold_ms", 1_500.0)),
        ),
    )


# Backward-compatible name for anyone importing the original scaffold helper.
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    settings = load_settings(path)
    return {
        "database_path": str(settings.database_path),
        "model_cache_dir": str(settings.model_cache_dir),
        "embedding_model": settings.embedding_model,
        "projects_root": str(settings.projects_root),
    }
