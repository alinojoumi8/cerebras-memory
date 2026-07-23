"""Allowlisted project-document scanner."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import stat

from config import Settings
from models import IngestDocument, ScanResult


_ALLOWED_EXTENSIONS = {".md", ".mdx", ".txt"}
_EXCLUDED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "bower_components",
    "vendor",
    "dist",
    "build",
    "target",
    "out",
    "coverage",
    ".coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".cache",
    "cache",
    "tmp",
    "temp",
    "logs",
    "log",
    "graphify-out",
    ".next",
    ".nuxt",
    ".turbo",
    ".parcel-cache",
    "generated",
    "secrets",
    ".secrets",
}
_SECRET_NAME_PARTS = {
    ".env",
    "credential",
    "credentials",
    "secret",
    "secrets",
    "token",
    "tokens",
    "password",
    "passwd",
    "id_rsa",
    "id_ed25519",
    "service-account",
    "service_account",
    "auth.json",
}


def _looks_secret(path: Path) -> bool:
    name = path.name.casefold()
    stem = path.stem.casefold()
    return any(part == name or part == stem or part in name for part in _SECRET_NAME_PARTS)


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _safe_read(path: Path, maximum: int) -> str | None:
    try:
        size = path.stat().st_size
        if size <= 0 or size > maximum:
            return None
        data = path.read_bytes()
    except OSError:
        raise
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def scan_projects(settings: Settings, _cutoff: datetime | None = None) -> ScanResult:
    result = ScanResult(source="projects")
    root = settings.projects_root
    if not root.exists() or not root.is_dir():
        result.successful = False
        result.error = "Project documentation root is unavailable"
        return result

    documents: list[IngestDocument] = []
    try:
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            current = Path(directory)
            kept: list[str] = []
            for name in directory_names:
                candidate = current / name
                if name.casefold() in _EXCLUDED_DIRECTORIES or _is_reparse_point(candidate):
                    result.skipped += 1
                else:
                    kept.append(name)
            directory_names[:] = kept

            for name in file_names:
                path = current / name
                result.scanned += 1
                if path.suffix.casefold() not in _ALLOWED_EXTENSIONS:
                    result.skipped += 1
                    continue
                if _looks_secret(path) or _is_reparse_point(path):
                    result.skipped += 1
                    continue
                text = _safe_read(path, settings.max_file_bytes)
                if text is None or not text.strip():
                    result.skipped += 1
                    continue
                relative = path.relative_to(root)
                project = relative.parts[0] if len(relative.parts) > 1 else root.name
                timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                documents.append(
                    IngestDocument(
                        source="projects",
                        source_key=relative.as_posix().casefold(),
                        title=relative.as_posix(),
                        text=text,
                        timestamp=timestamp,
                        project=project,
                        uri=str(path.resolve()),
                        metadata={
                            "relative_path": relative.as_posix(),
                            "extension": path.suffix.casefold(),
                            "size_bytes": path.stat().st_size,
                        },
                    )
                )
    except (OSError, UnicodeError) as exc:
        result.successful = False
        result.error = f"Project documentation scan failed: {exc}"
        return result
    result.documents = documents
    if documents:
        result.watermark = max(doc.timestamp for doc in documents).isoformat().replace("+00:00", "Z")
    return result
