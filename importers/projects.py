"""Allowlisted project-document scanner."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
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
    # Simulator run artifacts. A single worldsimulator study wrote ~9,000
    # per-agent copies of the same skill library under .runtime, which grew
    # the corpus 2.8x and made 68% of all documents near-duplicate noise.
    ".runtime",
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

# Whole filename words that mark a credential store. "service-account.md" splits
# into {"service", "account"}, so both halves are listed.
_SECRET_WORDS = {
    "credential",
    "credentials",
    "secret",
    "secrets",
    "passwd",
    "pwd",
    "id_rsa",
    "id_ed25519",
    "rsa",
    "ed25519",
}


def _looks_secret(path: Path) -> bool:
    """Whether a filename looks like a credential store rather than a document.

    Matching is on whole name components, not bare substrings. ``part in name``
    excluded ordinary documentation with no warning: ``tokenizer-notes.md``,
    ``secrets-management.md``, ``password-policy.md`` and anything containing
    ``.env`` were all silently dropped, counted only as ``skipped``.
    """

    name = path.name.casefold()
    stem = path.stem.casefold()
    if name in _SECRET_NAME_PARTS or stem in _SECRET_NAME_PARTS:
        return True
    # Split on the separators that delimit words in filenames, so
    # "service-account.md" still matches while "tokenizer-notes.md" does not.
    components = {piece for piece in re.split(r"[._\-\s]+", stem) if piece}
    if components & _SECRET_WORDS:
        return True
    # Dotfile-style credential stores such as ".env" or ".env.local".
    return name.startswith(".env")


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
                relative = path.relative_to(root)
                source_key = relative.as_posix().casefold()
                if _looks_secret(path) or _is_reparse_point(path):
                    result.skipped += 1
                    continue
                try:
                    text = _safe_read(path, settings.max_file_bytes)
                except OSError:
                    # One locked or permission-denied file used to fail the whole
                    # scan, leaving every project document unrefreshed. Skip the
                    # file, retain its key so nothing is deleted, and carry on.
                    result.skipped += 1
                    result.retained_keys.add(source_key)
                    continue
                if text is None or not text.strip():
                    # The file still exists; it just grew past max_file_bytes, is
                    # no longer valid UTF-8, or gained a NUL byte. Deleting an
                    # already-indexed document for that would discard the last
                    # good version in favour of nothing.
                    result.skipped += 1
                    result.retained_keys.add(source_key)
                    continue
                project = relative.parts[0] if len(relative.parts) > 1 else root.name
                timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                documents.append(
                    IngestDocument(
                        source="projects",
                        source_key=source_key,
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
