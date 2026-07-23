"""Defensive secret redaction applied before content reaches SQLite."""

from __future__ import annotations

import re


REDACTED = "[REDACTED]"

_PRIVATE_KEY = re.compile(
    r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----.*?-----END(?: [A-Z0-9]+)? PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)

_TOKEN_PATTERNS = (
    # Well-known provider token shapes.
    re.compile(r"\bsk-(?:proj-|live-)?[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:eyJ[A-Za-z0-9_-]{8,}\.){2}[A-Za-z0-9_-]{8,}\b"),
    # Authorization headers.
    re.compile(
        r"(?im)(\bauthorization\s*:\s*(?:bearer|basic)\s+)"
        r"(?!\[REDACTED\])[^\s,;]+"
    ),
    # Credential assignments in source, shell, JSON, YAML, and prose.
    re.compile(
        r"(?im)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
        r"password|passwd|pwd|secret|private[_-]?key)\b\s*(?:=|:))\s*"
        r"(?:['\"]?)(?!\[REDACTED\])[^\s,'\"}\]]{4,}(?:['\"]?)"
    ),
    re.compile(
        r"(?im)(\b(?:password|passwd|pwd)\s+is\s+)"
        r"(?!\[REDACTED\])[^\s,;]+"
    ),
)


def redact_text(value: str) -> str:
    """Return text with likely credentials replaced by a fixed marker.

    Redaction is intentionally conservative: false positives are preferable to
    persisting a live credential in a long-lived shared index.
    """

    text = _PRIVATE_KEY.sub(REDACTED, value)
    for pattern in _TOKEN_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda match: f"{match.group(1)}{REDACTED}", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def contains_private_key(value: str) -> bool:
    return bool(_PRIVATE_KEY.search(value))
