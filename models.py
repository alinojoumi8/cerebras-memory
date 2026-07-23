"""Shared ingestion data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class IngestDocument:
    source: str
    source_key: str
    title: str
    text: str
    timestamp: datetime
    project: str | None = None
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str = "derived"

    def normalized_timestamp(self) -> datetime:
        if self.timestamp.tzinfo is None:
            return self.timestamp.replace(tzinfo=timezone.utc)
        return self.timestamp.astimezone(timezone.utc)


@dataclass
class ScanResult:
    source: str
    documents: list[IngestDocument] = field(default_factory=list)
    scanned: int = 0
    skipped: int = 0
    malformed: int = 0
    successful: bool = True
    error: str | None = None
    watermark: str | None = None

    @property
    def seen_keys(self) -> set[str]:
        return {document.source_key for document in self.documents}
