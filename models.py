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
    retained_keys: set[str] = field(default_factory=set)

    @property
    def seen_keys(self) -> set[str]:
        """Keys that must survive reconciliation.

        Existence and freshness are separate questions. Reconciliation exists to
        delete documents whose source genuinely disappeared. Deriving this set
        from ``documents`` alone conflated the two: a session whose messages had
        simply aged out of the rolling window produced no document, dropped out
        of this set, and had its document, chunks and distillations deleted -
        even though the session was still sitting on disk and nothing would ever
        rebuild what was removed.

        ``retained_keys`` carries sources that were seen but produced no fresh
        document, so their existing rows are left alone.
        """

        return {document.source_key for document in self.documents} | set(self.retained_keys)
