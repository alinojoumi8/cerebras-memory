"""Administrative ingestion CLI. Ingestion is intentionally not an MCP tool."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import threading
from typing import Callable

from chunking import chunk_text
from config import Settings, load_settings
from importers import scan_claude, scan_codex, scan_grok, scan_hermes, scan_projects
from models import ScanResult
from redaction import redact_text
from runlock import IngestionAlreadyRunning, ingestion_lock
from stdio import configure_utf8_stdio
from store import KnowledgeStore, RefreshLease


Scanner = Callable[[Settings, datetime], ScanResult]
_REFRESH_HEARTBEAT_SECONDS = 60
_REFRESH_LEASE_SECONDS = 30 * 60


class _RefreshHeartbeat:
    def __init__(self, store: KnowledgeStore, lease: RefreshLease):
        self.store = store
        self.lease = lease
        self._source: str | None = None
        self._source_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="cerebras-memory-refresh-heartbeat",
            daemon=True,
        )
        self.last_error: str | None = None

    def start(self) -> None:
        self._thread.start()

    def set_source(self, source: str | None) -> None:
        with self._source_lock:
            self._source = source
        try:
            self.store.heartbeat_refresh_run(
                self.lease,
                current_source=source,
                lease_seconds=_REFRESH_LEASE_SECONDS,
            )
            self.last_error = None
        except Exception as exc:
            self.last_error = _safe_error(f"Refresh heartbeat failed: {exc}")

    def _run(self) -> None:
        while not self._stop.wait(_REFRESH_HEARTBEAT_SECONDS):
            with self._source_lock:
                source = self._source
            try:
                self.store.heartbeat_refresh_run(
                    self.lease,
                    current_source=source,
                    lease_seconds=_REFRESH_LEASE_SECONDS,
                )
                self.last_error = None
            except Exception as exc:
                self.last_error = _safe_error(f"Refresh heartbeat failed: {exc}")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _scanners(settings: Settings) -> list[tuple[str, Scanner]]:
    available: list[tuple[str, Scanner]] = [
        ("hermes", scan_hermes),
        ("claude", scan_claude),
        ("codex", scan_codex),
        ("grok", scan_grok),
        ("projects", scan_projects),
    ]
    return [item for item in available if item[0] in settings.enabled_sources]


def _safe_error(value: str) -> str:
    return redact_text(value).replace("\r", " ").replace("\n", " ")[:1000]


def _run_ingestion_body(
    settings: Settings,
    *,
    dry_run: bool = False,
    force: bool = False,
    store: KnowledgeStore | None = None,
    heartbeat: _RefreshHeartbeat | None = None,
) -> dict[str, object]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.transcript_days)
    if dry_run:
        store = None
    elif store is None:
        store = KnowledgeStore(settings)
    report: dict[str, object] = {
        "mode": "dry-run" if dry_run else ("full" if force else "incremental"),
        "cutoff": cutoff.isoformat().replace("+00:00", "Z"),
        "sources": {},
        "ok": True,
    }
    source_reports: dict[str, object] = report["sources"]  # type: ignore[assignment]

    for source, scanner in _scanners(settings):
        if heartbeat:
            heartbeat.set_source(source)
        if store:
            store.record_ingest_start(source)
        try:
            scan = scanner(settings, cutoff)
        except Exception as exc:  # one source must never prevent later sources
            scan = ScanResult(source=source, successful=False, error=f"Unhandled scan failure: {exc}")

        summary: dict[str, object] = {
            "scanned": scan.scanned,
            "documents": len(scan.documents),
            "skipped": scan.skipped,
            "malformed": scan.malformed,
            "watermark": scan.watermark,
            "status": "ok" if scan.successful else "failed",
        }
        source_reports[source] = summary
        if not scan.successful:
            error = _safe_error(scan.error or "Unknown scan failure")
            summary["error"] = error
            report["ok"] = False
            if store:
                store.record_ingest_failure(source, error)
            continue

        if dry_run:
            # Exercise redaction and chunking without creating a database or
            # downloading the model. No source content is printed.
            summary["chunks_estimated"] = sum(
                len(
                    chunk_text(
                        redact_text(document.text),
                        settings.chunk_size,
                        settings.chunk_overlap,
                    )
                )
                for document in scan.documents
            )
            continue

        assert store is not None
        imported = 0
        unchanged = 0
        chunks = 0
        write_failed = False
        try:
            for result in store.upsert_documents(scan.documents, force=force):
                chunks += result.chunks
                if result.status == "unchanged":
                    unchanged += 1
                else:
                    imported += 1
        except Exception as exc:
            write_failed = True
            error = _safe_error(f"Index write failed: {exc}")
            summary["status"] = "failed"
            summary["error"] = error
            report["ok"] = False
            store.record_ingest_failure(source, error)

        if write_failed:
            # Crucially, do not reconcile after any incomplete write pass.
            continue
        deleted = store.reconcile_source(source, scan.seen_keys)
        summary.update(
            {
                "imported": imported,
                "unchanged": unchanged,
                "chunks": chunks,
                "reconciled_deletions": deleted,
            }
        )
        store.record_ingest_success(
            source,
            watermark=scan.watermark,
            scanned=scan.scanned,
            imported=imported,
            skipped=scan.skipped + scan.malformed,
        )

        # Distillation is derived, retryable state. It runs only after the raw
        # source is safely committed and never changes the refresh exit code.
        if (
            source in {"hermes", "claude", "codex", "grok"}
            and settings.distillation.mode == "on"
        ):
            try:
                summary["distillation"] = store.distill_documents(source=source)
            except Exception as exc:
                summary["distillation"] = {
                    "status": "failed",
                    "error": _safe_error(f"Derived distillation failed: {exc}"),
                }

    # The sidecar index and benchmark are also derived state. Build only after
    # a completely successful refresh; exact search remains available if this
    # maintenance fails or the activation threshold has not been reached.
    if store is not None and report["ok"]:
        try:
            report["vector_maintenance"] = store.maintain_vector_index()
        except Exception as exc:
            report["vector_maintenance"] = {
                "status": "failed",
                "error": _safe_error(f"Vector maintenance failed: {exc}"),
            }
        if settings.canary_run_after_refresh:
            if not settings.canary_suite_path.exists():
                report["quality_gate"] = {
                    "status": "skipped",
                    "reason": "canary_suite_missing",
                }
            else:
                try:
                    from quality import evaluate_canary_suite

                    canary = evaluate_canary_suite(
                        store,
                        settings.canary_suite_path,
                        record=True,
                    )
                    report["quality_gate"] = {
                        "status": "passed" if canary["gate_passed"] else "failed",
                        **canary,
                    }
                except Exception as exc:
                    report["quality_gate"] = {
                        "status": "failed",
                        "error": _safe_error(f"Canary evaluation failed: {exc}"),
                    }
    return report


def run_ingestion(
    settings: Settings,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, object]:
    if dry_run:
        return _run_ingestion_body(settings, dry_run=True, force=force)

    store = KnowledgeStore(settings)
    mode = "full" if force else "incremental"
    lease = store.start_refresh_run(mode, lease_seconds=_REFRESH_LEASE_SECONDS)
    heartbeat = _RefreshHeartbeat(store, lease)
    heartbeat.start()
    report: dict[str, object] | None = None
    failure: str | None = None
    try:
        report = _run_ingestion_body(
            settings,
            dry_run=False,
            force=force,
            store=store,
            heartbeat=heartbeat,
        )
        report["run_id"] = lease.run_id
        return report
    except Exception as exc:
        failure = _safe_error(str(exc))
        raise
    finally:
        heartbeat.stop()
        if report is not None and heartbeat.last_error:
            report["refresh_heartbeat_warning"] = heartbeat.last_error
        try:
            finalized = store.finish_refresh_run(
                lease,
                succeeded=bool(report and report.get("ok")) and failure is None,
                report=report,
                error=failure,
            )
            if report is not None and not finalized:
                report["ok"] = False
                report["status"] = "refresh_lease_lost"
        except Exception as exc:
            if report is not None:
                report["ok"] = False
                report["status"] = "refresh_finalize_failed"
                report["refresh_finalize_error"] = _safe_error(str(exc))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh the private local Cerebras Memory index")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="scan/redact/chunk without writing")
    mode.add_argument("--full", action="store_true", help="force replacement and re-embedding")
    mode.add_argument("--incremental", action="store_true", help="skip unchanged documents (default)")
    parser.add_argument("--config", type=Path, help="path to config JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    lock_path = settings.database_path.parent / "ingest.lock"
    try:
        with ingestion_lock(lock_path):
            report = run_ingestion(settings, dry_run=args.dry_run, force=args.full)
    except IngestionAlreadyRunning as exc:
        report = {"ok": False, "status": "overlap_prevented", "error": str(exc)}
    except Exception as exc:
        report = {"ok": False, "status": "failed", "error": _safe_error(str(exc))}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
