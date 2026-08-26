"""A source that cannot see all of its roots must not reconcile.

``reconcile_source`` treats a key absent from a *successful* scan as a deletion.
Availability used to mean "at least one declared root existed", so a source whose
second root had gone away -- a stopped sync, a machine that is off, an unmounted
drive -- still reported success with fewer keys, and the missing machine's
documents were deleted along with their chunks, distillations and provenance
receipts.  The ``reconcile_min_ratio`` floor only trips once more than half the
source is gone, so a minority root disappeared silently.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from importers.agent_history import _jsonl_files, scan_claude
from ingest import _run_ingestion_body

RECENT = datetime.now(timezone.utc) - timedelta(days=1)


def _session(root: Path, name: str) -> Path:
    """Write one minimal Claude transcript with a unique session id."""

    stamp = RECENT.isoformat().replace("+00:00", "Z")
    lines = [
        {
            "type": "user",
            "sessionId": name,
            "timestamp": stamp,
            "cwd": "C:/work/alpha",
            "message": {"role": "user", "content": f"question from {name}"},
        },
        {
            "type": "assistant",
            "sessionId": name,
            "timestamp": stamp,
            "message": {"role": "assistant", "content": f"answer from {name}"},
        },
    ]
    path = root / f"{name}.jsonl"
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    return path


def _documents(store) -> int:
    with store._connect() as connection:
        return int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])


def test_jsonl_files_names_the_root_that_is_missing(tmp_path):
    present = tmp_path / "present"
    present.mkdir()
    (present / "a.jsonl").write_text("{}\n", encoding="utf-8")
    absent = tmp_path / "absent"

    files, unavailable = _jsonl_files([present, absent])

    assert files == []
    assert unavailable is not None
    assert str(absent) in unavailable


def test_jsonl_files_is_satisfied_when_every_root_is_present(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.jsonl").write_text("{}\n", encoding="utf-8")

    files, unavailable = _jsonl_files([first, second])

    assert unavailable is None
    assert [path.name for path in files] == ["a.jsonl"]


def test_an_unconfigured_source_is_not_silently_healthy(tmp_path):
    files, unavailable = _jsonl_files([])

    assert files == []
    assert unavailable == "no history root is configured"


def test_a_scanner_fails_loudly_when_one_root_is_missing(settings_factory, tmp_path):
    present = tmp_path / "present"
    present.mkdir()
    _session(present, "kept")
    settings = settings_factory(claude_roots=(present, tmp_path / "gone"))

    result = scan_claude(settings, RECENT - timedelta(days=30))

    assert result.successful is False
    assert "history root unavailable" in (result.error or "")
    assert str(tmp_path / "gone") in (result.error or "")


def test_a_missing_root_deletes_nothing(settings_factory, store, tmp_path):
    """The regression this whole guard exists for."""

    hub = tmp_path / "hub"
    spoke = tmp_path / "spoke"
    hub.mkdir()
    spoke.mkdir()
    for index in range(4):
        _session(hub, f"hub-{index}")
    _session(spoke, "spoke-0")

    settings = settings_factory(
        claude_roots=(hub, spoke), enabled_sources=frozenset({"claude"})
    )
    store.settings = settings

    first = _run_ingestion_body(settings, store=store)
    assert first["sources"]["claude"]["status"] == "ok"
    seeded = _documents(store)
    assert seeded == 5

    # The spoke goes away: sync stopped, or the machine is simply off.
    for path in spoke.iterdir():
        path.unlink()
    spoke.rmdir()

    second = _run_ingestion_body(settings, store=store)

    assert second["sources"]["claude"]["status"] == "failed"
    assert second["ok"] is False
    assert _documents(store) == seeded, "a missing root must never delete documents"


def test_reconciliation_really_does_delete_when_a_scan_succeeds(
    settings_factory, store, tmp_path
):
    """Contrast case: the guard is not protecting a no-op.

    With the root still present but emptied, the scan legitimately succeeds with
    fewer keys and the document is removed -- a 1-in-5 loss, far under the
    ``reconcile_min_ratio`` floor, so nothing else would have caught it.
    """

    hub = tmp_path / "hub"
    spoke = tmp_path / "spoke"
    hub.mkdir()
    spoke.mkdir()
    for index in range(4):
        _session(hub, f"hub-{index}")
    _session(spoke, "spoke-0")

    settings = settings_factory(
        claude_roots=(hub, spoke), enabled_sources=frozenset({"claude"})
    )
    store.settings = settings

    _run_ingestion_body(settings, store=store)
    assert _documents(store) == 5

    for path in spoke.iterdir():
        path.unlink()  # root stays, so the scan still succeeds

    report = _run_ingestion_body(settings, store=store)

    assert report["sources"]["claude"]["status"] == "ok"
    assert _documents(store) == 4
