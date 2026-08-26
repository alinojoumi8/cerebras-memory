"""Ingesting transcripts synced here from another machine.

``stable_document_id`` hashes the source key, so two machines whose scanners
fall back to a filename or a positional index produce the same id and silently
overwrite each other on upsert. Remote roots are therefore host-qualified --
and this machine's roots deliberately are not, because re-keying the documents
already stored would re-embed every chunk, orphan every distillation, and have
reconciliation delete the originals.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from config import _expand_agent_roots
from importers.agent_history import _jsonl_files, _session_key, scan_claude
from store import stable_document_id

RECENT = datetime.now(timezone.utc) - timedelta(days=1)
CUTOFF = RECENT - timedelta(days=30)


def _session(root: Path, filename: str, *, session_id: str | None = None) -> Path:
    """Write a Claude transcript, optionally without a session id.

    Omitting the id is the interesting case: the scanner then falls back to the
    filename, which is exactly what collides between machines.
    """

    stamp = RECENT.isoformat().replace("+00:00", "Z")
    user: dict = {
        "type": "user",
        "timestamp": stamp,
        "cwd": "C:/work/shared",
        "message": {"role": "user", "content": f"question in {filename}"},
    }
    assistant: dict = {
        "type": "assistant",
        "timestamp": stamp,
        "message": {"role": "assistant", "content": f"answer in {filename}"},
    }
    if session_id is not None:
        user["sessionId"] = session_id
        assistant["sessionId"] = session_id
    path = root / filename
    path.write_text(
        json.dumps(user) + chr(10) + json.dumps(assistant) + chr(10), encoding="utf-8"
    )
    return path


def test_this_machine_keeps_the_unqualified_key_space():
    """The no-migration guarantee, stated as an assertion."""

    assert _session_key("", "abc123") == "session:abc123"
    assert stable_document_id("claude", _session_key("", "abc123")) == stable_document_id(
        "claude", "session:abc123"
    )


def test_another_machine_gets_its_own_key_space():
    assert _session_key("laptop", "abc123") == "session:laptop:abc123"
    assert stable_document_id("claude", _session_key("laptop", "abc123")) != (
        stable_document_id("claude", "session:abc123")
    )


def test_agent_roots_accept_a_bare_path_or_a_machine_entry(tmp_path):
    paths, hosts = _expand_agent_roots(
        [str(tmp_path / "local"), {"host": "laptop", "path": str(tmp_path / "synced")}]
    )

    assert paths == (
        (tmp_path / "local").resolve(),
        (tmp_path / "synced").resolve(),
    )
    # Only the other machine is labelled.
    assert hosts == {str((tmp_path / "synced").resolve()): "laptop"}


def test_a_host_label_may_not_contain_the_key_separator(tmp_path):
    """Source keys are colon separated, so a colon in a host could collide."""

    with pytest.raises(ValueError, match="colon"):
        _expand_agent_roots([{"host": "laptop:2", "path": str(tmp_path)}])


def test_an_agent_root_object_must_name_a_path(tmp_path):
    with pytest.raises(ValueError, match="must have a path"):
        _expand_agent_roots([{"host": "laptop"}])


def test_files_carry_the_machine_that_owns_their_root(tmp_path):
    local = tmp_path / "local"
    synced = tmp_path / "synced"
    local.mkdir()
    synced.mkdir()
    _session(local, "here.jsonl")
    _session(synced, "there.jsonl")

    files, unavailable = _jsonl_files(
        [local, synced], hosts={str(synced): "laptop"}
    )

    assert unavailable is None
    assert {path.name: host for path, host in files} == {
        "here.jsonl": "",
        "there.jsonl": "laptop",
    }


def test_identically_named_sessions_from_two_machines_stay_distinct(
    settings_factory, tmp_path
):
    """The collision regression.

    Both files are named the same and neither carries a session id, so both
    scanners fall back to the filename. Without host qualification these produce
    one document id and one machine silently overwrites the other.
    """

    local = tmp_path / "local"
    synced = tmp_path / "synced"
    local.mkdir()
    synced.mkdir()
    _session(local, "session.jsonl")
    _session(synced, "session.jsonl")

    settings = settings_factory(
        claude_roots=(local, synced),
        agent_root_hosts={str(synced): "laptop"},
    )

    result = scan_claude(settings, CUTOFF)

    assert result.successful
    assert len(result.documents) == 2, "one machine overwrote the other"
    keys = {document.source_key for document in result.documents}
    assert keys == {"session:session", "session:laptop:session"}
    assert (
        len(
            {
                stable_document_id(document.source, document.source_key)
                for document in result.documents
            }
        )
        == 2
    )


def test_a_synced_session_records_which_machine_it_came_from(
    settings_factory, tmp_path
):
    synced = tmp_path / "synced"
    synced.mkdir()
    _session(synced, "remote.jsonl", session_id="remote-1")

    settings = settings_factory(
        claude_roots=(synced,), agent_root_hosts={str(synced): "laptop"}
    )

    result = scan_claude(settings, CUTOFF)

    assert result.successful
    document = result.documents[0]
    assert document.source_key == "session:laptop:remote-1"
    assert document.metadata["host"] == "laptop"


def test_a_local_session_is_keyed_exactly_as_before(settings_factory, tmp_path):
    """Byte-for-byte compatibility with everything already in the database."""

    local = tmp_path / "local"
    local.mkdir()
    _session(local, "local.jsonl", session_id="local-1")

    settings = settings_factory(claude_roots=(local,))

    result = scan_claude(settings, CUTOFF)

    document = result.documents[0]
    assert document.source_key == "session:local-1"
    assert stable_document_id(document.source, document.source_key) == (
        stable_document_id("claude", "session:local-1")
    )
    assert document.metadata["host"] is None
