from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from importers.agent_history import (
    _read_jsonl,
    parse_claude_records,
    parse_codex_records,
    parse_grok_records,
    parse_hermes_session,
    scan_claude,
)
from redaction import REDACTED, redact_text


FIXTURES = Path(__file__).parent / "fixtures"
CUTOFF = datetime(2026, 6, 1, tzinfo=timezone.utc)
FALLBACK = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _records(name: str):
    records, _ = _read_jsonl(FIXTURES / name)
    return records


def test_claude_main_chain_role_and_block_filtering():
    messages = parse_claude_records(_records("claude.jsonl"), cutoff=CUTOFF, fallback=FALLBACK)
    assert [message.role for message in messages] == ["user", "assistant"]
    combined = " ".join(message.text for message in messages)
    assert "visible user prompt" in combined
    assert "visible assistant reply" in combined
    assert "hidden reasoning" not in combined
    assert "hidden tool call" not in combined
    assert "hidden sidechain" not in combined
    assert "outside rolling window" not in combined
    assert "sk-proj-" not in combined
    assert REDACTED in combined


def test_claude_strips_synthetic_local_command_and_tool_xml():
    records = [
        {
            "type": "user",
            "timestamp": "2026-07-01T12:00:00Z",
            "message": {
                "role": "user",
                "content": "<local-command-stdout>hidden command output</local-command-stdout>",
            },
        },
        {
            "type": "user",
            "timestamp": "2026-07-01T12:00:01Z",
            "message": {
                "role": "user",
                "content": "<command-args>model switch</command-args>\nvisible user words",
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-07-01T12:00:02Z",
            "message": {
                "role": "assistant",
                "content": "<tool_result>hidden tool result</tool_result>\nvisible assistant words",
            },
        },
    ]
    messages = parse_claude_records(records, cutoff=CUTOFF, fallback=FALLBACK)
    combined = " ".join(message.text for message in messages)
    assert [message.text for message in messages] == [
        "visible user words",
        "visible assistant words",
    ]
    assert "command output" not in combined
    assert "model switch" not in combined
    assert "tool result" not in combined


def test_codex_prefers_canonical_messages_and_excludes_internal_records():
    messages = parse_codex_records(_records("codex.jsonl"), cutoff=CUTOFF, fallback=FALLBACK)
    assert [message.text for message in messages] == [
        "codex visible user",
        "codex visible assistant",
    ]


def test_grok_reconstructs_only_user_and_agent_chunks():
    messages = parse_grok_records(_records("grok.jsonl"), cutoff=CUTOFF, fallback=FALLBACK)
    assert [(message.role, message.text) for message in messages] == [
        ("user", "grok visible user reconstructed"),
        ("assistant", "grok visible assistant"),
    ]


def test_hermes_uses_visible_content_only():
    session = _records("hermes.jsonl")[0]
    messages = parse_hermes_session(session, cutoff=CUTOFF, fallback=FALLBACK)
    combined = " ".join(message.text for message in messages)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert "hidden reasoning" not in combined
    assert "hidden tool" not in combined
    assert "old hermes" not in combined


def test_malformed_jsonl_is_tolerated_by_scanner(settings_factory, tmp_path):
    root = tmp_path / "claude"
    root.mkdir()
    target = root / "session.jsonl"
    target.write_text((FIXTURES / "claude.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    settings = settings_factory(claude_roots=(root,))
    result = scan_claude(settings, CUTOFF)
    assert result.successful
    assert result.malformed == 1
    assert len(result.documents) == 1


def test_common_secret_and_private_key_redaction():
    raw = (
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz\n"
        "password=hunter2-secret\n"
        "-----BEGIN PRIVATE KEY-----\nsecret body\n-----END PRIVATE KEY-----"
    )
    safe = redact_text(raw)
    assert "hunter2" not in safe
    assert "secret body" not in safe
    assert "abcdefghijklmnopqrstuvwxyz" not in safe
    assert safe.count(REDACTED) >= 3
    assert redact_text(safe) == safe


def test_missing_timestamp_is_excluded_from_strict_window():
    records = [
        {"type": "user", "message": {"role": "user", "content": "undated content"}}
    ]
    assert parse_claude_records(records, cutoff=CUTOFF, fallback=FALLBACK) == []
