"""Parsers for redacted Hermes, Claude Code, Codex, and Grok dialogue."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable
from urllib.parse import unquote

from config import Settings
from models import IngestDocument, ScanResult
from redaction import redact_text


_UTC = timezone.utc
_EPOCH = datetime.fromtimestamp(0, _UTC)
_NON_DIALOGUE_XML = re.compile(
    r"<(?P<tag>system-reminder|local-command-caveat|local-command-stdout|"
    r"local-command-stderr|local-command-output|command-name|command-message|"
    r"command-args|tool-result|tool_result|tool-use|tool_use)\b[^>]*>"
    r".*?</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class DialogueMessage:
    role: str
    text: str
    timestamp: datetime


def _timestamp(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000
        try:
            return datetime.fromtimestamp(number, _UTC)
        except (OSError, OverflowError, ValueError):
            return fallback
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_UTC)
            return parsed.astimezone(_UTC)
        except ValueError:
            pass
    return fallback


def _plain_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        kind = str(content.get("type", "text")).casefold()
        value = content.get("text")
        return value if kind in {"text", "input_text", "output_text"} and isinstance(value, str) else ""
    if not isinstance(content, list):
        return ""
    pieces: list[str] = []
    for block in content:
        if isinstance(block, str):
            pieces.append(block)
            continue
        if not isinstance(block, dict):
            continue
        kind = str(block.get("type", "")).casefold()
        if kind not in {"text", "input_text", "output_text"}:
            continue
        value = block.get("text")
        if isinstance(value, str):
            pieces.append(value)
    return "\n".join(pieces)


def _clean_dialogue_text(value: str) -> str:
    text = _NON_DIALOGUE_XML.sub("", value)
    return redact_text(text).strip()


def parse_claude_records(
    records: Iterable[dict[str, Any]], *, cutoff: datetime, fallback: datetime
) -> list[DialogueMessage]:
    messages: list[DialogueMessage] = []
    for record in records:
        top_type = str(record.get("type", "")).casefold()
        if top_type not in {"user", "assistant"}:
            continue
        if record.get("isSidechain") or record.get("is_sidechain") or record.get("isMeta"):
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", top_type)).casefold()
        if role not in {"user", "assistant"}:
            continue
        text = _clean_dialogue_text(_plain_text(message.get("content")))
        # A missing/invalid message timestamp cannot be proven to fall inside
        # the strict rolling window, so it is excluded rather than inheriting a
        # recently touched file's mtime.
        when = _timestamp(record.get("timestamp") or message.get("timestamp"), _EPOCH)
        if text and when >= cutoff:
            messages.append(DialogueMessage(role, text, when))
    return messages


def parse_codex_records(
    records: Iterable[dict[str, Any]], *, cutoff: datetime, fallback: datetime
) -> list[DialogueMessage]:
    canonical: list[DialogueMessage] = []
    fallback_events: list[DialogueMessage] = []
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        when = _timestamp(record.get("timestamp") or payload.get("timestamp"), _EPOCH)
        if when < cutoff:
            continue
        record_type = str(record.get("type", "")).casefold()
        payload_type = str(payload.get("type", "")).casefold()
        if record_type == "response_item" and payload_type == "message":
            role = str(payload.get("role", "")).casefold()
            if role not in {"user", "assistant"}:
                continue
            text = _clean_dialogue_text(_plain_text(payload.get("content")))
            if text:
                canonical.append(DialogueMessage(role, text, when))
        elif record_type == "event_msg" and payload_type in {"user_message", "agent_message"}:
            role = "user" if payload_type == "user_message" else "assistant"
            text = _clean_dialogue_text(
                str(payload.get("message") or payload.get("text") or "")
            )
            if text:
                fallback_events.append(DialogueMessage(role, text, when))
    return canonical if canonical else fallback_events


def parse_grok_records(
    records: Iterable[dict[str, Any]], *, cutoff: datetime, fallback: datetime
) -> list[DialogueMessage]:
    messages: list[DialogueMessage] = []
    current_role: str | None = None
    current_parts: list[str] = []
    current_time = fallback

    def flush() -> None:
        nonlocal current_role, current_parts, current_time
        text = _clean_dialogue_text("".join(current_parts))
        if current_role and text and current_time >= cutoff:
            messages.append(DialogueMessage(current_role, text, current_time))
        current_role = None
        current_parts = []
        current_time = fallback

    for record in records:
        params = record.get("params")
        update = params.get("update") if isinstance(params, dict) else None
        if not isinstance(update, dict):
            continue
        kind = str(update.get("sessionUpdate", "")).casefold()
        if kind not in {"user_message_chunk", "agent_message_chunk"}:
            if kind == "turn_completed":
                flush()
            continue
        role = "user" if kind == "user_message_chunk" else "assistant"
        when = _timestamp(record.get("timestamp"), _EPOCH)
        text = _plain_text(update.get("content"))
        if role != current_role:
            flush()
            current_role = role
            current_time = when
        current_parts.append(text)
        if when > current_time:
            current_time = when
    flush()
    return messages


def parse_hermes_session(
    session: dict[str, Any], *, cutoff: datetime, fallback: datetime
) -> list[DialogueMessage]:
    output: list[DialogueMessage] = []
    raw_messages = session.get("messages")
    if not isinstance(raw_messages, list):
        return output
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).casefold()
        if role not in {"user", "assistant"}:
            continue
        # Only the visible content field is accepted. Reasoning and provider
        # message/tool fields are deliberately never traversed.
        text = _clean_dialogue_text(_plain_text(message.get("content")))
        when = _timestamp(message.get("timestamp"), _EPOCH)
        if text and when >= cutoff:
            output.append(DialogueMessage(role, text, when))
    return output


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                malformed += 1
    return records, malformed


def _iso(value: datetime) -> str:
    return value.astimezone(_UTC).isoformat().replace("+00:00", "Z")


def _dialogue_document(
    *,
    source: str,
    source_key: str,
    title: str,
    messages: list[DialogueMessage],
    project: str | None,
    uri: str | None,
    metadata: dict[str, Any],
) -> IngestDocument | None:
    if not messages:
        return None
    paragraphs = [f"{item.role.upper()} [{_iso(item.timestamp)}]\n{item.text}" for item in messages]
    return IngestDocument(
        source=source,
        source_key=source_key,
        title=title,
        text="\n\n".join(paragraphs),
        timestamp=max(item.timestamp for item in messages),
        project=project,
        uri=uri,
        metadata={**metadata, "message_count": len(messages), "roles": ["user", "assistant"]},
    )


def _file_fallback(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, _UTC)


def _project_from_cwd(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return Path(value).name or None
    except (OSError, ValueError):
        return None


def _dedupe_documents(documents: Iterable[IngestDocument]) -> list[IngestDocument]:
    selected: dict[str, IngestDocument] = {}
    for document in documents:
        prior = selected.get(document.source_key)
        if prior is None or document.timestamp >= prior.timestamp:
            selected[document.source_key] = document
    return list(selected.values())


def _jsonl_files(roots: Iterable[Path], name: str = "*.jsonl") -> tuple[list[Path], bool]:
    existing = [root for root in roots if root.exists() and root.is_dir()]
    if not existing:
        return [], False
    files: set[Path] = set()
    for root in existing:
        files.update(path for path in root.rglob(name) if path.is_file() and not path.is_symlink())
    return sorted(files), True


def scan_claude(settings: Settings, cutoff: datetime) -> ScanResult:
    result = ScanResult(source="claude")
    files, available = _jsonl_files(settings.claude_roots)
    if not available:
        result.successful = False
        result.error = "Claude history root is unavailable"
        return result
    documents: list[IngestDocument] = []
    try:
        for path in files:
            result.scanned += 1
            records, malformed = _read_jsonl(path)
            result.malformed += malformed
            fallback = _file_fallback(path)
            messages = parse_claude_records(records, cutoff=cutoff, fallback=fallback)
            session_id = next(
                (
                    str(record.get("sessionId"))
                    for record in records
                    if record.get("sessionId")
                ),
                path.stem,
            )
            cwd = next((record.get("cwd") for record in records if record.get("cwd")), None)
            document = _dialogue_document(
                source="claude",
                source_key=f"session:{session_id}",
                title=f"Claude Code session {session_id}",
                messages=messages,
                project=_project_from_cwd(cwd) or path.parent.name,
                uri=str(path.resolve()),
                metadata={"session_id": session_id, "cwd": cwd},
            )
            if document:
                documents.append(document)
            else:
                # The session exists on disk but produced no fresh document,
                # almost always because every message aged out of the rolling
                # window. Retaining the key keeps reconciliation from deleting
                # knowledge that is still on disk and will never be rebuilt.
                result.retained_keys.add(f"session:{session_id}")
                result.skipped += 1
    except (OSError, UnicodeError) as exc:
        result.successful = False
        result.error = f"Claude scan failed: {exc}"
        return result
    result.documents = _dedupe_documents(documents)
    result.watermark = _iso(max((doc.timestamp for doc in result.documents), default=cutoff))
    return result


def scan_codex(settings: Settings, cutoff: datetime) -> ScanResult:
    result = ScanResult(source="codex")
    files, available = _jsonl_files(settings.codex_roots)
    if not available:
        result.successful = False
        result.error = "Codex history roots are unavailable"
        return result
    documents: list[IngestDocument] = []
    try:
        for path in files:
            result.scanned += 1
            records, malformed = _read_jsonl(path)
            result.malformed += malformed
            fallback = _file_fallback(path)
            messages = parse_codex_records(records, cutoff=cutoff, fallback=fallback)
            session_meta = next(
                (
                    record.get("payload")
                    for record in records
                    if record.get("type") == "session_meta" and isinstance(record.get("payload"), dict)
                ),
                {},
            )
            session_id = str(session_meta.get("id") or path.stem)
            cwd = session_meta.get("cwd")
            document = _dialogue_document(
                source="codex",
                source_key=f"session:{session_id}",
                title=f"Codex session {session_id}",
                messages=messages,
                project=_project_from_cwd(cwd),
                uri=str(path.resolve()),
                metadata={"session_id": session_id, "cwd": cwd},
            )
            if document:
                documents.append(document)
            else:
                # The session exists on disk but produced no fresh document,
                # almost always because every message aged out of the rolling
                # window. Retaining the key keeps reconciliation from deleting
                # knowledge that is still on disk and will never be rebuilt.
                result.retained_keys.add(f"session:{session_id}")
                result.skipped += 1
    except (OSError, UnicodeError) as exc:
        result.successful = False
        result.error = f"Codex scan failed: {exc}"
        return result
    result.documents = _dedupe_documents(documents)
    result.watermark = _iso(max((doc.timestamp for doc in result.documents), default=cutoff))
    return result


def scan_grok(settings: Settings, cutoff: datetime) -> ScanResult:
    result = ScanResult(source="grok")
    files, available = _jsonl_files(settings.grok_roots, "updates.jsonl")
    if not available:
        result.successful = False
        result.error = "Grok history roots are unavailable"
        return result
    documents: list[IngestDocument] = []
    try:
        for path in files:
            result.scanned += 1
            records, malformed = _read_jsonl(path)
            result.malformed += malformed
            fallback = _file_fallback(path)
            messages = parse_grok_records(records, cutoff=cutoff, fallback=fallback)
            session_id = path.parent.name
            if records:
                params = records[0].get("params")
                if isinstance(params, dict) and params.get("sessionId"):
                    session_id = str(params["sessionId"])
            encoded_cwd = path.parent.parent.name
            cwd = unquote(encoded_cwd)
            document = _dialogue_document(
                source="grok",
                source_key=f"session:{session_id}",
                title=f"Grok session {session_id}",
                messages=messages,
                project=_project_from_cwd(cwd),
                uri=str(path.resolve()),
                metadata={"session_id": session_id, "cwd": cwd},
            )
            if document:
                documents.append(document)
            else:
                # The session exists on disk but produced no fresh document,
                # almost always because every message aged out of the rolling
                # window. Retaining the key keeps reconciliation from deleting
                # knowledge that is still on disk and will never be rebuilt.
                result.retained_keys.add(f"session:{session_id}")
                result.skipped += 1
    except (OSError, UnicodeError) as exc:
        result.successful = False
        result.error = f"Grok scan failed: {exc}"
        return result
    result.documents = _dedupe_documents(documents)
    result.watermark = _iso(max((doc.timestamp for doc in result.documents), default=cutoff))
    return result


def scan_hermes(settings: Settings, cutoff: datetime) -> ScanResult:
    result = ScanResult(source="hermes")
    executable = shutil.which(settings.hermes_command)
    if not executable:
        result.successful = False
        result.error = "Hermes CLI is unavailable"
        return result
    try:
        completed = subprocess.run(
            [
                executable,
                "sessions",
                "export",
                "--format",
                "jsonl",
                "--redact",
                "--yes",
                "-",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result.successful = False
        result.error = f"Hermes redacted export failed: {exc}"
        return result
    if completed.returncode != 0:
        result.successful = False
        result.error = f"Hermes redacted export exited with code {completed.returncode}"
        return result

    documents: list[IngestDocument] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        # Hermes may emit human-readable progress/footer lines on stdout. They
        # are exporter noise rather than malformed history records.
        if not stripped or not stripped.startswith("{"):
            continue
        result.scanned += 1
        try:
            session = json.loads(stripped)
        except json.JSONDecodeError:
            result.malformed += 1
            continue
        if not isinstance(session, dict):
            result.malformed += 1
            continue
        fallback = _timestamp(session.get("ended_at") or session.get("started_at"), cutoff)
        messages = parse_hermes_session(session, cutoff=cutoff, fallback=fallback)
        session_id = str(session.get("id") or f"row-{result.scanned}")
        cwd = session.get("cwd")
        document = _dialogue_document(
            source="hermes",
            source_key=f"session:{session_id}",
            title=str(session.get("title") or f"Hermes session {session_id}"),
            messages=messages,
            project=_project_from_cwd(cwd),
            uri=f"hermes://session/{session_id}",
            metadata={
                "session_id": session_id,
                "cwd": cwd,
                "origin": session.get("source"),
                "model": session.get("model"),
            },
        )
        if document:
            documents.append(document)
        else:
            # Same reasoning as the file-based scanners above.
            result.retained_keys.add(f"session:{session_id}")
            result.skipped += 1
    result.documents = _dedupe_documents(documents)
    result.watermark = _iso(max((doc.timestamp for doc in result.documents), default=cutoff))
    return result
