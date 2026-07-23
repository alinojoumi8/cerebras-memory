"""Audit the current distillation index without exposing stored content."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_settings  # noqa: E402
from distillation import (  # noqa: E402
    AGENT_SOURCES,
    DISTILLATION_FIELDS,
    validate_distillation,
)
from redaction import redact_text  # noqa: E402
from store import KnowledgeStore  # noqa: E402


def main() -> int:
    settings = load_settings()
    store = KnowledgeStore(settings)
    eligible = set(store._qualifying_distillation_documents())
    issues: Counter[str] = Counter()

    with sqlite3.connect(settings.database_path) as connection:
        connection.row_factory = sqlite3.Row
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        foreign_key_violations = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
        rows = connection.execute(
            """
            SELECT x.*, d.source
            FROM distillations x
            JOIN documents d ON d.id = x.document_id
            ORDER BY x.document_id, x.unit_ordinal
            """
        ).fetchall()
        current_rows = [
            row
            for row in rows
            if row["distiller_model"] == settings.distillation.model
            and row["prompt_version"] == settings.distillation.prompt_version
        ]
        current_document_ids = {str(row["document_id"]) for row in current_rows}
        old_model_rows = len(rows) - len(current_rows)
        chunk_ordinals: dict[str, set[int]] = defaultdict(set)
        chunk_content: dict[str, dict[int, str]] = defaultdict(dict)
        for row in connection.execute(
            """
            SELECT c.document_id, c.ordinal, c.content
            FROM chunks c
            JOIN distillation_state s ON s.document_id = c.document_id
            """
        ):
            document_id = str(row["document_id"])
            ordinal = int(row["ordinal"])
            chunk_ordinals[document_id].add(ordinal)
            chunk_content[document_id][ordinal] = str(row["content"])

        state_rows = connection.execute(
            "SELECT * FROM distillation_state"
        ).fetchall()
        state_by_document = {str(row["document_id"]): row for row in state_rows}
        unit_states = connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM distillation_unit_state
            WHERE distiller_model = ? AND prompt_version = ?
            GROUP BY status
            """,
            (settings.distillation.model, settings.distillation.prompt_version),
        ).fetchall()
        unit_state_counts = {str(row["status"]): int(row["count"]) for row in unit_states}
        table_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        fts_rows = (
            int(connection.execute("SELECT COUNT(*) FROM distillations_fts").fetchone()[0])
            if "distillations_fts" in table_names
            else -1
        )

    expected_fields = set(DISTILLATION_FIELDS)
    for row in current_rows:
        document_id = str(row["document_id"])
        if str(row["source"]) not in AGENT_SOURCES:
            issues["non_agent_source"] += 1
        try:
            structured = json.loads(str(row["summary_json"]))
            if not isinstance(structured, dict) or set(structured) != expected_fields:
                issues["schema_fields"] += 1
                continue
            if validate_distillation(structured) != structured:
                issues["schema_normalization"] += 1
            serialized = json.dumps(structured, ensure_ascii=False, sort_keys=True)
            if redact_text(serialized) != serialized:
                issues["redaction"] += 1
        except (TypeError, ValueError, json.JSONDecodeError):
            issues["invalid_json"] += 1
            continue

        if int(row["embedding_dimensions"]) != settings.embedding_dimensions:
            issues["embedding_dimensions"] += 1
        if str(row["embedding_model"]) != settings.embedding_model:
            issues["embedding_model"] += 1
        if len(bytes(row["embedding"])) != settings.embedding_dimensions * 4:
            issues["embedding_bytes"] += 1
        start = int(row["start_ordinal"])
        end = int(row["end_ordinal"])
        ordinals = chunk_ordinals.get(document_id, set())
        if start > end or start not in ordinals or end not in ordinals:
            issues["raw_range_bounds"] += 1
        if not all(ordinal in ordinals for ordinal in range(start, end + 1)):
            issues["raw_range_gap"] += 1
        claims = f"{structured.get('summary', '')} {structured.get('outcome', '')}".casefold()
        denies_response = bool(
            re.search(
                r"\b(?:no|without)\s+(?:assistant|agent)?\s*response\b"
                r"|\bonly\s+the\s+user(?:'s)?\s+(?:initial\s+)?(?:question|message)\b",
                claims,
            )
        )
        raw_range = "\n".join(
            chunk_content[document_id].get(ordinal, "")
            for ordinal in range(start, end + 1)
        )
        if denies_response and re.search(r"(?m)^ASSISTANT\s+\[", raw_range):
            issues["contradictory_no_response"] += 1
        expected_id = store._stable_distillation_id(
            document_id,
            str(row["input_hash"]),
            settings.distillation.model,
            settings.distillation.prompt_version,
        )
        if str(row["id"]) != expected_id:
            issues["unstable_id"] += 1

    for document_id in eligible:
        state = state_by_document.get(document_id)
        if state is None:
            issues["missing_state"] += 1
            continue
        if (
            str(state["status"]) != "ready"
            or str(state["model"]) != settings.distillation.model
            or str(state["prompt_version"]) != settings.distillation.prompt_version
            or int(state["failures"]) != 0
            or int(state["units_ready"]) != int(state["units_total"])
        ):
            issues["non_ready_state"] += 1

    if current_document_ids != eligible:
        issues["document_set_mismatch"] = len(current_document_ids ^ eligible)
    if unit_state_counts != {"ready": len(current_rows)}:
        issues["unit_state_mismatch"] += 1
    if fts_rows != len(rows):
        issues["fts_count_mismatch"] += 1
    if old_model_rows:
        issues["old_model_rows"] = old_model_rows
    if quick_check != "ok":
        issues["quick_check"] += 1
    if foreign_key_violations:
        issues["foreign_keys"] = foreign_key_violations
    if journal_mode.casefold() != "wal":
        issues["journal_mode"] += 1

    report = {
        "passed": not issues,
        "eligible_documents": len(eligible),
        "current_documents": len(current_document_ids),
        "current_units": len(current_rows),
        "fts_rows": fts_rows,
        "unit_states": unit_state_counts,
        "model": settings.distillation.model,
        "prompt_version": settings.distillation.prompt_version,
        "quick_check": quick_check,
        "foreign_key_violations": foreign_key_violations,
        "journal_mode": journal_mode,
        "issues": dict(sorted(issues.items())),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
