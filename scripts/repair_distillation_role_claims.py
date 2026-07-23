"""Selectively regenerate summaries that contradict raw dialogue roles."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_settings  # noqa: E402
from store import KnowledgeStore  # noqa: E402


DENIES_RESPONSE = re.compile(
    r"\b(?:no|without)\s+(?:assistant|agent)?\s*response\b"
    r"|\bonly\s+the\s+user(?:'s)?\s+(?:initial\s+)?(?:question|message)\b"
)


def main() -> int:
    settings = load_settings()
    affected: dict[str, set[str]] = defaultdict(set)
    with sqlite3.connect(settings.database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, document_id, input_hash, start_ordinal, end_ordinal, summary_json
            FROM distillations
            WHERE distiller_model = ? AND prompt_version = ?
            """,
            (settings.distillation.model, settings.distillation.prompt_version),
        ).fetchall()
        for row in rows:
            structured = json.loads(str(row["summary_json"]))
            claims = f"{structured.get('summary', '')} {structured.get('outcome', '')}".casefold()
            if not DENIES_RESPONSE.search(claims):
                continue
            raw = "\n".join(
                str(chunk[0])
                for chunk in connection.execute(
                    """
                    SELECT content FROM chunks
                    WHERE document_id = ? AND ordinal BETWEEN ? AND ?
                    ORDER BY ordinal
                    """,
                    (
                        row["document_id"],
                        row["start_ordinal"],
                        row["end_ordinal"],
                    ),
                )
            )
            if re.search(r"(?m)^ASSISTANT\s+\[", raw):
                affected[str(row["document_id"])].add(str(row["input_hash"]))

    store = KnowledgeStore(settings)
    reports = [
        store.distill_document(document_id, force_input_hashes=input_hashes)
        for document_id, input_hashes in sorted(affected.items())
    ]
    report = {
        "affected_documents": len(affected),
        "affected_units": sum(len(hashes) for hashes in affected.values()),
        "generated": sum(int(item.get("generated", 0)) for item in reports),
        "failed_documents": sum(item.get("status") != "ready" for item in reports),
        "reports": reports,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["failed_documents"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
