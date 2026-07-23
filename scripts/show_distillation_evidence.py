"""Print one redacted distillation unit with its mapped raw evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_settings  # noqa: E402
from distillation import reconstruct_chunks  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument("--unit", type=int, default=0)
    args = parser.parse_args()
    settings = load_settings()
    with sqlite3.connect(settings.database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT x.*, d.source, d.title
            FROM distillations x
            JOIN documents d ON d.id = x.document_id
            WHERE x.document_id = ? AND x.unit_ordinal = ?
              AND x.distiller_model = ? AND x.prompt_version = ?
            """,
            (
                args.document_id,
                args.unit,
                settings.distillation.model,
                settings.distillation.prompt_version,
            ),
        ).fetchone()
        if row is None:
            raise SystemExit("current distillation unit not found")
        chunks = connection.execute(
            """
            SELECT ordinal, content FROM chunks
            WHERE document_id = ? AND ordinal BETWEEN ? AND ?
            ORDER BY ordinal
            """,
            (args.document_id, row["start_ordinal"], row["end_ordinal"]),
        ).fetchall()
    raw, _spans = reconstruct_chunks(
        [(int(chunk["ordinal"]), str(chunk["content"])) for chunk in chunks]
    )
    print(
        json.dumps(
            {
                "document_id": args.document_id,
                "source": row["source"],
                "title": row["title"],
                "unit": int(row["unit_ordinal"]),
                "raw_range": [int(row["start_ordinal"]), int(row["end_ordinal"])],
                "summary": json.loads(str(row["summary_json"])),
                "raw_evidence": raw,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
