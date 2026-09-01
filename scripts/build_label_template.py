"""Emit a stratified labelling template for the search-quality gold set.

The gold set is saturated at recall@8 = 1.0 over 24 single-label cases, so it
can no longer detect improvement. Growing it needs *human* queries: labels
generated from the documents they are meant to retrieve are circular and score
well by construction rather than by merit (the same defect that made the old
distillation evaluation meaningless).

This script therefore emits candidate documents stratified across sources and
projects with the ``query`` field left blank. Fill in a question you would
actually ask, adjust ``relevant`` (add every document that genuinely answers it,
with a gain of 3 for a direct answer and 1 for useful supporting context), then
merge the result into ``evaluation/search-quality-baseline.json``.

``evaluate_search.py`` refuses any case with an empty query, so an unfinished
template cannot silently inflate the metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_settings  # noqa: E402
from store import KnowledgeStore  # noqa: E402


def build_template(
    *,
    config_path: Path | None = None,
    per_source: int = 20,
    existing: Path | None = None,
) -> dict[str, Any]:
    store = KnowledgeStore(load_settings(config_path))

    already: set[str] = set()
    if existing and existing.exists():
        payload = json.loads(existing.read_text(encoding="utf-8"))
        for case in payload.get("cases", []):
            if not isinstance(case, dict):
                continue
            if case.get("expected_document_id"):
                already.add(str(case["expected_document_id"]))
            for item in case.get("relevant", []) or []:
                if isinstance(item, dict) and item.get("document_id"):
                    already.add(str(item["document_id"]))

    with store._connect() as connection:  # noqa: SLF001 - local admin script
        rows = connection.execute(
            """
            SELECT id, source, project, title, timestamp
            FROM documents
            WHERE kind = 'derived'
            ORDER BY source, project, timestamp DESC
            """
        ).fetchall()

    # Stratify: round-robin across (source, project) buckets so a single large
    # project cannot dominate the template.
    buckets: dict[tuple[str, str], list[Any]] = {}
    for row in rows:
        if str(row["id"]) in already:
            continue
        buckets.setdefault((str(row["source"]), str(row["project"] or "")), []).append(row)

    per_source_counts: dict[str, int] = {}
    cases: list[dict[str, Any]] = []
    exhausted = False
    while not exhausted:
        exhausted = True
        for key in sorted(buckets):
            source, project = key
            if per_source_counts.get(source, 0) >= per_source:
                continue
            if not buckets[key]:
                continue
            exhausted = False
            row = buckets[key].pop(0)
            per_source_counts[source] = per_source_counts.get(source, 0) + 1
            case: dict[str, Any] = {
                "label": f"{source}-{(project or 'global')}-{len(cases) + 1}",
                "source": source,
                "query": "",
                "_hint_title": str(row["title"]),
                "relevant": [{"document_id": str(row["id"]), "gain": 3}],
            }
            if project:
                case["scope_project"] = project
            cases.append(case)

    return {
        "version": 2,
        "description": (
            "Graded relevance label set. Every case needs a human-written query. "
            "gain 3 = directly answers, 1 = useful supporting context. "
            "Delete _hint_title once the query is written; it exists only to "
            "remind you what the document is, and must not be pasted into the "
            "query, which would make the case circular."
        ),
        "cases": cases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--per-source",
        type=int,
        default=20,
        help="maximum candidate cases to emit per source",
    )
    parser.add_argument(
        "--existing",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "search-quality-baseline.json",
        help="skip documents already judged in this label set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "label-template.json",
    )
    args = parser.parse_args(argv)
    template = build_template(
        config_path=args.config,
        per_source=args.per_source,
        existing=args.existing,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(template['cases'])} unlabelled candidates to {args.output}\n"
        "Fill in every 'query' field, then merge into the gold set.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
