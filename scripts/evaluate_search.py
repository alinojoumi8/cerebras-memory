"""Evaluate the fixed local search-quality label set without printing content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_settings  # noqa: E402
from store import KnowledgeStore  # noqa: E402


def _rank(results: Sequence[dict[str, Any]], document_id: str) -> int | None:
    return next(
        (index for index, item in enumerate(results, start=1) if item["document_id"] == document_id),
        None,
    )


def _metrics(ranks: Sequence[int | None]) -> dict[str, float]:
    count = len(ranks)
    return {
        "recall_at_8": round(
            sum(rank is not None and rank <= 8 for rank in ranks) / count if count else 0.0,
            6,
        ),
        "mrr_at_8": round(
            sum(1.0 / rank if rank else 0.0 for rank in ranks) / count if count else 0.0,
            6,
        ),
    }


def _percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.999999) - 1))
    return ordered[index]


def evaluate(
    label_path: Path,
    *,
    config_path: Path | None = None,
    include_distillations: bool | None = None,
) -> dict[str, Any]:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 24:
        raise ValueError("The labeled baseline must contain exactly 24 cases")
    store = KnowledgeStore(load_settings(config_path))

    # Load the cached reranker outside the measured interval.
    if cases:
        first = cases[0]
        store.search_response(
            str(first["query"]),
            limit=8,
            sources=[str(first["source"])] if first.get("source") else None,
            global_search=True,
            rerank=True,
            include_distillations=include_distillations,
        )

    baseline_ranks: list[int | None] = []
    upgraded_ranks: list[int | None] = []
    warm_latencies: list[float] = []
    reports: list[dict[str, Any]] = []
    distinct = True
    scopes_correct = True
    for case in cases:
        query = str(case["query"])
        expected = str(case["expected_document_id"])
        sources = [str(case["source"])] if case.get("source") else None
        roots = [Path(case["scope_path"])] if case.get("scope_path") else None
        baseline = store.search_response(
            query,
            limit=8,
            sources=sources,
            global_search=True,
            rerank=False,
            include_distillations=False,
        )
        started = time.perf_counter()
        upgraded = store.search_response(
            query,
            limit=8,
            sources=sources,
            roots=roots,
            cwd=roots[0] if roots else None,
            include_distillations=include_distillations,
        )
        warm_latencies.append((time.perf_counter() - started) * 1000.0)
        baseline_rank = _rank(baseline["results"], expected)
        upgraded_rank = _rank(upgraded["results"], expected)
        baseline_ranks.append(baseline_rank)
        upgraded_ranks.append(upgraded_rank)
        document_ids = [item["document_id"] for item in upgraded["results"]]
        distinct = distinct and len(document_ids) == len(set(document_ids))
        expected_scope = case.get("expected_scope")
        if expected_scope is not None:
            scopes_correct = scopes_correct and upgraded["scope"] == expected_scope
        reports.append(
            {
                "label": case.get("label"),
                "source": case.get("source"),
                "expected_document_id": expected,
                "baseline_rank": baseline_rank,
                "upgraded_rank": upgraded_rank,
                "scope": upgraded["scope"],
                "latency_ms": round(warm_latencies[-1], 3),
                "distinct_top_8": len(document_ids) == len(set(document_ids)),
            }
        )

    baseline_metrics = _metrics(baseline_ranks)
    upgraded_metrics = _metrics(upgraded_ranks)
    p95 = _percentile_95(warm_latencies)
    no_regression = bool(
        upgraded_metrics["recall_at_8"] >= baseline_metrics["recall_at_8"]
        and upgraded_metrics["mrr_at_8"] >= baseline_metrics["mrr_at_8"]
    )
    return {
        "label_set": str(label_path.resolve()),
        "cases": len(cases),
        "sources": sorted({str(case.get("source")) for case in cases}),
        "distillation_channel": (
            "configured" if include_distillations is None else "enabled" if include_distillations else "disabled"
        ),
        "baseline": baseline_metrics,
        "upgraded": upgraded_metrics,
        "no_retrieval_quality_regression": no_regression,
        "all_top_8_document_ids_distinct": distinct,
        "all_expected_scopes_correct": scopes_correct,
        "warm_p95_latency_ms": round(p95, 3),
        "warm_p95_below_1500_ms": p95 < 1_500.0,
        "gate_passed": bool(no_regression and distinct and scopes_correct and p95 < 1_500.0),
        "results": reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "evaluation" / "search-quality-baseline.json",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-distillations",
        action="store_true",
        default=None,
        help="force the staged distillation channel on for the quality gate",
    )
    args = parser.parse_args(argv)
    result = evaluate(
        args.labels,
        config_path=args.config,
        include_distillations=args.include_distillations,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
