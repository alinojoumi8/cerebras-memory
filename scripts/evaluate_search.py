"""Evaluate the local search-quality label set without printing content.

The gate compares the full pipeline against a *persisted historical baseline*
rather than a same-run handicapped configuration.  A same-run comparison can
only ever prove that reranking beats naive search; it cannot detect that
absolute quality fell, which is how an 11% MRR swing previously passed.

The unaided (no rerank, no distillation, no scoping) configuration is still
measured and reported, but as a diagnostic only.  It no longer gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_settings  # noqa: E402
from store import KnowledgeStore  # noqa: E402


DEFAULT_REFERENCE = PROJECT_ROOT / "evaluation" / "quality-reference-baseline.json"

# Retrieval work is identical for any limit <= reranker.candidate_documents:
# the document stage always advances 20 anchors and the reranker always scores
# their context variants.  Only the final slice differs, so evaluating at 20
# costs the same as 8 while making recall@20 measurable.
EVAL_LIMIT = 20

# Metrics that must not regress against the reference baseline.
GATED_METRICS = ("recall_at_8", "mrr_at_8", "ndcg_at_10", "recall_at_20")

# Absolute minimums applied alongside the relative comparison.  A purely
# relative gate ("no worse than last time") permits an unbounded downward
# ratchet: every individual step looks acceptable while the absolute quality
# falls, which is exactly how MRR@8 drifted 0.890 -> 0.759 across a month with
# each decline absorbed into a re-recorded reference.  Floors are stored in the
# baseline file so they are explicit, versioned, and reviewable rather than
# buried in code.
DEFAULT_FLOORS = {
    "recall_at_8": 1.0,
    "recall_at_20": 1.0,
    "mrr_at_8": 0.78,
    "ndcg_at_10": 0.83,
}


def _label_set_fingerprint(cases: Sequence[Mapping[str, Any]]) -> str:
    """Content hash of the scored questions.

    Comparing metrics across different label sets is meaningless, so the
    baseline records which questions produced it.  Hashing the normalised case
    list rather than the raw file keeps the fingerprint stable under
    reformatting while still catching any semantic edit.
    """

    normalised = json.dumps(list(cases), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _case_relevance(case: Mapping[str, Any]) -> dict[str, float]:
    """Map ``document_id -> gain``.

    Accepts graded judgments via ``relevant`` and falls back to the legacy
    single-label ``expected_document_id`` with gain 1.0, so existing label sets
    keep producing identical numbers.
    """

    relevant = case.get("relevant")
    if isinstance(relevant, list) and relevant:
        graded: dict[str, float] = {}
        for item in relevant:
            if isinstance(item, Mapping):
                document_id = str(item.get("document_id") or "").strip()
                gain = float(item.get("gain", 1.0))
            else:
                document_id, gain = str(item).strip(), 1.0
            if document_id and gain > 0:
                graded[document_id] = gain
        if graded:
            return graded
    expected = case.get("expected_document_id")
    return {str(expected): 1.0} if expected else {}


def _first_relevant_rank(ranked: Sequence[str], relevance: Mapping[str, float]) -> int | None:
    return next(
        (index for index, document_id in enumerate(ranked, start=1) if document_id in relevance),
        None,
    )


def _recall_at(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    if not relevance:
        return 0.0
    found = sum(1 for document_id in ranked[:k] if document_id in relevance)
    return found / len(relevance)


def _ndcg_at(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    if not relevance:
        return 0.0
    dcg = sum(
        relevance.get(document_id, 0.0) / math.log2(index + 1)
        for index, document_id in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(index + 1) for index, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def _aggregate(per_case: Sequence[Mapping[str, float]]) -> dict[str, float]:
    count = len(per_case)
    if not count:
        return {name: 0.0 for name in GATED_METRICS}
    return {
        name: round(sum(float(case[name]) for case in per_case) / count, 6)
        for name in GATED_METRICS
    }


def _case_metrics(ranked: Sequence[str], relevance: Mapping[str, float]) -> dict[str, float]:
    rank = _first_relevant_rank(ranked, relevance)
    return {
        "recall_at_8": _recall_at(ranked, relevance, 8),
        "recall_at_20": _recall_at(ranked, relevance, 20),
        "mrr_at_8": (1.0 / rank) if rank and rank <= 8 else 0.0,
        "ndcg_at_10": _ndcg_at(ranked, relevance, 10),
    }


def _percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.999999) - 1))
    return ordered[index]


def _load_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _compare_to_reference(
    metrics: Mapping[str, float],
    reference: Mapping[str, Any] | None,
    *,
    tolerance: float,
    label_set_fingerprint: str | None = None,
    allow_label_set_change: bool = False,
) -> dict[str, Any]:
    """Compare current metrics against the persisted baseline.

    Absent a reference the comparison is vacuously true, so the very first run
    can record one.  ``tolerance`` absorbs float noise only; it is not a licence
    to drift.

    When ``label_set_fingerprint`` is supplied and the reference recorded a
    different one, the comparison *refuses* rather than producing a number.
    Scoring a new question set against old metrics is not a regression signal:
    the current set is saturated at ``recall@8 = recall@20 = 1.0``, so any
    larger or harder set would read as a catastrophic regression when in fact
    only the instrument changed.
    """

    if not reference:
        return {
            "reference_available": False,
            "no_retrieval_quality_regression": True,
            "deltas": {},
            "regressions": [],
        }
    reference_fingerprint = reference.get("label_set_sha256")
    if (
        label_set_fingerprint
        and reference_fingerprint
        and reference_fingerprint != label_set_fingerprint
        and not allow_label_set_change
    ):
        return {
            "reference_available": True,
            "reference_recorded_at": reference.get("recorded_at"),
            "label_set_matches_reference": False,
            "reference_label_set_sha256": reference_fingerprint,
            "label_set_sha256": label_set_fingerprint,
            # Cannot verify the absence of a regression, so do not claim it.
            "no_retrieval_quality_regression": False,
            "deltas": {},
            "regressions": ["label_set_changed"],
        }
    reference_metrics = reference.get("metrics")
    if not isinstance(reference_metrics, Mapping):
        reference_metrics = reference
    deltas: dict[str, float] = {}
    regressions: list[str] = []
    for name in GATED_METRICS:
        if name not in reference_metrics:
            continue
        delta = float(metrics[name]) - float(reference_metrics[name])
        deltas[name] = round(delta, 6)
        if delta < -tolerance:
            regressions.append(name)
    return {
        "reference_available": True,
        "reference_recorded_at": reference.get("recorded_at"),
        "label_set_matches_reference": (
            None
            if not (label_set_fingerprint and reference_fingerprint)
            else reference_fingerprint == label_set_fingerprint
        ),
        "reference_metrics": {
            name: reference_metrics[name] for name in GATED_METRICS if name in reference_metrics
        },
        "no_retrieval_quality_regression": not regressions,
        "deltas": deltas,
        "regressions": regressions,
    }


def _check_floors(
    metrics: Mapping[str, float],
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Check absolute minimums, independent of the relative comparison.

    A metric can be at or above the last recorded point and still be
    unacceptable.  Floors come from the baseline file when present so they can
    be raised deliberately; ``DEFAULT_FLOORS`` applies otherwise.
    """

    configured = (reference or {}).get("floors")
    floors = configured if isinstance(configured, Mapping) else DEFAULT_FLOORS
    breaches: list[str] = []
    applied: dict[str, float] = {}
    for name in GATED_METRICS:
        if name not in floors:
            continue
        floor = float(floors[name])
        applied[name] = floor
        if float(metrics[name]) < floor:
            breaches.append(name)
    return {
        "floors": applied,
        "floors_source": "baseline" if isinstance(configured, Mapping) else "default",
        "all_metrics_above_floor": not breaches,
        "below_floor": breaches,
    }


def evaluate(
    label_path: Path,
    *,
    config_path: Path | None = None,
    include_distillations: bool | None = None,
    reference_path: Path | None = None,
    tolerance: float = 1e-6,
    latency_budget_ms: float = 1_500.0,
    allow_label_set_change: bool = False,
) -> dict[str, Any]:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("The labeled baseline must contain at least one case")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each labeled case must be an object")
        if not str(case.get("query") or "").strip():
            raise ValueError("Each labeled case requires a query")
        if not _case_relevance(case):
            raise ValueError(
                f"Case {case.get('label') or '<unlabeled>'} has no relevance judgment"
            )
    store = KnowledgeStore(load_settings(config_path))

    # Load the cached reranker outside the measured interval.
    first = cases[0]
    store.search_response(
        str(first["query"]),
        limit=EVAL_LIMIT,
        sources=[str(first["source"])] if first.get("source") else None,
        global_search=True,
        rerank=True,
        include_distillations=include_distillations,
    )

    unaided_cases: list[dict[str, float]] = []
    upgraded_cases: list[dict[str, float]] = []
    warm_latencies: list[float] = []
    reports: list[dict[str, Any]] = []
    distinct = True
    scopes_correct = True
    for case in cases:
        query = str(case["query"])
        relevance = _case_relevance(case)
        sources = [str(case["source"])] if case.get("source") else None
        roots = [Path(case["scope_path"])] if case.get("scope_path") else None
        # ``scope_project`` names a project directly; it is *not* a path.  Many
        # project labels come from a session ``cwd`` and have no directory
        # beneath ``projects_root`` (19 of the 51 distinct values in
        # evaluation/label-template.json), so resolving them through the
        # filesystem silently yields ``origin: global`` and quietly measures
        # something other than what the case says.  ``resolve_project_scope``
        # matches by name instead.  Cases carrying neither key keep the
        # existing root/cwd fall-through untouched, so ``origin`` stays
        # ``global`` rather than becoming ``global_explicit``.
        project = str(case["scope_project"]).strip() if case.get("scope_project") else None
        unaided = store.search_response(
            query,
            limit=EVAL_LIMIT,
            sources=sources,
            global_search=True,
            rerank=False,
            include_distillations=False,
        )
        started = time.perf_counter()
        upgraded = store.search_response(
            query,
            limit=EVAL_LIMIT,
            sources=sources,
            project=project,
            roots=None if project else roots,
            cwd=None if project else (roots[0] if roots else None),
            include_distillations=include_distillations,
        )
        warm_latencies.append((time.perf_counter() - started) * 1000.0)

        unaided_ids = [str(item["document_id"]) for item in unaided["results"]]
        upgraded_ids = [str(item["document_id"]) for item in upgraded["results"]]
        unaided_cases.append(_case_metrics(unaided_ids, relevance))
        case_metrics = _case_metrics(upgraded_ids, relevance)
        upgraded_cases.append(case_metrics)

        top_8 = upgraded_ids[:8]
        distinct = distinct and len(top_8) == len(set(top_8))
        expected_scope = case.get("expected_scope")
        if expected_scope is not None:
            scopes_correct = scopes_correct and upgraded["scope"] == expected_scope
        reports.append(
            {
                "label": case.get("label"),
                "source": case.get("source"),
                "relevant_documents": sorted(relevance),
                "unaided_rank": _first_relevant_rank(unaided_ids, relevance),
                "upgraded_rank": _first_relevant_rank(upgraded_ids, relevance),
                "metrics": {name: round(value, 6) for name, value in case_metrics.items()},
                "scope": upgraded["scope"],
                "latency_ms": round(warm_latencies[-1], 3),
                "distinct_top_8": len(top_8) == len(set(top_8)),
            }
        )

    unaided_metrics = _aggregate(unaided_cases)
    upgraded_metrics = _aggregate(upgraded_cases)
    p95 = _percentile_95(warm_latencies)
    reference = _load_reference(reference_path)
    fingerprint = _label_set_fingerprint(cases)
    regression = _compare_to_reference(
        upgraded_metrics,
        reference,
        tolerance=tolerance,
        label_set_fingerprint=fingerprint,
        allow_label_set_change=allow_label_set_change,
    )
    floors = _check_floors(upgraded_metrics, reference)
    latency_ok = p95 < latency_budget_ms
    return {
        "label_set": str(label_path.resolve()),
        "label_set_sha256": fingerprint,
        "cases": len(cases),
        "sources": sorted({str(case.get("source")) for case in cases}),
        "distillation_channel": (
            "configured"
            if include_distillations is None
            else "enabled"
            if include_distillations
            else "disabled"
        ),
        # Diagnostic only: the deliberately handicapped same-run configuration.
        "unaided": unaided_metrics,
        "upgraded": upgraded_metrics,
        "regression": regression,
        "no_retrieval_quality_regression": regression["no_retrieval_quality_regression"],
        "absolute_floors": floors,
        "all_metrics_above_floor": floors["all_metrics_above_floor"],
        "all_top_8_document_ids_distinct": distinct,
        "all_expected_scopes_correct": scopes_correct,
        "warm_p95_latency_ms": round(p95, 3),
        "warm_p95_latency_budget_ms": latency_budget_ms,
        "warm_p95_within_budget": latency_ok,
        "gate_passed": bool(
            regression["no_retrieval_quality_regression"]
            and floors["all_metrics_above_floor"]
            and distinct
            and scopes_correct
            and latency_ok
        ),
        "results": reports,
    }


def _write_reference(path: Path, result: Mapping[str, Any]) -> None:
    payload = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label_set": result["label_set"],
        "label_set_sha256": result.get("label_set_sha256"),
        "cases": result["cases"],
        "distillation_channel": result["distillation_channel"],
        "metrics": result["upgraded"],
        "warm_p95_latency_ms": result["warm_p95_latency_ms"],
    }
    # Floors are a policy decision, not a measurement.  Carry the existing ones
    # forward so recording a new measurement cannot silently drop the absolute
    # bar; raising or lowering them stays a deliberate edit to this file.
    existing = _load_reference(path) or {}
    floors = existing.get("floors")
    payload["floors"] = floors if isinstance(floors, Mapping) else dict(DEFAULT_FLOORS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
        "--baseline",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="persisted historical baseline the gate compares against",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record this run's metrics as the new persisted baseline (passing runs only)",
    )
    parser.add_argument(
        "--force-baseline",
        action="store_true",
        help=(
            "record the baseline even though the gate failed; this enshrines the "
            "current regression as the new bar and should be a deliberate, "
            "reviewed act"
        ),
    )
    parser.add_argument(
        "--allow-label-set-change",
        action="store_true",
        help="compare against a reference recorded from a different label set",
    )
    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        default=1_500.0,
        help="warm p95 latency budget for the gate",
    )
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
        reference_path=args.baseline,
        latency_budget_ms=args.latency_budget_ms,
        allow_label_set_change=args.allow_label_set_change,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.write_baseline or args.force_baseline:
        # A failing run must not be able to record itself as the new reference.
        # Doing so converts the gate into a ratchet that can only ever move
        # downward: each regression becomes the bar the next run is measured
        # against, and the decline stops being visible.
        if result["gate_passed"] or args.force_baseline:
            if not result["gate_passed"]:
                failed = sorted(
                    set(result["regression"].get("regressions", ()))
                    | set(result["absolute_floors"].get("below_floor", ()))
                )
                print(
                    "WARNING: recording a baseline from a FAILING run. "
                    f"Unmet: {', '.join(failed) or 'latency/scope/distinctness'}. "
                    "This makes the current regression the new bar.",
                    file=sys.stderr,
                )
            _write_reference(args.baseline, result)
            print(f"\nRecorded new baseline: {args.baseline}", file=sys.stderr)
        else:
            print(
                "\nRefusing to record a baseline: the gate did not pass. "
                "Fix the regression, or pass --force-baseline if lowering the "
                "bar is genuinely intended.",
                file=sys.stderr,
            )
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
