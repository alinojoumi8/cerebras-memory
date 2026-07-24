"""Versioned, content-free retrieval canaries and promotion gates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

from redaction import redact_text

if TYPE_CHECKING:
    from store import KnowledgeStore


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.999999) - 1))
    return ordered[index]


def load_canary_suite(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Canary suite must be a JSON object")
    version = str(payload.get("version") or "").strip()
    cases = payload.get("cases")
    if not version:
        raise ValueError("Canary suite requires a version")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        raise ValueError("Canary suite must contain between 1 and 100 cases")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each canary case must be an object")
        case_id = str(case.get("id") or "").strip()
        if not case_id or case_id in seen:
            raise ValueError("Canary case IDs must be present and unique")
        if not str(case.get("query") or "").strip():
            raise ValueError(f"Canary case {case_id} requires a query")
        seen.add(case_id)
    return payload, hashlib.sha256(raw).hexdigest()


def _case_sources(case: dict[str, Any]) -> list[str] | None:
    if isinstance(case.get("sources"), list):
        return [str(value) for value in case["sources"]]
    if case.get("source"):
        return [str(case["source"])]
    return None


def _evaluate_case(
    store: "KnowledgeStore",
    case: dict[str, Any],
    *,
    default_limit: int,
    default_latency_ms: float,
) -> tuple[dict[str, Any], float]:
    case_id = str(case["id"])
    query = redact_text(str(case["query"])).strip()
    limit = min(max(1, int(case.get("limit", default_limit))), 20)
    scope_path = case.get("scope_path")
    roots = [Path(str(scope_path))] if scope_path else None
    started = time.perf_counter()
    response = store.search_response(
        query,
        limit=limit,
        sources=_case_sources(case),
        project=str(case["project"]) if case.get("project") else None,
        global_search=bool(case.get("global_search", not bool(scope_path))),
        rerank=case.get("rerank"),
        roots=roots,
        cwd=roots[0] if roots else None,
        include_distillations=case.get("include_distillations"),
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    results = response["results"]
    document_ids = [str(item["document_id"]) for item in results]
    failures: list[str] = []

    expected_empty = bool(case.get("expected_empty", False))
    if expected_empty and results:
        failures.append("expected_empty")
    expected_document_id = case.get("expected_document_id")
    expected_rank: int | None = None
    if expected_document_id:
        expected = str(expected_document_id)
        expected_rank = next(
            (index for index, value in enumerate(document_ids, start=1) if value == expected),
            None,
        )
        if expected_rank is None:
            failures.append("expected_document_missing")

    forbidden_ids = {str(value) for value in case.get("forbidden_document_ids", [])}
    if forbidden_ids.intersection(document_ids):
        failures.append("forbidden_document_returned")
    forbidden_sources = {str(value).casefold() for value in case.get("forbidden_sources", [])}
    if any(str(item["source"]).casefold() in forbidden_sources for item in results):
        failures.append("forbidden_source_returned")
    allowed_projects = {
        str(value).casefold() for value in case.get("allowed_projects", [])
    }
    if allowed_projects and any(
        not item.get("project")
        or str(item["project"]).casefold() not in allowed_projects
        for item in results
    ):
        failures.append("project_scope_leak")

    expected_scope = case.get("expected_scope")
    if expected_scope is not None and response["scope"] != expected_scope:
        failures.append("scope_mismatch")
    if bool(case.get("require_distinct", True)) and len(document_ids) != len(
        set(document_ids)
    ):
        failures.append("duplicate_documents")
    if bool(case.get("require_stable_citations", True)) and any(
        not str(item.get("citation") or "").startswith("cerebras-memory://document/")
        for item in results
    ):
        failures.append("unstable_citation")
    if bool(case.get("require_untrusted_evidence", True)) and any(
        item.get("content_trust") != "untrusted_evidence" for item in results
    ):
        failures.append("trust_boundary_missing")

    latency_limit = float(case.get("latency_threshold_ms", default_latency_ms))
    if latency_ms > latency_limit:
        failures.append("latency_threshold_exceeded")
    return (
        {
            "case_id": case_id,
            "passed": not failures,
            "failures": failures,
            "result_count": len(results),
            "expected_rank": expected_rank,
            "scope": response["scope"],
            "latency_ms": round(latency_ms, 3),
            "score_stage": (
                results[0].get("score_stage") if results else None
            ),
        },
        latency_ms,
    )


def evaluate_canary_suite(
    store: "KnowledgeStore",
    path: Path,
    *,
    record: bool = True,
) -> dict[str, Any]:
    suite, suite_hash = load_canary_suite(path)
    defaults = suite.get("defaults") if isinstance(suite.get("defaults"), dict) else {}
    default_limit = min(max(1, int(defaults.get("limit", 8))), 20)
    default_latency_ms = float(
        defaults.get(
            "latency_threshold_ms",
            store.settings.canary_latency_threshold_ms,
        )
    )
    cases: list[dict[str, Any]] = suite["cases"]
    started_at = _iso_now()

    # Model initialization is not part of the warm latency gate.
    warm = cases[0]
    try:
        store.search_response(
            redact_text(str(warm["query"])),
            limit=1,
            sources=_case_sources(warm),
            global_search=True,
            rerank=warm.get("rerank"),
        )
    except Exception:
        pass

    reports: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in cases:
        try:
            report, latency = _evaluate_case(
                store,
                case,
                default_limit=default_limit,
                default_latency_ms=default_latency_ms,
            )
        except Exception as exc:
            report = {
                "case_id": str(case["id"]),
                "passed": False,
                "failures": [f"exception:{type(exc).__name__}"],
                "result_count": 0,
                "expected_rank": None,
                "scope": None,
                "latency_ms": 0.0,
                "score_stage": None,
            }
            latency = 0.0
        reports.append(report)
        latencies.append(latency)

    passed = sum(bool(report["passed"]) for report in reports)
    p95 = _p95(latencies)
    result: dict[str, Any] = {
        "suite_path": str(path.resolve()),
        "suite_version": str(suite["version"]),
        "suite_hash": suite_hash,
        "started_at": started_at,
        "completed_at": _iso_now(),
        "cases_total": len(reports),
        "cases_passed": passed,
        "p95_latency_ms": round(p95, 3),
        "gate_passed": passed == len(reports),
        "results": reports,
    }
    if record:
        result["run_id"] = store.record_canary_result(result)
    return result
