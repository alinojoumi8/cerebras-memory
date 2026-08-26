"""Metric and gate semantics for the search-quality harness."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_search  # noqa: E402
from evaluate_search import (  # noqa: E402
    DEFAULT_FLOORS,
    _aggregate,
    _case_metrics,
    _case_relevance,
    _check_floors,
    _compare_to_reference,
    _label_set_fingerprint,
    _ndcg_at,
    _recall_at,
    _write_reference,
    evaluate,
)


def _metrics(**overrides):
    base = {"recall_at_8": 1.0, "recall_at_20": 1.0, "mrr_at_8": 0.80, "ndcg_at_10": 0.85}
    base.update(overrides)
    return _aggregate([base])


def test_legacy_single_label_cases_keep_their_meaning():
    """The graded path must not silently change existing numbers."""

    relevance = _case_relevance({"expected_document_id": "doc_a"})
    assert relevance == {"doc_a": 1.0}

    hit = _case_metrics(["doc_x", "doc_a", "doc_y"], relevance)
    assert hit["recall_at_8"] == 1.0
    assert hit["mrr_at_8"] == pytest.approx(0.5)

    miss = _case_metrics(["doc_x", "doc_y"], relevance)
    assert miss["recall_at_8"] == 0.0
    assert miss["mrr_at_8"] == 0.0


def test_graded_relevance_is_read_and_ranked_by_gain():
    relevance = _case_relevance(
        {
            "relevant": [
                {"document_id": "doc_a", "gain": 3.0},
                {"document_id": "doc_b", "gain": 1.0},
                {"document_id": "doc_zero", "gain": 0.0},
            ]
        }
    )
    assert relevance == {"doc_a": 3.0, "doc_b": 1.0}

    # Perfect ordering scores 1.0; swapping the gains does not.
    assert _ndcg_at(["doc_a", "doc_b"], relevance, 10) == pytest.approx(1.0)
    assert _ndcg_at(["doc_b", "doc_a"], relevance, 10) < 1.0
    assert _recall_at(["doc_a"], relevance, 10) == pytest.approx(0.5)
    assert _recall_at(["doc_a", "doc_b"], relevance, 10) == pytest.approx(1.0)


def test_recall_at_20_sees_what_recall_at_8_cannot():
    relevance = {"doc_late": 1.0}
    ranked = [f"doc_{index}" for index in range(12)] + ["doc_late"]
    metrics = _case_metrics(ranked, relevance)
    assert metrics["recall_at_8"] == 0.0
    assert metrics["recall_at_20"] == 1.0


def test_gate_compares_against_a_persisted_reference_not_a_same_run_baseline():
    metrics = _aggregate([{"recall_at_8": 1.0, "recall_at_20": 1.0, "mrr_at_8": 0.70, "ndcg_at_10": 0.80}])

    # No reference yet: the first run is allowed to record one.
    fresh = _compare_to_reference(metrics, None, tolerance=1e-6)
    assert fresh["reference_available"] is False
    assert fresh["no_retrieval_quality_regression"] is True

    # Absolute decline against the recorded point is a regression, which the
    # old same-run comparison could not express.
    reference = {"metrics": {"recall_at_8": 1.0, "recall_at_20": 1.0, "mrr_at_8": 0.85, "ndcg_at_10": 0.90}}
    declined = _compare_to_reference(metrics, reference, tolerance=1e-6)
    assert declined["no_retrieval_quality_regression"] is False
    assert sorted(declined["regressions"]) == ["mrr_at_8", "ndcg_at_10"]
    assert declined["deltas"]["mrr_at_8"] == pytest.approx(-0.15)

    improved = _compare_to_reference(
        _aggregate([{"recall_at_8": 1.0, "recall_at_20": 1.0, "mrr_at_8": 0.95, "ndcg_at_10": 0.95}]),
        reference,
        tolerance=1e-6,
    )
    assert improved["no_retrieval_quality_regression"] is True
    assert improved["regressions"] == []


def test_label_set_size_is_not_hard_coded(tmp_path):
    """The old harness raised unless the set held exactly 24 cases."""

    label_path = tmp_path / "labels.json"
    label_path.write_text(
        json.dumps({"cases": [{"query": "q", "expected_document_id": "doc_a"}]}),
        encoding="utf-8",
    )
    # Reaching the store means the size check passed; the store is not built
    # here, so assert on the validation that runs before it instead.
    label_path.write_text(json.dumps({"cases": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="at least one case"):
        evaluate(label_path)

    label_path.write_text(
        json.dumps({"cases": [{"query": "q"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no relevance judgment"):
        evaluate(label_path)


def test_absolute_floor_fails_a_run_that_matches_the_reference_exactly():
    """A relative-only gate cannot stop a downward ratchet.

    Re-recording the reference from a failing run makes the next identical run
    pass, which is how MRR@8 drifted from 0.890 to 0.759 unchallenged.
    """

    degraded = _metrics(mrr_at_8=0.758681, ndcg_at_10=0.817753)
    reference = {"metrics": dict(degraded), "floors": dict(DEFAULT_FLOORS)}

    # Relative comparison is satisfied: the run equals the recorded point.
    relative = _compare_to_reference(degraded, reference, tolerance=1e-6)
    assert relative["no_retrieval_quality_regression"] is True

    # The absolute floor is not.
    floors = _check_floors(degraded, reference)
    assert floors["all_metrics_above_floor"] is False
    assert sorted(floors["below_floor"]) == ["mrr_at_8", "ndcg_at_10"]
    assert floors["floors_source"] == "baseline"

    healthy = _check_floors(_metrics(), reference)
    assert healthy["all_metrics_above_floor"] is True
    assert healthy["below_floor"] == []


def test_floors_fall_back_to_the_default_when_the_baseline_defines_none():
    result = _check_floors(_metrics(mrr_at_8=0.1), {"metrics": {}})
    assert result["floors_source"] == "default"
    assert result["below_floor"] == ["mrr_at_8"]


def test_comparison_refuses_across_a_changed_label_set():
    """Old metrics cannot judge new questions.

    recall@8 is saturated at 1.0 on the current 24-case set, so any larger or
    harder set would read as a catastrophic regression when only the instrument
    changed.
    """

    reference = {
        "metrics": {"recall_at_8": 1.0, "recall_at_20": 1.0, "mrr_at_8": 0.80, "ndcg_at_10": 0.85},
        "label_set_sha256": "aaa",
    }
    refused = _compare_to_reference(
        _metrics(recall_at_8=0.5),
        reference,
        tolerance=1e-6,
        label_set_fingerprint="bbb",
    )
    assert refused["label_set_matches_reference"] is False
    assert refused["regressions"] == ["label_set_changed"]
    # Must not claim the absence of a regression it could not measure.
    assert refused["no_retrieval_quality_regression"] is False

    allowed = _compare_to_reference(
        _metrics(),
        reference,
        tolerance=1e-6,
        label_set_fingerprint="bbb",
        allow_label_set_change=True,
    )
    assert allowed["no_retrieval_quality_regression"] is True

    matched = _compare_to_reference(
        _metrics(),
        reference,
        tolerance=1e-6,
        label_set_fingerprint="aaa",
    )
    assert matched["label_set_matches_reference"] is True


def test_label_set_fingerprint_is_stable_under_reformatting_but_not_edits():
    cases = [{"label": "a", "query": "q1", "expected_document_id": "doc_a"}]
    reordered = [{"expected_document_id": "doc_a", "query": "q1", "label": "a"}]
    assert _label_set_fingerprint(cases) == _label_set_fingerprint(reordered)

    edited = [{"label": "a", "query": "q2", "expected_document_id": "doc_a"}]
    assert _label_set_fingerprint(cases) != _label_set_fingerprint(edited)


def _canned_result(*, gate_passed: bool) -> dict:
    return {
        "label_set": "labels.json",
        "label_set_sha256": "abc",
        "cases": 24,
        "distillation_channel": "enabled",
        "upgraded": {"recall_at_8": 1.0, "recall_at_20": 1.0, "mrr_at_8": 0.7, "ndcg_at_10": 0.8},
        "warm_p95_latency_ms": 900.0,
        "regression": {"regressions": [] if gate_passed else ["mrr_at_8"]},
        "absolute_floors": {"below_floor": [] if gate_passed else ["mrr_at_8"]},
        "gate_passed": gate_passed,
    }


def test_write_baseline_refuses_a_failing_run(tmp_path, monkeypatch, capsys):
    baseline = tmp_path / "reference.json"
    monkeypatch.setattr(
        evaluate_search, "evaluate", lambda *a, **k: _canned_result(gate_passed=False)
    )

    exit_code = evaluate_search.main(
        [str(tmp_path / "labels.json"), "--baseline", str(baseline), "--write-baseline"]
    )
    assert exit_code == 1
    assert not baseline.exists(), "a failing run must not become the new bar"
    assert "Refusing to record a baseline" in capsys.readouterr().err


def test_force_baseline_writes_but_warns_loudly(tmp_path, monkeypatch, capsys):
    baseline = tmp_path / "reference.json"
    monkeypatch.setattr(
        evaluate_search, "evaluate", lambda *a, **k: _canned_result(gate_passed=False)
    )

    exit_code = evaluate_search.main(
        [str(tmp_path / "labels.json"), "--baseline", str(baseline), "--force-baseline"]
    )
    # Recording a baseline does not launder the failure into a pass.
    assert exit_code == 1
    assert baseline.exists()
    stderr = capsys.readouterr().err
    assert "WARNING" in stderr and "FAILING" in stderr and "mrr_at_8" in stderr


def test_passing_run_records_the_baseline_with_its_label_set_hash(tmp_path, monkeypatch):
    baseline = tmp_path / "reference.json"
    monkeypatch.setattr(
        evaluate_search, "evaluate", lambda *a, **k: _canned_result(gate_passed=True)
    )

    assert (
        evaluate_search.main(
            [str(tmp_path / "labels.json"), "--baseline", str(baseline), "--write-baseline"]
        )
        == 0
    )
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["label_set_sha256"] == "abc"
    assert payload["floors"] == DEFAULT_FLOORS


def test_recording_a_baseline_preserves_the_configured_floors(tmp_path):
    baseline = tmp_path / "reference.json"
    baseline.write_text(
        json.dumps({"metrics": {}, "floors": {"mrr_at_8": 0.9}}), encoding="utf-8"
    )

    _write_reference(baseline, _canned_result(gate_passed=True))

    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["floors"] == {"mrr_at_8": 0.9}, "recording a measurement must not drop the bar"
