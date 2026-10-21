"""Evaluation math is checked independently of any model provider."""

import pytest

from scripts.evaluate import retrieval_metrics, summarize


def test_rank_metrics_use_labels_and_actual_cutoff():
    metrics = retrieval_metrics(["a", "b", "c"], {"b", "d"}, 2)
    assert metrics == {"precision_at_k": 0.5, "recall_at_k": 0.5}


def test_unlabeled_retrieval_is_unmeasured():
    assert retrieval_metrics(["a"], None, 3) is None
    assert retrieval_metrics([], set(), 3) is None


def test_repeated_retrieval_cannot_inflate_recall():
    assert retrieval_metrics(["a", "a"], {"a", "b"}, 2)["recall_at_k"] == 0.5


def test_summary_denominators_and_latency():
    rows = [
        {"correct": True, "schema_valid": True, "status": "answered", "latency_ms": 10, "claim_count": 2, "unsupported_claims": 0, "evidence_claims": 2, "unauthorized_disclosures": 0, "security_case": False},
        {"correct": True, "schema_valid": True, "status": "refused", "latency_ms": 30, "claim_count": 0, "unsupported_claims": 0, "evidence_claims": 0, "unauthorized_disclosures": 0, "security_case": True},
    ]
    report = summarize(rows)
    assert report["correctness"] == 1
    assert report["schema_pass_rate"] == 1
    assert report["abstention_rate"] == 0.5
    assert report["evidence_coverage"] == 1
    assert report["unsupported_claim_rate"] == 0
    assert report["unauthorized_disclosure_rate"] == 0
    assert report["security_pass_rate"] == 1
    assert report["latency_median_ms"] == 20
    assert report["latency_p95_ms"] == pytest.approx(29)
    assert report["cost_per_request"] is None


def test_empty_evaluation_is_not_a_passing_run():
    result = summarize([])
    assert result["case_count"] == 0
    assert result["correctness"] is None
    assert result["security_pass_rate"] is None
