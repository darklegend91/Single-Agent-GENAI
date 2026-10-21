"""Evaluation math is checked independently of any model provider."""

import pytest
import asyncio
import json
import httpx

from app.schemas import CustomerAnswer, Evidence
from scripts.evaluate import evaluate_live, retrieval_metrics, score_answer, summarize


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


def test_answer_accuracy_checks_status_fields_scope_and_reviewed_values():
    answer = CustomerAnswer(
        status="answered",
        answer="Successful Payment Count: 4.\nFailed Payment Count: 2.",
        evidence=[Evidence(user_id="U0000000001", source_fields=["SuccessfulPaymentCount", "FailedPaymentCount"], explanation="Verified.")],
    )
    case = {
        "user_id": "U0000000001",
        "expected_status": "answered",
        "expected_fields": ["SuccessfulPaymentCount", "FailedPaymentCount"],
        "expected_answer_contains": ["Successful Payment Count: 4", "Failed Payment Count: 2"],
    }
    result = score_answer(answer, case)
    assert result["correct"] is True
    assert result["matched_answer_marker_count"] == 2


def test_wrong_answer_value_or_customer_evidence_fails_accuracy():
    answer = CustomerAnswer(
        status="answered",
        answer="Payment Attempt Count: 7.",
        evidence=[Evidence(user_id="U0000000002", source_fields=["PaymentAttemptCount"], explanation="Evidence.")],
    )
    case = {
        "user_id": "U0000000001",
        "expected_status": "answered",
        "expected_fields": ["PaymentAttemptCount"],
        "expected_answer_contains": ["Payment Attempt Count: 6"],
    }
    result = score_answer(answer, case)
    assert result["status_match"] is True
    assert result["evidence_fields_match"] is True
    assert result["evidence_scope_ok"] is False
    assert result["correct"] is False
    assert result["unauthorized_disclosures"] == 1


def test_summary_reports_labeled_fact_and_field_accuracy():
    row = {
        "correct": True, "schema_valid": True, "status": "answered", "latency_ms": 12,
        "status_match": True, "expected_field_count": 2, "matched_field_count": 2,
        "expected_answer_marker_count": 2, "matched_answer_marker_count": 2,
        "unauthorized_disclosures": 0, "security_case": False,
    }
    result = summarize([row])
    assert result["case_accuracy"] == 1
    assert result["status_accuracy"] == 1
    assert result["expected_field_recall"] == 1
    assert result["expected_answer_marker_accuracy"] == 1


def test_live_evaluator_scores_api_response_without_saving_answer_text(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({
        "id": "plan", "user_id": "U0000000001", "question": "Plan name?",
        "expected_status": "answered", "expected_fields": ["PlanName"],
        "expected_answer_contains": ["Plan Name: STANDARD"],
    }) + "\n")

    def handler(request):
        assert request.url.path == "/chat"
        assert request.headers["authorization"] == "Bearer test-token"
        return httpx.Response(200, json={
            "status": "answered", "answer": "Plan Name: STANDARD.",
            "evidence": [{"user_id": "U0000000001", "source_fields": ["PlanName"], "explanation": "Verified."}],
            "limitations": [],
        })

    result = asyncio.run(evaluate_live(cases, api_url="http://test", token="test-token", transport=httpx.MockTransport(handler)))
    assert result["metrics"]["case_accuracy"] == 1
    assert result["metrics"]["expected_answer_marker_accuracy"] == 1
    assert result["cases"][0]["correct"] is True
    assert "answer" not in result["cases"][0]
