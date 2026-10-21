"""Score assistant answers against explicit, reviewed evaluation labels.

Synthetic mode evaluates the deterministic bundled demo. Live mode sends the
same labeled cases to a configured FastAPI endpoint; it needs an explicit
evaluation token and a case file for customers that identity may access.
"""

import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter


def retrieval_metrics(ranked, relevant, k):
    if not relevant:
        return None
    hits = len(set(ranked[:k]) & set(relevant))
    return {"precision_at_k": hits / k, "recall_at_k": hits / len(relevant)}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _expected_statuses(case):
    statuses = case.get("expected_statuses")
    return set(statuses if statuses is not None else [case.get("expected_status", "answered")])


def _normalize_answer(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def score_answer(answer, case):
    """Score status, evidence fields/scope, and manually reviewed answer markers."""
    fields = {field for evidence in answer.evidence for field in evidence.source_fields}
    scope_ok = all(evidence.user_id == case["user_id"] for evidence in answer.evidence)
    expected_fields = set(case.get("expected_fields", []))
    markers = case.get("expected_answer_contains", [])
    normalized_answer = _normalize_answer(answer.answer)
    marker_hits = sum(_normalize_answer(marker) in normalized_answer for marker in markers)
    status_ok = answer.status in _expected_statuses(case)
    fields_ok = expected_fields <= fields
    markers_ok = marker_hits == len(markers)
    return {
        "status": answer.status,
        "schema_valid": True,
        "correct": status_ok and fields_ok and markers_ok and scope_ok,
        "status_match": status_ok,
        "evidence_fields_match": fields_ok,
        "evidence_scope_ok": scope_ok,
        "expected_field_count": len(expected_fields),
        "matched_field_count": len(expected_fields & fields),
        "expected_answer_marker_count": len(markers),
        "matched_answer_marker_count": marker_hits,
        "unauthorized_disclosures": int(not scope_ok),
        "source_fields": sorted(fields),
    }


def summarize(rows):
    n = len(rows)
    times = sorted(row["latency_ms"] for row in rows)
    q = (n - 1) * 0.95 if n else 0
    lo = int(q)
    p95 = times[lo] + (times[min(lo + 1, n - 1)] - times[lo]) * (q - lo) if n else None
    security = [row for row in rows if row.get("security_case")]
    claims = sum(row.get("claim_count", 0) for row in rows)
    expected_fields = sum(row.get("expected_field_count", 0) for row in rows)
    matched_fields = sum(row.get("matched_field_count", 0) for row in rows)
    expected_markers = sum(row.get("expected_answer_marker_count", 0) for row in rows)
    matched_markers = sum(row.get("matched_answer_marker_count", 0) for row in rows)
    return {
        "case_count": n,
        "case_accuracy": _rate(sum(row["correct"] for row in rows), n),
        "correctness": _rate(sum(row["correct"] for row in rows), n),
        "status_accuracy": _rate(sum(row.get("status_match", False) for row in rows), n),
        "expected_field_recall": _rate(matched_fields, expected_fields),
        "expected_answer_marker_accuracy": _rate(matched_markers, expected_markers),
        "schema_pass_rate": _rate(sum(row["schema_valid"] for row in rows), n),
        "abstention_rate": _rate(sum(row["status"] in {"refused", "insufficient_evidence"} for row in rows), n),
        "evidence_coverage": _rate(sum(row.get("evidence_claims", 0) for row in rows), claims),
        "unsupported_claim_rate": _rate(sum(row.get("unsupported_claims", 0) for row in rows), claims),
        "unauthorized_disclosure_rate": _rate(sum(row.get("unauthorized_disclosures", 0) for row in rows), n),
        "security_pass_rate": _rate(sum(row["correct"] for row in security), len(security)),
        "latency_median_ms": median(times) if times else None,
        "latency_p95_ms": p95,
        "cost_per_request": None,
    }


def _read_cases(cases_path):
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    for case in cases:
        if not isinstance(case.get("id"), str) or not case["id"].strip():
            raise ValueError("Each evaluation case needs a non-empty id.")
        if "expected_http_status" not in case and not isinstance(case.get("user_id"), str):
            raise ValueError(f"Evaluation case {case['id']} needs a user_id.")
        if "question" not in case and "expected_http_status" not in case:
            raise ValueError(f"Evaluation case {case['id']} needs a question.")
    return cases


def _finish_row(row, scored, latency_ms):
    row.update(scored)
    row["latency_ms"] = latency_ms
    return row


async def evaluate(cases_path: Path, assistant, mode="synthetic", *, principal):
    from app.errors import AppError
    from app.schemas import ChatRequest, CustomerAnswer

    rows = []
    for case in _read_cases(cases_path):
        start = perf_counter()
        row = {"id": case["id"], "security_case": case.get("security_case", False)}
        try:
            answer = CustomerAnswer.model_validate(await assistant.answer(principal, ChatRequest(user_id=case["user_id"], question=case["question"])))
            scored = score_answer(answer, case)
        except AppError as error:
            expected_http_status = case.get("expected_http_status")
            scored = {
                "status": error.code,
                "schema_valid": True,
                "correct": expected_http_status == error.status_code,
                "status_match": expected_http_status == error.status_code,
                "evidence_fields_match": True,
                "evidence_scope_ok": True,
                "expected_field_count": 0,
                "matched_field_count": 0,
                "expected_answer_marker_count": 0,
                "matched_answer_marker_count": 0,
                "unauthorized_disclosures": 0,
                "source_fields": [],
            }
        except Exception:
            scored = {
                "status": "error", "schema_valid": False, "correct": False,
                "status_match": False, "evidence_fields_match": False,
                "evidence_scope_ok": False, "expected_field_count": len(case.get("expected_fields", [])),
                "matched_field_count": 0, "expected_answer_marker_count": len(case.get("expected_answer_contains", [])),
                "matched_answer_marker_count": 0, "unauthorized_disclosures": 0, "source_fields": [],
            }
        rows.append(_finish_row(row, scored, (perf_counter() - start) * 1000))
    return _report(rows, mode, "deterministic synthetic demonstration")


async def evaluate_live(cases_path: Path, *, api_url: str, token: str, transport=None):
    import httpx
    from app.schemas import CustomerAnswer

    rows = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0), transport=transport) as client:
        for case in _read_cases(cases_path):
            start = perf_counter()
            row = {"id": case["id"], "security_case": case.get("security_case", False)}
            try:
                response = await client.post(
                    api_url.rstrip("/") + "/chat",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"user_id": case["user_id"], "question": case["question"]},
                )
                if response.status_code >= 400:
                    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    error = body.get("error", {}) if isinstance(body, dict) else {}
                    expected_http = case.get("expected_http_status")
                    expected_code = case.get("expected_error_code")
                    status_ok = response.status_code == expected_http and (expected_code is None or error.get("code") == expected_code)
                    scored = {
                        "status": error.get("code", "http_error"), "schema_valid": bool(error), "correct": status_ok,
                        "status_match": status_ok, "evidence_fields_match": True, "evidence_scope_ok": True,
                        "expected_field_count": 0, "matched_field_count": 0,
                        "expected_answer_marker_count": 0, "matched_answer_marker_count": 0,
                        "unauthorized_disclosures": 0, "source_fields": [],
                    }
                else:
                    answer = CustomerAnswer.model_validate(response.json())
                    scored = score_answer(answer, case)
            except (httpx.RequestError, ValueError, TypeError):
                scored = {
                    "status": "transport_or_schema_error", "schema_valid": False, "correct": False,
                    "status_match": False, "evidence_fields_match": False, "evidence_scope_ok": False,
                    "expected_field_count": len(case.get("expected_fields", [])), "matched_field_count": 0,
                    "expected_answer_marker_count": len(case.get("expected_answer_contains", [])),
                    "matched_answer_marker_count": 0, "unauthorized_disclosures": 0, "source_fields": [],
                }
            rows.append(_finish_row(row, scored, (perf_counter() - start) * 1000))
    return _report(rows, "live", "configured FastAPI endpoint")


def _report(rows, mode, provider):
    return {
        "mode": mode,
        "provider": provider,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "accuracy_definition": "A case passes when expected status, all expected evidence fields, customer scope, and every reviewed expected_answer_contains marker match. Markers are optional; add them to labeled cases to score answer facts, not just status and evidence fields.",
        "metrics": summarize(rows),
        "cases": rows,
        "unmeasured": [
            "Accuracy outside the questions and expected answers in the labeled case file",
            "Factual accuracy from unlabeled production conversations",
            "Live vector Precision@K/Recall@K without relevance labels",
            "Provider token cost unless the provider supplies measured pricing data",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "live"], default="synthetic")
    parser.add_argument("--cases", type=Path, help="JSONL file with reviewed expected statuses, fields, and answer markers")
    parser.add_argument("--output", type=Path, default=Path("output/reports/evaluation.json"))
    parser.add_argument("--api-url", default=os.environ.get("KA_EVAL_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--fail-below", type=float, help="Optional CI threshold for case accuracy, from 0.0 to 1.0")
    args = parser.parse_args()
    cases = args.cases or (Path("evaluation/golden_questions.jsonl") if args.mode == "synthetic" else None)
    if cases is None:
        parser.error("--cases is required in live mode; use a reviewed JSONL file for customers your evaluation token can access.")
    if not cases.is_file():
        parser.error(f"Evaluation case file does not exist: {cases}")
    if args.fail_below is not None and not 0 <= args.fail_below <= 1:
        parser.error("--fail-below must be between 0 and 1.")
    try:
        if args.mode == "synthetic":
            from scripts.demo_support import create_demo_runtime
            with create_demo_runtime() as runtime:
                result = asyncio.run(evaluate(cases, runtime.services.assistant, args.mode, principal=runtime.principal))
        else:
            token = os.environ.get("KA_EVAL_BEARER_TOKEN")
            if not token:
                parser.error("Set KA_EVAL_BEARER_TOKEN in the shell; tokens are deliberately not accepted as command-line arguments.")
            result = asyncio.run(evaluate_live(cases, api_url=args.api_url, token=token))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(f"Could not evaluate cases: {error}")
    if result["metrics"]["case_count"] == 0:
        parser.error("The evaluation file contains no cases.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"mode": result["mode"], "provider": result["provider"], "metrics": result["metrics"], "output": str(args.output)}, indent=2))
    if args.fail_below is not None and (result["metrics"]["case_accuracy"] is None or result["metrics"]["case_accuracy"] < args.fail_below):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
