"""Measured evaluation of explicit synthetic or configured live runs."""

import argparse
import asyncio
import json
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


def summarize(rows):
    n = len(rows)
    times = sorted(row["latency_ms"] for row in rows)
    q = (n - 1) * 0.95 if n else 0
    lo = int(q)
    p95 = times[lo] + (times[min(lo + 1, n - 1)] - times[lo]) * (q - lo) if n else None
    security = [row for row in rows if row.get("security_case")]
    claims = sum(row.get("claim_count", 0) for row in rows)
    return {
        "case_count": n,
        "correctness": _rate(sum(row["correct"] for row in rows), n),
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


async def evaluate(cases_path: Path, assistant, mode="synthetic", *, principal):
    from app.errors import AppError
    from app.schemas import ChatRequest, CustomerAnswer

    rows = []
    for line in cases_path.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        start = perf_counter()
        row = {"id": case["id"], "security_case": case.get("security_case", False)}
        try:
            answer = CustomerAnswer.model_validate(await assistant.answer(principal, ChatRequest(user_id=case["user_id"], question=case["question"])))
            fields = {field for evidence in answer.evidence for field in evidence.source_fields}
            scoped = all(evidence.user_id == case["user_id"] for evidence in answer.evidence)
            row.update(status=answer.status, schema_valid=True, correct=answer.status in case.get("expected_statuses", [case.get("expected_status", "answered")]) and set(case.get("expected_fields", [])) <= fields and scoped, unauthorized_disclosures=int(not scoped), source_fields=sorted(fields))
        except AppError as error:
            row.update(status=error.code, schema_valid=True, correct=error.status_code == case.get("expected_http_status"), unauthorized_disclosures=0)
        except Exception:
            row.update(status="error", schema_valid=False, correct=False, unauthorized_disclosures=0)
        row["latency_ms"] = (perf_counter() - start) * 1000
        rows.append(row)
    return {"mode": mode, "provider": "deterministic synthetic demonstration" if mode == "synthetic" else "configured live provider", "metrics": summarize(rows), "cases": rows, "unmeasured": ["Live quality in synthetic mode", "Claim-level unsupported rate without independent claim labels", "Live vector Precision@K/Recall@K without relevance labels", "Provider token cost", "Prompt baseline comparison unless separately run"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "live"], default="synthetic")
    parser.add_argument("--cases", type=Path, default=Path("evaluation/golden_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/reports/evaluation.json"))
    args = parser.parse_args()
    if args.mode == "live":
        parser.error("Use the configured API/live integration tests; this command currently supports synthetic evaluation only.")
    from scripts.demo_support import create_demo_runtime
    with create_demo_runtime() as runtime:
        result = asyncio.run(evaluate(args.cases, runtime.services.assistant, args.mode, principal=runtime.principal))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
    raise SystemExit(0 if result["metrics"]["correctness"] == 1 else 1)


if __name__ == "__main__":
    main()
