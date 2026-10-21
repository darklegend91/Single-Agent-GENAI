from frontend.dashboard_metrics import summarize_chat_session


def test_dashboard_metrics_count_answer_status_evidence_and_latency():
    messages = [
        {"role": "user", "content": "summary"},
        {"role": "assistant", "content": {"status": "answered", "evidence": [{"source_fields": ["plan"]}]}, "latency_ms": 100},
        {"role": "assistant", "content": {"status": "insufficient_evidence", "evidence": []}, "latency_ms": 300},
        {"role": "assistant", "content": {"status": "answered", "evidence": []}, "latency_ms": 200},
    ]

    result = summarize_chat_session(messages)

    assert result["answer_count"] == 3
    assert result["statuses"] == {"answered": 2, "insufficient_evidence": 1}
    assert result["answered_count"] == 2
    assert result["evidence_count"] == 1
    assert result["evidence_rate"] == 0.5
    assert result["average_latency_ms"] == 200
    assert result["median_latency_ms"] == 200
    assert result["p95_latency_ms"] == 300


def test_dashboard_metrics_empty_session_has_no_invented_rates_or_timings():
    result = summarize_chat_session([])

    assert result["answer_count"] == 0
    assert result["evidence_rate"] is None
    assert result["average_latency_ms"] is None
    assert result["median_latency_ms"] is None
    assert result["p95_latency_ms"] is None
