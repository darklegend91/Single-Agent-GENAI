"""Session metrics for the Streamlit security and quality dashboard."""
from math import ceil, isfinite
from statistics import mean, median


def summarize_chat_session(messages):
    """Summarize observed assistant replies without implying factual accuracy."""
    answers = [
        message.get("content")
        for message in messages
        if message.get("role") == "assistant" and isinstance(message.get("content"), dict)
    ]
    statuses = {}
    for answer in answers:
        status = answer.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1

    answered = [answer for answer in answers if answer.get("status") == "answered"]
    grounded = [answer for answer in answered if answer.get("evidence")]
    latency_values = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        latency = message.get("latency_ms")
        if isinstance(latency, (int, float)) and isfinite(latency) and latency >= 0:
            latency_values.append(float(latency))

    ordered = sorted(latency_values)
    p95 = ordered[max(0, ceil(0.95 * len(ordered)) - 1)] if ordered else None
    return {
        "answer_count": len(answers),
        "statuses": statuses,
        "answered_count": len(answered),
        "evidence_count": len(grounded),
        "evidence_rate": len(grounded) / len(answered) if answered else None,
        "latency_count": len(ordered),
        "average_latency_ms": mean(ordered) if ordered else None,
        "median_latency_ms": median(ordered) if ordered else None,
        "p95_latency_ms": p95,
    }
