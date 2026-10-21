"""Streamlit client for the customer knowledge FastAPI backend."""
import os
import hashlib
import json
import time
from pathlib import Path
import streamlit as st
from frontend.client import BackendClient, BackendError
from frontend.customer_scope import resolve_customer_id
from frontend.dashboard_metrics import summarize_chat_session


st.set_page_config(page_title="Customer Knowledge Assistant", page_icon="📚", layout="wide")
st.title("Customer Knowledge Assistant")
st.caption("Streamlit · FastAPI · Ollama | Customer answers grounded in your source records")

with st.sidebar:
    st.header("Connection")
    base_url = os.environ.get("KA_API_URL", "http://127.0.0.1:8000")
    st.caption(f"FastAPI: {base_url}")
    token = os.environ.get("KA_FRONTEND_TOKEN", "")
    default_user_id = os.environ.get("KA_FRONTEND_CUSTOMER", "U0000000001")
    try:
        allowed_customers = json.loads(os.environ.get("KA_FRONTEND_CUSTOMERS", "[]"))
    except ValueError:
        allowed_customers = []
    if allowed_customers:
        selected = default_user_id if default_user_id in allowed_customers else allowed_customers[0]
        user_id = st.selectbox("Customer ID", options=allowed_customers, index=allowed_customers.index(selected))
        st.caption("Choose a customer included in this limited index.")
    else:
        user_id = st.text_input("Customer ID", value=default_user_id, max_chars=11)
    st.caption("Local model: " + os.environ.get("KA_MODEL_ID", "ollama:qwen3:4b-instruct"))
    if st.button("Check backend", disabled=not token):
        try:
            status = BackendClient(base_url, token, timeout=5).health()
            st.success("Backend ready")
            st.json(status)
        except BackendError as error:
            st.error(str(error))
    st.link_button("FastAPI documentation", base_url + "/docs")

# Keep one chat across customer questions; reset it if the credential changes.
scope = hashlib.sha256(token.encode()).hexdigest()
if st.session_state.get("scope") != scope:
    st.session_state.scope = scope
    st.session_state.messages = []

client = BackendClient(base_url, token)
st.caption(f"Selected customer: {user_id}")
chat_tab, summary_tab, revenue_tab, security_tab = st.tabs(["Ask assistant", "Customer details", "Revenue", "Security & accuracy"])


def render_answer(result, user_id):
    st.markdown(f"**Customer ID:** `{user_id}`")
    st.text(result.get("answer", "No answer returned."))
    st.caption("Status: " + result.get("status", "unknown"))
    if result.get("evidence"):
        with st.expander("Source evidence", expanded=False):
            for item in result["evidence"]:
                st.write("**Customer:** " + item["user_id"])
                st.code(", ".join(item["source_fields"]), language=None)
                st.text(item["explanation"])
    if result.get("limitations"):
        with st.expander("Limitations"):
            for limitation in result["limitations"]:
                st.text(limitation)


with chat_tab:
    if allowed_customers:
        st.info("Ask about any indexed customer by including their ID, such as “Summarize subscription and payments for U0000000010.” Without an ID, the selected sidebar customer is used.")
    else:
        st.info("All customer records are loaded. Ask about any customer by including their ID, such as “Summarize subscription and payments for U0000000010.” The selected sidebar ID is used when your question has no ID.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.text(message["content"])
            else:
                render_answer(message["content"], message.get("user_id", user_id))
    question = st.chat_input("Ask about any customer… include their ID", max_chars=2000, disabled=not token)
    if not token:
        st.caption("Start the local stack with `python -m scripts.run_local` to connect securely.")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.text(question)
        try:
            request_user_id = resolve_customer_id(question, user_id, allowed_customers)
        except ValueError as error:
            st.error(str(error))
        else:
            with st.chat_message("assistant"):
                try:
                    with st.spinner(f"Checking evidence for {request_user_id}…"):
                        started = time.perf_counter()
                        answer = client.chat(request_user_id, question)
                        latency_ms = (time.perf_counter() - started) * 1000
                    render_answer(answer, request_user_id)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "user_id": request_user_id, "latency_ms": latency_ms})
                except BackendError as error:
                    st.error(str(error))

with summary_tab:
    st.subheader(f"Authorized customer record · {user_id}")
    if st.button("Load customer summary", disabled=not token):
        try:
            summary = client.summary(user_id)
            for name, domain in summary["domains"].items():
                with st.expander(name.title(), expanded=name == "subscription"):
                    st.json(domain["values"])
                    for limitation in domain.get("limitations", []):
                        st.caption(limitation)
        except BackendError as error:
            st.error(str(error))

with revenue_tab:
    st.subheader(f"Verified monthly settled revenue · {user_id}")
    if st.button("Load revenue", disabled=not token):
        try:
            result = client.revenue(user_id)
            st.metric("Total settled revenue (paise)", result["total_paise"] if result["total_paise"] is not None else "Unavailable")
            st.caption("Full 12-month period" if result["complete"] else "Incomplete period — total covers available months only")
            st.dataframe([{"Month": month, "Settled revenue (paise)": value} for month, value in result["monthly_paise"].items()], hide_index=True)
            st.caption("Source fields: " + ", ".join(result["source_fields"]))
        except BackendError as error:
            st.error(str(error))

with security_tab:
    st.subheader("Live quality and performance")
    metrics = summarize_chat_session(st.session_state.messages)
    left, middle, right = st.columns(3)
    left.metric("Chat answers this session", metrics["answer_count"])
    middle.metric("Answered with source evidence", f"{metrics['evidence_count']} / {metrics['answered_count']}" if metrics["answered_count"] else "No answered requests")
    right.metric("Evidence coverage", f"{metrics['evidence_rate']:.0%}" if metrics["evidence_rate"] is not None else "Not measured")
    st.caption("Evidence coverage checks whether an answer marked ‘answered’ includes source evidence. It is a grounding signal, not a factual accuracy score.")

    st.markdown("**Observed API response time**")
    if metrics["latency_count"]:
        first, second, third = st.columns(3)
        first.metric("Average", f"{metrics['average_latency_ms']:.0f} ms")
        second.metric("Median", f"{metrics['median_latency_ms']:.0f} ms")
        third.metric("P95", f"{metrics['p95_latency_ms']:.0f} ms")
        st.caption(f"Based on {metrics['latency_count']} successful chat request(s) in this browser session. Includes backend/model time and local network overhead; excludes failed requests and page reloads.")
    else:
        st.info("Ask a question in the chat to measure response speed. Timings are collected for this browser session only.")

    statuses = metrics["statuses"]
    st.markdown("**Answer outcomes**")
    st.dataframe(
        [{"Outcome": name.replace("_", " ").title(), "Count": count} for name, count in sorted(statuses.items())]
        or [{"Outcome": "No requests yet", "Count": 0}],
        hide_index=True,
        use_container_width=True,
    )
    st.caption("A reliable factual accuracy percentage requires questions with reviewed, expected answers. This dashboard does not invent such a score; use the evaluation workflow for labeled tests.")

    configured_report = os.environ.get("KA_EVALUATION_REPORT")
    report_path = Path(configured_report) if configured_report else Path("output/reports/live_evaluation.json")
    if not configured_report and not report_path.is_file():
        report_path = Path("output/reports/evaluation.json")
    st.markdown("**Latest labeled answer evaluation**")
    try:
        evaluation = json.loads(report_path.read_text()) if report_path.is_file() else None
    except (OSError, json.JSONDecodeError):
        evaluation = None
    if evaluation and evaluation.get("metrics", {}).get("case_count", 0):
        eval_metrics = evaluation["metrics"]
        accuracy = eval_metrics.get("case_accuracy")
        eval_cols = st.columns(3)
        eval_cols[0].metric("Labeled case accuracy", f"{accuracy:.1%}" if accuracy is not None else "Not measured")
        eval_cols[1].metric("Expected-field recall", f"{eval_metrics['expected_field_recall']:.1%}" if eval_metrics.get("expected_field_recall") is not None else "Not labeled")
        eval_cols[2].metric("Expected-answer markers", f"{eval_metrics['expected_answer_marker_accuracy']:.1%}" if eval_metrics.get("expected_answer_marker_accuracy") is not None else "Not labeled")
        evaluated_at = evaluation.get("evaluated_at", "time unavailable")
        st.caption(f"{evaluation.get('mode', 'unknown')} · {evaluation.get('provider', 'provider unavailable')} · {eval_metrics['case_count']} reviewed case(s) · {evaluated_at}")
        failed_cases = [row for row in evaluation.get("cases", []) if not row.get("correct", False)]
        if failed_cases:
            st.dataframe(
                [{"Case": row.get("id"), "Outcome": row.get("status"), "Expected fields": f"{row.get('matched_field_count', 0)} / {row.get('expected_field_count', 0)}", "Expected answer checks": f"{row.get('matched_answer_marker_count', 0)} / {row.get('expected_answer_marker_count', 0)}"} for row in failed_cases],
                hide_index=True,
                use_container_width=True,
            )
        st.caption("Accuracy applies only to the labeled cases in this report. Synthetic mode uses a deterministic demo; run live mode to measure the configured API and model.")
    else:
        st.info(f"No labeled evaluation report found at `{report_path}`. Run the evaluator from a terminal to create one.")
    st.code("python -m scripts.evaluate --mode live --cases evaluation/my_live_cases.jsonl --output output/reports/live_evaluation.json", language="bash")

    st.divider()
    st.subheader("Security and backend status")
    if st.button("Refresh security diagnostics", disabled=not token):
        try:
            st.session_state.security_health = client.health()
        except BackendError as error:
            st.session_state.security_health = {"error": str(error)}
    health = st.session_state.get("security_health")
    if health is None:
        st.info("Select ‘Refresh security diagnostics’ to check API, model, and index readiness.")
    elif "error" in health:
        st.error(health["error"])
    else:
        status_col, mode_col, telemetry_col = st.columns(3)
        status_col.metric("API readiness", health.get("status", "unknown").replace("_", " ").title())
        mode_col.metric("Backend mode", health.get("mode", "unknown").title())
        telemetry_col.metric("Telemetry", health.get("telemetry", "unknown").title())
        st.dataframe(
            [{"Component": name.title(), "Status": value.replace("_", " ").title()} for name, value in health.get("components", {}).items()],
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("**Access scope**")
    st.write("Frontend customer allowlist configured" if allowed_customers else "No frontend customer allowlist configured")
    st.caption(f"{len(allowed_customers)} customer ID(s) in the configured allowlist." if allowed_customers else "The API validates customer IDs against the loaded source data. Access also depends on the configured bearer token and backend deployment settings.")
    st.caption("Security indicators describe application configuration and readiness; they are not a penetration test or a guarantee of production security.")

    st.divider()
    st.subheader("Langfuse tracing")
    st.caption("Optional Langfuse evaluation scores output-claim validation, answer outcome, source-evidence presence, request-stage latency, model token usage, and agent tool calls. Customer questions, prompts, answers, and raw records are excluded.")
    if st.button("Check Langfuse connection", disabled=not token):
        try:
            st.session_state.langfuse_status = client.langfuse_status()
        except BackendError as error:
            st.session_state.langfuse_status = {"status": "check_failed", "message": str(error)}
    langfuse_status = st.session_state.get("langfuse_status")
    if langfuse_status:
        state = langfuse_status.get("status", "unknown")
        if state == "connected":
            st.success(langfuse_status.get("message", "Langfuse is connected."))
        elif state == "disabled":
            st.info(langfuse_status.get("message", "Langfuse is disabled."))
        else:
            st.warning(langfuse_status.get("message", "Langfuse connection could not be verified."))
    st.caption("Configure KA_LANGFUSE_PUBLIC_KEY and KA_LANGFUSE_SECRET_KEY in the backend environment, then restart FastAPI. Do not enter project keys in this page.")
    st.link_button("Open Langfuse", os.environ.get("KA_LANGFUSE_BASE_URL", "https://cloud.langfuse.com"))
