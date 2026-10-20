"""Boundary contracts reject untrusted inputs and unchecked model prose."""

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.schemas import AgentResult, ChatRequest, Claim, CustomerAnswer, Evidence


@pytest.mark.parametrize("user_id", ["U1", "U00000000001", "u0000000001", "U000000000١", "U0000000001 OR 1=1", "U0000000001\n"])
def test_chat_rejects_malformed_user_ids(user_id):
    with pytest.raises(ValidationError):
        ChatRequest(user_id=user_id, question="Summarize this customer")


@pytest.mark.parametrize("question", ["", "  \t\n ", "x" * 2001])
def test_chat_rejects_empty_or_oversized_question(question):
    with pytest.raises(ValidationError):
        ChatRequest(user_id="U0000000001", question=question)


def test_chat_accepts_question_boundary_and_rejects_client_permissions():
    assert len(ChatRequest(user_id="U0000000001", question="x" * 2000).question) == 2000
    with pytest.raises(ValidationError):
        ChatRequest(user_id="U0000000001", question="summary", customer_ids=["U0000000002"])


def test_answered_requires_evidence_and_nonempty_answer():
    with pytest.raises(ValidationError):
        CustomerAnswer(status="answered", answer="Invented information", evidence=[])
    evidence = Evidence(user_id="U0000000001", source_fields=["AccountStatus"], explanation="Source record")
    with pytest.raises(ValidationError):
        CustomerAnswer(status="answered", answer=" ", evidence=[evidence])
    assert CustomerAnswer(status="answered", answer="AccountStatus: active", evidence=[evidence]).evidence == [evidence]


@pytest.mark.parametrize("fields", [[], [""], ["  "]])
def test_evidence_requires_nonempty_source_fields(fields):
    with pytest.raises(ValidationError):
        Evidence(user_id="U0000000001", source_fields=fields, explanation="Source record")


def test_abstention_does_not_require_evidence():
    assert CustomerAnswer(status="insufficient_evidence", answer="No relevant evidence.").evidence == []


def test_all_public_models_forbid_unknown_fields():
    examples = [
        (Evidence, dict(user_id="U0000000001", source_fields=["AccountStatus"], explanation="source")),
        (CustomerAnswer, dict(status="refused", answer="Request refused.")),
        (Claim, dict(user_id="U0000000001", source_field="AccountStatus", value="active")),
        (AgentResult, dict(status="insufficient_evidence", claims=[])),
    ]
    for model, payload in examples:
        with pytest.raises(ValidationError):
            model(**payload, answer_prose="Invented facts")


def test_claim_requires_exactly_one_reference_and_preserves_integer_precision():
    value = 9007199254740993
    assert Claim(user_id="U0000000001", source_field="NetRevenuePaise", value=value, unit="paise").value == value
    for references in [{}, {"source_field": "NetRevenuePaise", "metric_id": "monthly_settled_total_paise"}]:
        with pytest.raises(ValidationError):
            Claim(user_id="U0000000001", value=1, **references)


@pytest.mark.parametrize("value", [None, {"text": "invented"}, [1], float("nan"), float("inf")])
def test_claim_rejects_missing_structured_or_nonfinite_value(value):
    with pytest.raises(ValidationError):
        Claim(user_id="U0000000001", source_field="NetRevenuePaise", value=value)


def test_agent_rejects_unapproved_metric_and_free_form_limitations():
    with pytest.raises(ValidationError):
        Claim(user_id="U0000000001", metric_id="invented_total", value=1)
    with pytest.raises(ValidationError):
        AgentResult(status="insufficient_evidence", limitations=["Customer is definitely fraudulent"])


@pytest.mark.parametrize("limits", [{"top_k": 0}, {"max_documents": 101}, {"context_max_chars": 0}, {"question_max_chars": 2001}, {"request_timeout_seconds": -1}, {"model_request_limit": 0}, {"tool_call_limit": 0}, {"ingestion_batch_size": 0}, {"relevance_threshold": float("nan")}])
def test_settings_reject_unbounded_or_invalid_limits(credential_mapping, limits):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, credential_mapping=credential_mapping, **limits)


def test_settings_read_explicit_environment_mapping(monkeypatch):
    monkeypatch.setenv("KA_CREDENTIAL_MAPPING", '{"synthetic-token":{"subject":"analyst","customer_ids":["U0000000001"],"ingest_admin":false}}')
    assert Settings(_env_file=None).credential_mapping["synthetic-token"].subject == "analyst"
