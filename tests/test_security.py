"""Authorization comes only from configured immutable server identities."""

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.errors import AuthenticationError, AuthorizationError
from app.security import Principal, authenticate, authorize


def test_configured_credential_authenticates_with_immutable_scope(credential_mapping):
    settings = Settings(_env_file=None, credential_mapping=credential_mapping)
    principal = authenticate("synthetic-customer-token", settings)
    assert principal.subject == "analyst"
    assert principal.customer_ids == frozenset({"U0000000001"})
    assert not principal.ingest_admin
    with pytest.raises((ValidationError, AttributeError)):
        principal.ingest_admin = True


@pytest.mark.parametrize("token", ["", "unknown", "synthetic-customer-token ", "synthetic-customer-tokén", "\ud800"])
def test_unknown_credentials_fail_closed(credential_mapping, token):
    settings = Settings(_env_file=None, credential_mapping=credential_mapping)
    with pytest.raises(AuthenticationError) as caught:
        authenticate(token, settings)
    assert caught.value.status_code == 401
    assert token not in str(caught.value) if token else True


def test_customer_scope_is_checked_even_for_ingestion_admin(credential_mapping):
    settings = Settings(_env_file=None, credential_mapping=credential_mapping)
    customer = authenticate("synthetic-customer-token", settings)
    authorize(customer, "U0000000001")
    for principal in [customer, authenticate("synthetic-admin-token", settings)]:
        with pytest.raises(AuthorizationError) as caught:
            authorize(principal, "U0000000002")
        assert caught.value.status_code == 403


def test_missing_and_empty_credential_mapping_cannot_start(monkeypatch):
    monkeypatch.delenv("KA_CREDENTIAL_MAPPING", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, credential_mapping={})


def test_credentials_and_provider_secrets_are_redacted_from_repr(credential_mapping):
    settings = Settings(_env_file=None, credential_mapping=credential_mapping, openai_api_key="synthetic-provider-secret", milvus_token="synthetic-milvus-secret", langfuse_secret_key="synthetic-trace-secret")
    representation = repr(settings)
    for secret in [*credential_mapping, "synthetic-provider-secret", "synthetic-milvus-secret", "synthetic-trace-secret"]:
        assert secret not in representation


def test_configured_credential_map_cannot_be_modified_after_validation(credential_mapping):
    settings = Settings(_env_file=None, credential_mapping=credential_mapping)
    with pytest.raises(TypeError):
        settings.credential_mapping["attacker-token"] = settings.credential_mapping["synthetic-admin-token"]


@pytest.mark.parametrize("mapping", [{"": {"subject": "x", "customer_ids": []}}, {" token ": {"subject": "x", "customer_ids": []}}, {"\ud800": {"subject": "x", "customer_ids": []}}, {"token": {"subject": "", "customer_ids": []}}, {"token": {"subject": "x", "customer_ids": ["U1"]}}, {"token": {"subject": "x", "customer_ids": [], "ingest_admin": "false"}}])
def test_invalid_server_credentials_are_rejected(mapping):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, credential_mapping=mapping)


def test_principal_constructor_does_not_accept_unknown_permissions():
    with pytest.raises(ValidationError):
        Principal(subject="x", customer_ids=frozenset(), ingest_admin=False, all_customers=True)


def test_dataset_scope_requires_an_explicit_loopback_identity(credential_mapping):
    scoped = {"local-token": {"subject": "local", "customer_ids": [], "dataset_scope": True}}
    settings = Settings(_env_file=None, credential_mapping=scoped)
    principal = authenticate("local-token", settings)
    authorize(principal, "U0000000045")
    with pytest.raises(AuthorizationError):
        authorize(principal, "not-a-customer")
    with pytest.raises(ValidationError, match="loopback"):
        Settings(_env_file=None, credential_mapping=scoped, host="0.0.0.0")
    authorize(authenticate("synthetic-customer-token", Settings(_env_file=None, credential_mapping=credential_mapping)), "U0000000001")
    with pytest.raises(AuthorizationError):
        authorize(authenticate("synthetic-customer-token", Settings(_env_file=None, credential_mapping=credential_mapping)), "U0000000045")
