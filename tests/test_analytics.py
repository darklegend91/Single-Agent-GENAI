import pytest

from app.data.ingestion import stage_dataset
from app.data.repository import SourceRepository
from app.errors import AuthorizationError, NotFoundError, ValidationFailure
from app.security import Principal
from app.services.analytics import AnalyticsService, format_rupees
from tests.test_data_quality import OWNER, configured, write_csv


@pytest.fixture
def service(tmp_path):
    return AnalyticsService(SourceRepository(stage_dataset(configured(tmp_path)).sqlite_path))


def test_domain_summaries_keep_distinct_payment_risk_and_unknown_codes(service):
    summary = service.summary(OWNER, "U0000000001")
    assert summary.domains["payments"].values["PaymentAttemptCount"] == 7
    assert summary.domains["payments"].values["SuccessfulPaymentCount"] == 4
    assert summary.domains["payments"].values["FailedPaymentCount"] == 2
    assert summary.domains["risk"].values["ConfirmedRiskEvents"] == 2
    assert summary.domains["risk"].values["PendingRiskEvents"] == 3
    assert summary.domains["subscription"].values["LastCancellationReasonCode"] == "UNEXPLAINED"
    assert "unknown_code" in summary.domains["subscription"].limitations
    assert summary.domains["support"].source_fields == ["SupportTicketCount", "OpenTicketCount", "ResolvedTicketCount", "HighPriorityTicketCount"]
    assert set(summary.domains) == {"customer", "subscription", "payments", "revenue", "engagement", "referrals", "support", "feedback", "risk", "rewards"}


def test_revenue_uses_exact_sum_available_months_and_explicit_provenance(service):
    revenue = service.revenue(OWNER, "U0000000001")
    assert revenue.total_paise == 9007199254741000
    assert revenue.complete is False
    assert revenue.available_months == ["202510", "202511"]
    assert revenue.monthly_paise["202512"] is None
    assert "incomplete_period" in revenue.limitations
    assert revenue.metric_provenance["monthly_settled_total_paise"].source_fields == ["SettledRevenue202510Paise", "SettledRevenue202511Paise"]
    assert revenue.metric_provenance["monthly_settled_total_paise"].dataset_version
    assert format_rupees(revenue.total_paise) == "90071992547410.00"


def test_absent_revenue_never_becomes_complete_zero(service):
    revenue = service.revenue(OWNER, "U0000000002")
    assert revenue.total_paise is None
    assert revenue.complete is False
    assert revenue.available_months == []


def test_complete_zero_period_is_distinct_from_missing(tmp_path):
    row = {"UserID": "U0000000001"}
    for month in ["202510", "202511", "202512", "202601", "202602", "202603", "202604", "202605", "202606", "202607", "202608", "202609"]:
        row[f"SettledRevenue{month}Paise"] = "0"
    service = AnalyticsService(SourceRepository(stage_dataset(configured(tmp_path, write_csv(tmp_path, [row]))).sqlite_path))
    revenue = service.revenue(OWNER, "U0000000001")
    assert revenue.total_paise == 0
    assert revenue.complete is True


@pytest.mark.parametrize("operation", ["lookup", "summary", "revenue", "domain"])
def test_repository_and_every_tool_authorize_independently(tmp_path, operation):
    repository = SourceRepository(stage_dataset(configured(tmp_path)).sqlite_path)
    service = AnalyticsService(repository)
    outsider = Principal(subject="admin", customer_ids=set(), ingest_admin=True)
    with pytest.raises(AuthorizationError):
        if operation == "lookup":
            repository.lookup(outsider, "U0000000001")
        elif operation == "domain":
            service.domain(outsider, "U0000000001", "risk")
        else:
            getattr(service, operation)(outsider, "U0000000001")


def test_authorized_missing_record_is_controlled_and_unknown_domain_rejected(service):
    principal = Principal(subject="tester", customer_ids={"U9999999999"})
    with pytest.raises(NotFoundError):
        service.summary(principal, "U9999999999")
    with pytest.raises(ValidationFailure):
        service.domain(OWNER, "U0000000001", "arbitrary SQL")
