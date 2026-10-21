"""Deterministic LangChain documents from validated, provenance-bearing records."""

import hashlib
import json

from langchain_core.documents import Document

from app.data.repository import CustomerRecord
from app.errors import DependencyError

# A retrieval document is a concise discovery summary. Exact financial values,
# timestamps and unrestricted row contents remain in authorized SQLite tools.
SUMMARY_FIELDS = (
    "UserID",
    "RegionCode",
    "CityCode",
    "CustomerSegment",
    "AccountStatus",
    "RegistrationPlatform",
    "AcquisitionChannel",
    "PreferredLanguage",
    "PlanID",
    "PlanName",
    "PlanInterval",
    "SubscriptionStatus",
    "AutoRenewEnabled",
    "PaymentAttemptCount",
    "SuccessfulPaymentCount",
    "FailedPaymentCount",
    "UsageEventCount",
    "TotalUsageUnits",
    "MostUsedFeature",
    "SupportTicketCount",
    "OpenTicketCount",
    "ResolvedTicketCount",
    "RiskEventCount",
    "ConfirmedRiskEvents",
    "PendingRiskEvents",
    "DirectReferralsRegistered",
    "DirectReferralsPaid",
    "ReferralInvitesSent",
    "FeedbackCount",
    "HistoricalRewardCount",
    "EverFulfilledReward",
)
DOMAIN_TOOLS = (
    "customer",
    "subscription",
    "payments",
    "revenue",
    "engagement",
    "referrals",
    "support",
    "feedback",
    "risk",
    "rewards",
)


def build_document(record: CustomerRecord, *, include_all_fields: bool = False) -> Document:
    if not record.provenance or "UserID" not in record.provenance:
        raise DependencyError()
    provenance = record.provenance["UserID"]
    if any(
        (p.dataset_version, p.schema_version, p.source_sha256)
        != (
            provenance.dataset_version,
            provenance.schema_version,
            provenance.source_sha256,
        )
        for p in record.provenance.values()
    ):
        raise DependencyError()
    fields = record.values.keys() if include_all_fields else SUMMARY_FIELDS
    values = {field: record.values[field] for field in fields if field in record.values}
    source_fields = sorted(values)
    limitations = sorted(
        {
            limitation
            for field in source_fields
            for limitation in record.field_limitations.get(field, [])
        }
    )
    identity = json.dumps(
        [
            provenance.dataset_version,
            provenance.schema_version,
            provenance.source_sha256,
            record.user_id,
        ],
        separators=(",", ":"),
    )
    return Document(
        id=hashlib.sha256(identity.encode()).hexdigest(),
        page_content=json.dumps(
            {
                "user_id": record.user_id,
                "fields": values,
                "domain_tools": DOMAIN_TOOLS,
                "limitations": limitations,
            },
            separators=(",", ":"),
            ensure_ascii=True,
        ),
        metadata={
            "user_id": record.user_id,
            "dataset_version": provenance.dataset_version,
            "schema_version": provenance.schema_version,
            "source_sha256": provenance.source_sha256,
            "source_row": provenance.source_row,
            "source_fields": source_fields,
        },
    )
