"""Conservative field semantics from column names; no undocumented code decoding."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


MONTHS = ("202510", "202511", "202512", "202601", "202602", "202603", "202604", "202605", "202606", "202607", "202608", "202609")
MONTHLY_FIELDS = tuple(f"SettledRevenue{month}Paise" for month in MONTHS)

DOMAIN_FIELDS = MappingProxyType({
    "customer": ("UserID", "SyntheticName", "RegistrationAtUTC", "RegionCode", "CityCode", "CustomerSegment", "AccountStatus", "RegistrationPlatform", "AcquisitionChannel", "PreferredLanguage", "ConsentStatus", "TenureDays", "FirstCampaignID", "AcquisitionTouchCount"),
    "subscription": ("SubscriptionID", "PlanID", "PlanName", "PlanInterval", "PlanPricePaise", "SubscriptionStatus", "AutoRenewEnabled", "SubscriptionStartAtUTC", "RenewalEvents", "LastCancellationReasonCode"),
    "payments": ("InvoiceCount", "TotalInvoicedPaise", "TotalDiscountPaise", "TotalTaxPaise", "PaymentAttemptCount", "SuccessfulPaymentCount", "FailedPaymentCount", "PaymentRetryCount", "TotalSettledPaise", "RefundAmountPaise", "ChargebackAmountPaise", "CreditNoteAmountPaise", "MainPaymentMethod", "LastSettledPaymentAtUTC"),
    "revenue": ("NetRevenuePaise", *MONTHLY_FIELDS),
    "engagement": ("UsageEventCount", "TotalUsageUnits", "TotalEngagementSeconds", "MonthsWithActivity", "LatestActivityAtUTC", "MostUsedFeature", "LoginEventCount"),
    "referrals": ("ReferrerUserID", "DirectReferralsRegistered", "DirectReferralsPaid", "ReferralInvitesSent", "ReferralLinkOpens", "ReferralConversions"),
    "support": ("SupportTicketCount", "OpenTicketCount", "ResolvedTicketCount", "HighPriorityTicketCount"),
    "feedback": ("FeedbackCount", "AverageFeedbackScoreX100"),
    "risk": ("RiskEventCount", "ConfirmedRiskEvents", "PendingRiskEvents"),
    "rewards": ("HistoricalRewardCount", "HistoricalRewardValuePaise", "EverFulfilledReward", "LatestRewardMonth"),
})

TIMESTAMP_FIELDS = frozenset({"RegistrationAtUTC", "SubscriptionStartAtUTC", "LastSettledPaymentAtUTC", "LatestActivityAtUTC"})
BOOLEAN_FIELDS = frozenset({"AutoRenewEnabled", "EverFulfilledReward"})
COUNT_FIELDS = frozenset({"TenureDays", "AcquisitionTouchCount", "DirectReferralsRegistered", "DirectReferralsPaid", "ReferralInvitesSent", "ReferralLinkOpens", "ReferralConversions", "RenewalEvents", "InvoiceCount", "PaymentAttemptCount", "SuccessfulPaymentCount", "FailedPaymentCount", "PaymentRetryCount", "UsageEventCount", "TotalUsageUnits", "TotalEngagementSeconds", "MonthsWithActivity", "LoginEventCount", "SupportTicketCount", "OpenTicketCount", "ResolvedTicketCount", "HighPriorityTicketCount", "FeedbackCount", "RiskEventCount", "ConfirmedRiskEvents", "PendingRiskEvents", "HistoricalRewardCount"})
CATEGORICAL_FIELDS = frozenset({"RegionCode", "CityCode", "CustomerSegment", "AccountStatus", "RegistrationPlatform", "AcquisitionChannel", "PreferredLanguage", "ConsentStatus", "PlanID", "PlanName", "PlanInterval", "SubscriptionStatus", "LastCancellationReasonCode", "MainPaymentMethod", "MostUsedFeature"})


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    domain: str
    unit: str
    kind: Literal["integer", "boolean", "timestamp", "month", "text", "identifier", "code"]
    description: str
    code_meanings_known: bool = False
    signed: bool = False


def _definition(name: str, domain: str) -> FieldDefinition:
    if name.endswith("Paise"):
        return FieldDefinition(name, domain, "paise", "integer", "Integer paise; preserve separately from other financial measures.", signed=name == "NetRevenuePaise")
    if name in COUNT_FIELDS or name == "AverageFeedbackScoreX100":
        unit = "ratio" if name == "AverageFeedbackScoreX100" else "count"
        description = {"AverageFeedbackScoreX100": "Recorded integer score; the heading includes X100 but the multiplier and underlying scale are undocumented.", "TotalEngagementSeconds": "Recorded aggregate engagement in seconds, not an event timeline.", "TenureDays": "Recorded tenure in days; reference date undocumented.", "ConfirmedRiskEvents": "Count recorded as confirmed risk events; does not establish fraud.", "PendingRiskEvents": "Count recorded as pending risk events; does not establish fraud."}.get(name, "Recorded aggregate count; no event timeline or individual details supplied.")
        return FieldDefinition(name, domain, unit, "integer", description)
    if name in BOOLEAN_FIELDS:
        return FieldDefinition(name, domain, "boolean", "boolean", "Recorded boolean flag; missing is distinct from false.")
    if name in TIMESTAMP_FIELDS:
        return FieldDefinition(name, domain, "timestamp", "timestamp", "UTC indicated by source heading; numeric encoding undocumented. Parse only with an operator-configured format.")
    if name == "LatestRewardMonth":
        return FieldDefinition(name, domain, "text", "month", "Source reward month; six-digit YYYYMM validated where possible, invalid source preserved.")
    if name in CATEGORICAL_FIELDS:
        return FieldDefinition(name, domain, "code", "code", "Literal source code or category; business meaning and vocabulary undocumented.")
    if name.endswith("ID"):
        return FieldDefinition(name, domain, "text", "identifier", "Literal source identifier; UserID is U plus ten ASCII digits.")
    return FieldDefinition(name, domain, "text", "text", "Literal source text; no inferred business meaning.")


FIELD_DICTIONARY = MappingProxyType({field: _definition(field, domain) for domain, fields in DOMAIN_FIELDS.items() for field in fields})
INTEGER_FIELDS = frozenset(name for name, definition in FIELD_DICTIONARY.items() if definition.kind == "integer")
