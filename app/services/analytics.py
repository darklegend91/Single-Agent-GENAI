"""Fixed-domain summaries and exact monthly revenue, with no invented code meanings."""

from decimal import Decimal, localcontext

from pydantic import BaseModel, ConfigDict, Field

from app.data.dictionary import DOMAIN_FIELDS, MONTHLY_FIELDS, MONTHS
from app.data.repository import FieldProvenance, SourceRepository, SourceValue
from app.errors import ValidationFailure
from app.schemas import MetricID, UserID
from app.security import Principal, authorize


class DomainSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: UserID
    domain: str
    values: dict[str, SourceValue] = Field(repr=False)
    source_fields: list[str]
    provenance: dict[str, FieldProvenance]
    limitations: list[str]


class CustomerSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: UserID
    domains: dict[str, DomainSummary]
    limitations: list[str]


class MetricProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    metric_id: MetricID
    dataset_version: str
    schema_version: str
    source_sha256: str
    source_fields: list[str]
    calculation: str


class RevenueSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: UserID
    monthly_paise: dict[str, int | None]
    total_paise: int | None
    available_months: list[str]
    complete: bool
    source_fields: list[str]
    provenance: dict[str, FieldProvenance]
    metric_provenance: dict[str, MetricProvenance]
    limitations: list[str]


def format_rupees(paise: int) -> str:
    """Decimal division with enough precision even beyond the default context."""
    if type(paise) is not int:
        raise ValidationFailure()
    with localcontext() as context:
        context.prec = max(28, len(str(abs(paise))) + 3)
        return format(Decimal(paise) / Decimal(100), ".2f")


class AnalyticsService:
    def __init__(self, repository: SourceRepository):
        self.repository = repository

    @staticmethod
    def _domain(record, domain: str) -> DomainSummary:
        fields = DOMAIN_FIELDS[domain]
        limitations = {"aggregate_only"} if domain in {"payments", "revenue", "engagement", "referrals", "support", "feedback", "risk", "rewards"} else set()
        for field in fields:
            limitations.update(record.field_limitations.get(field, []))
            if record.values.get(field) is None:
                limitations.add("missing_field")
        return DomainSummary(user_id=record.user_id, domain=domain, values={field: record.values.get(field) for field in fields}, source_fields=list(fields), provenance={field: record.provenance[field] for field in fields if field in record.provenance}, limitations=sorted(limitations))

    def domain(self, principal: Principal, user_id: str, domain: str) -> DomainSummary:
        authorize(principal, user_id)
        if domain not in DOMAIN_FIELDS:
            raise ValidationFailure()
        return self._domain(self.repository.lookup(principal, user_id), domain)

    def summary(self, principal: Principal, user_id: str) -> CustomerSummary:
        authorize(principal, user_id)
        record = self.repository.lookup(principal, user_id)
        domains = {domain: self._domain(record, domain) for domain in DOMAIN_FIELDS}
        return CustomerSummary(user_id=user_id, domains=domains, limitations=sorted({limitation for domain in domains.values() for limitation in domain.limitations}))

    def revenue(self, principal: Principal, user_id: str) -> RevenueSummary:
        authorize(principal, user_id)
        record = self.repository.lookup(principal, user_id)
        monthly = {month: record.values.get(field) for month, field in zip(MONTHS, MONTHLY_FIELDS, strict=True)}
        available = [month for month, value in monthly.items() if value is not None]
        complete = len(available) == len(MONTHS)
        fields = [f"SettledRevenue{month}Paise" for month in available]
        reference = record.provenance["UserID"]
        metric_sources = {
            "monthly_settled_total_paise": (fields, "Exact integer sum of available validated monthly paise; null if none available."),
            "monthly_settled_complete": (list(MONTHLY_FIELDS), "True only when all twelve monthly source values are available."),
            "monthly_settled_available_months": (fields, "Count of available validated monthly source values."),
        }
        return RevenueSummary(user_id=user_id, monthly_paise=monthly, total_paise=sum(monthly[month] for month in available) if available else None, available_months=available, complete=complete, source_fields=list(MONTHLY_FIELDS), provenance={field: record.provenance[field] for field in MONTHLY_FIELDS if field in record.provenance}, metric_provenance={metric: MetricProvenance(metric_id=metric, dataset_version=reference.dataset_version, schema_version=reference.schema_version, source_sha256=reference.source_sha256, source_fields=source_fields, calculation=calculation) for metric, (source_fields, calculation) in metric_sources.items()}, limitations=[] if complete else ["incomplete_period", "missing_field"])
