"""Explicit synthetic runtime. Never reads the shared customer dataset."""

from contextlib import contextmanager
import csv
from dataclasses import dataclass
from pathlib import Path
import secrets
from tempfile import TemporaryDirectory

from app.config import Settings
from app.data.dictionary import FIELD_DICTIONARY, MONTHLY_FIELDS
from app.data.ingestion import stage_dataset
from app.rag.documents import build_document
from app.rag.embeddings import embedding_signature
from app.rag.retriever import Retriever
from app.rag.store import ActiveSnapshot
from app.security import authenticate
from app.services.analytics import AnalyticsService


class SyntheticStore:
    """Only the explicit bundled demo uses deterministic document selection."""

    def __init__(self, documents):
        self.documents = tuple(documents)

    def search(self, snapshot, question, *, k, expr):
        # Match the exact server-generated filter; never parse arbitrary expressions.
        return [(document, 1.0) for document in self.documents if expr == f'user_id == "{document.metadata["user_id"]}"'][:k]


def write_synthetic_csv(path):
    fields = list(FIELD_DICTIONARY)
    rows = []
    for index in (1, 2):
        row = {field: "0" if definition.kind == "integer" else "false" if definition.kind == "boolean" else "" for field, definition in FIELD_DICTIONARY.items()}
        row.update(UserID=f"U{index:010d}", SyntheticName=f"Demo Customer {index}", RegionCode="DEMO", CityCode="DEMO_CITY", CustomerSegment="PROFESSIONAL", AccountStatus="ACTIVE", PlanName="STANDARD", PlanID="DEMO_STANDARD", PlanInterval="ANNUAL", PlanPricePaise="120000", SubscriptionID=f"DEMO_SUB_{index}", SubscriptionStatus="ACTIVE", AutoRenewEnabled="true", PaymentAttemptCount="7", SuccessfulPaymentCount="4", FailedPaymentCount="2", PaymentRetryCount="1", InvoiceCount="4", TotalInvoicedPaise="130000", TotalSettledPaise="120000", NetRevenuePaise="110000", RefundAmountPaise="10000", MainPaymentMethod="DEMO_CARD", UsageEventCount="42", TotalUsageUnits="120", MonthsWithActivity="8", LoginEventCount="20", MostUsedFeature="DASHBOARD", SupportTicketCount="3", OpenTicketCount="1", ResolvedTicketCount="2", HighPriorityTicketCount="1", RiskEventCount="3", ConfirmedRiskEvents="1", PendingRiskEvents="2", ReferralInvitesSent="5", ReferralLinkOpens="3", ReferralConversions="1", DirectReferralsRegistered="2", DirectReferralsPaid="1", FeedbackCount="2", AverageFeedbackScoreX100="450", HistoricalRewardCount="1", HistoricalRewardValuePaise="500", EverFulfilledReward="true", LatestRewardMonth="202609")
        row.update({field: "10000" for field in MONTHLY_FIELDS})
        rows.append(row)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class DemoRuntime:
    settings: Settings
    services: object
    principal: object
    token: str
    snapshot: ActiveSnapshot


@contextmanager
def create_demo_runtime():
    from app.agent.orchestrator import Assistant, make_demo_model
    from app.main import AppServices
    from app.observability.telemetry import RecordingAdapter, Telemetry

    with TemporaryDirectory(prefix="knowledge-assistant-demo-") as directory:
        root = Path(directory)
        source = root / "synthetic.csv"
        write_synthetic_csv(source)
        token = secrets.token_urlsafe(32)
        settings = Settings(_env_file=None, credential_mapping={token: {"subject": "synthetic-demo-user", "customer_ids": ["U0000000001"], "ingest_admin": False}}, dataset_path=source, staging_dir=root / "staging", active_manifest_path=root / "active.json", rate_limit_per_minute=1000)
        staged = stage_dataset(settings, version="synthetic-demo-local")
        signature = embedding_signature(settings)
        snapshot = ActiveSnapshot(version=staged.version, sqlite_path=staged.sqlite_path, collection_name="synthetic_demo", schema_version=staged.schema_version, source_sha256=staged.source_sha256, embedding_signature=signature, row_count=staged.row_count, document_count=staged.row_count, timestamp_formats={})
        repository = snapshot.repository()
        documents = []
        for record in repository.iter_records():
            document = build_document(record)
            document.metadata["embedding_signature"] = signature
            documents.append(document)
        store = SyntheticStore(documents)
        telemetry = Telemetry(RecordingAdapter())
        assistant = Assistant(settings, snapshot=snapshot, retriever=Retriever(settings, snapshot, vector_store=store), model=make_demo_model(), synthetic_demo=True, telemetry=telemetry)
        services = AppServices(assistant=assistant, analytics=AnalyticsService(repository), telemetry=telemetry, readiness=lambda: {"synthetic_source": "ready", "deterministic_demo_provider": "ready"}, demo=True, demo_token=token)
        yield DemoRuntime(settings, services, authenticate(token, settings), token, snapshot)
