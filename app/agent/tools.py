"""Only fixed-domain read tools; no SQL, file, network, or arbitrary execution tool."""
import asyncio
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any
from pydantic_ai import RunContext
from app.data.repository import SourceRepository
from app.errors import AuthorizationError
from app.guardrails.output import EvidenceLedger
from app.rag.context import ContextBundle
from app.schemas import UserID
from app.security import Principal, authorize
from app.services.analytics import AnalyticsService, DomainSummary, RevenueSummary

@dataclass(frozen=True)
class AgentDependencies:
    principal: Principal
    user_id: str
    context: ContextBundle
    repository: SourceRepository
    services: AnalyticsService
    evidence: EvidenceLedger
    telemetry: Any = None

    def tool_stage(self):
        if not self.telemetry: return nullcontext()
        from app.observability.telemetry import current_correlation_id
        return self.telemetry.stage('tools',current_correlation_id.get(),{})

    def authorize_scope(self, user_id: str):
        authorize(self.principal,user_id)
        if user_id != self.user_id: raise AuthorizationError()

async def _domain(ctx:RunContext[AgentDependencies],user_id:str,domain:str)->DomainSummary:
    ctx.deps.authorize_scope(user_id)
    with ctx.deps.tool_stage():
        result=await asyncio.to_thread(ctx.deps.services.domain,ctx.deps.principal,user_id,domain)
        ctx.deps.evidence.add_domain(result)
    return result

async def customer_record(ctx:RunContext[AgentDependencies],user_id:UserID)->DomainSummary:
    """Read recorded customer identity/profile fields for the request customer only."""
    return await _domain(ctx,user_id,'customer')

async def subscription_summary(ctx:RunContext[AgentDependencies],user_id:UserID)->DomainSummary:
    """Read subscription aggregates and literal cancellation codes, without inferring meanings."""
    return await _domain(ctx,user_id,'subscription')

async def payment_summary(ctx:RunContext[AgentDependencies],user_id:UserID)->DomainSummary:
    """Read distinct payment, invoice, settled and refund aggregate source fields."""
    return await _domain(ctx,user_id,'payments')

async def revenue_summary(ctx:RunContext[AgentDependencies],user_id:UserID)->RevenueSummary:
    """Compute exact available-month settled paise total, completeness, and source provenance."""
    ctx.deps.authorize_scope(user_id)
    with ctx.deps.tool_stage():
        result=await asyncio.to_thread(ctx.deps.services.revenue,ctx.deps.principal,user_id)
        ctx.deps.evidence.add_revenue(result)
    return result

async def support_summary(ctx:RunContext[AgentDependencies],user_id:UserID)->DomainSummary:
    """Read support counts; ticket descriptions and event timelines are unavailable."""
    return await _domain(ctx,user_id,'support')

async def risk_summary(ctx:RunContext[AgentDependencies],user_id:UserID)->DomainSummary:
    """Read separate confirmed and pending risk counts; these do not establish fraud."""
    return await _domain(ctx,user_id,'risk')

async def engagement_summary(ctx:RunContext[AgentDependencies],user_id:UserID)->DomainSummary:
    """Read recorded usage and engagement aggregates."""
    return await _domain(ctx,user_id,'engagement')

async def referrals_summary(ctx:RunContext[AgentDependencies],user_id:UserID)->DomainSummary:
    """Read referral aggregates without accessing a referred customer's record."""
    return await _domain(ctx,user_id,'referrals')

async def feedback_summary(ctx:RunContext[AgentDependencies],user_id:UserID)->DomainSummary:
    """Read feedback counts and literal recorded score without inventing the score scale."""
    return await _domain(ctx,user_id,'feedback')

async def rewards_summary(ctx:RunContext[AgentDependencies],user_id:UserID)->DomainSummary:
    """Read historical reward counts, values, and recorded fulfilment flag."""
    return await _domain(ctx,user_id,'rewards')

APPROVED_TOOLS=(customer_record,subscription_summary,payment_summary,revenue_summary,support_summary,risk_summary,engagement_summary,referrals_summary,feedback_summary,rewards_summary)
