"""Retrieval-first, bounded single-agent orchestration; facts render deterministically."""
import asyncio
from contextlib import nullcontext
import json
import re
from pathlib import Path
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.settings import ToolOrOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from app.agent.tools import AgentDependencies, APPROVED_TOOLS
from app.data.dictionary import DOMAIN_FIELDS, FIELD_DICTIONARY
from app.errors import AppError, DependencyError, RequestTimeoutError, ValidationFailure
from app.guardrails.input import check_input
from app.guardrails.output import EvidenceLedger, controlled_answer, validate_result, METRIC_UNITS
from app.rag.context import build_context
from app.rag.documents import build_document
from app.rag.retriever import RetrievedEvidence, Retriever
from app.rag.store import load_active_snapshot
from app.schemas import AgentResult, ChatRequest
from langchain_core.documents import Document
from app.security import Principal, authorize
from app.services.analytics import AnalyticsService

PROMPTS=Path(__file__).resolve().parents[1]/'prompts'/'v1'

_TOOL_ROUTES=(
    ('revenue_summary',('revenue','monthly settled','settled total','monthly amount')),
    ('payment_summary',('payment','invoice','refund','chargeback','transaction','settled amount')),
    ('subscription_summary',('subscription','plan','renew','cancel','auto renew','tenure')),
    ('support_summary',('support','ticket','complaint')),
    ('risk_summary',('risk','fraud')),
    ('engagement_summary',('engagement','usage','login','activity','feature')),
    ('referrals_summary',('referral','invite')),
    ('feedback_summary',('feedback','rating','review','satisfaction')),
    ('rewards_summary',('reward','redemption','loyalty')),
    ('customer_record',('customer','profile','account','segment','registration','location')),
)

_DOMAIN_BY_TOOL={
    'customer_record':'customer','subscription_summary':'subscription','payment_summary':'payments',
    'support_summary':'support','risk_summary':'risk','engagement_summary':'engagement',
    'referrals_summary':'referrals','feedback_summary':'feedback','rewards_summary':'rewards',
}
_FIELD_ALIASES={
    'UserID':('user id','customer id'), 'CustomerSegment':('segment',),
    'AccountStatus':('account status',), 'SubscriptionStatus':('subscription status','plan status'),
    'PlanName':('plan name','subscription plan'), 'PlanID':('plan id',),
    'PlanPricePaise':('plan price','subscription price'), 'AutoRenewEnabled':('auto renew','renewal enabled'),
    'InvoiceCount':('invoice count','invoices'), 'TotalInvoicedPaise':('invoiced amount','total invoiced'),
    'PaymentAttemptCount':('payment attempts','payment attempt count'),
    'SuccessfulPaymentCount':('successful payments','successful payment count'),
    'FailedPaymentCount':('failed payments','failed payment count'), 'PaymentRetryCount':('payment retries','retry count'),
    'TotalSettledPaise':('settled amount','total settled'), 'RefundAmountPaise':('refund amount','refunds','refund'),
    'ChargebackAmountPaise':('chargeback amount','chargebacks','chargeback'),
    'NetRevenuePaise':('net revenue',), 'OpenTicketCount':('open tickets','open support tickets'),
    'ResolvedTicketCount':('resolved tickets',), 'SupportTicketCount':('support tickets','ticket count'),
    'HighPriorityTicketCount':('high priority tickets',), 'RiskEventCount':('risk events','risk event count'),
    'ConfirmedRiskEvents':('confirmed risk events',), 'PendingRiskEvents':('pending risk events',),
    'UsageEventCount':('usage events','usage event count'), 'TotalUsageUnits':('usage units',),
    'MostUsedFeature':('most used feature',), 'LoginEventCount':('login events',),
    'DirectReferralsRegistered':('registered referrals',), 'DirectReferralsPaid':('paid referrals',),
    'ReferralInvitesSent':('referral invites','invites sent'), 'FeedbackCount':('feedback count','feedback'),
    'AverageFeedbackScoreX100':('feedback score','average feedback score'),
    'HistoricalRewardCount':('reward count','rewards'), 'HistoricalRewardValuePaise':('reward value',),
    'EverFulfilledReward':('fulfilled reward','reward fulfilled'), 'LatestRewardMonth':('latest reward month',),
    'RegistrationPlatform':('registration platform','signup platform'), 'AcquisitionChannel':('acquisition channel',),
    'PreferredLanguage':('preferred language',), 'RegionCode':('region',), 'CityCode':('city',),
    'TenureDays':('tenure','tenure days'), 'RenewalEvents':('renewal events',),
}


def select_source_fields(question:str,available):
    normalized=' '.join(re.sub(r'[^a-z0-9]+',' ',question.casefold()).split())
    found=[]
    for field in available:
        camel=' '.join(re.sub(r'([a-z0-9])([A-Z])',r'\1 \2',field).casefold().split())
        aliases=_FIELD_ALIASES.get(field,())
        if any(re.search(rf'(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])',normalized) for phrase in (camel,*aliases)):
            found.append(field)
    return found


def route_question(question:str):
    """Keep only the fixed read tools relevant to a question on the model request."""
    normalized=' '.join(question.casefold().replace('-', ' ').split())
    selected=[name for name,terms in _TOOL_ROUTES if any(term in normalized for term in terms)]
    if not selected:
        selected=['customer_record']
    by_name={tool.__name__:tool for tool in APPROVED_TOOLS}
    tools=tuple(by_name[name] for name in selected)
    # Task-specific guardrails reinforce exact arithmetic and domain distinctions.
    primary=selected[0]
    task='revenue' if primary=='revenue_summary' else 'payment' if primary=='payment_summary' else 'customer'
    return tools,task

class Assistant:
    def __init__(self,settings,snapshot=None,retriever=None,model=None,*,synthetic_demo=False,telemetry=None):
        if getattr(model,'_synthetic_demo',False) and (not synthetic_demo or snapshot is None or not snapshot.version.startswith('synthetic-demo-')):
            raise ValueError('The deterministic demo model requires an explicit synthetic-demo snapshot.')
        self.settings=settings; self.snapshot=snapshot; self.retriever=retriever; self.model=model
        self.synthetic_demo=synthetic_demo; self.telemetry=telemetry

    def _stage(self,name):
        if not self.telemetry: return nullcontext()
        from app.observability.telemetry import current_correlation_id
        return self.telemetry.stage(name,current_correlation_id.get(),{})

    def _final(self,answer):
        if self.telemetry:
            from app.observability.telemetry import current_correlation_id
            self.telemetry.record('final_status',current_correlation_id.get(),status=answer.status,claim_count=answer.answer.count('\n')+1 if answer.status=='answered' else 0,evidence_count=len(answer.evidence))
        return answer

    async def answer(self,principal:Principal,request:ChatRequest):
        authorize(principal,request.user_id)
        if len(request.question)>self.settings.question_max_chars: raise ValidationFailure()
        with self._stage('routing'):
            policy=check_input(request.question)
            if any(user_id != request.user_id for user_id in re.findall(r'\bU[0-9]{10}\b',request.question)):
                return self._final(controlled_answer('refused','customer_scope_mismatch'))
        if not policy.allowed: return self._final(controlled_answer(policy.status,policy.reason))
        try:
            async with asyncio.timeout(self.settings.request_timeout_seconds):
                return self._final(await self._answer(principal,request))
        except TimeoutError:
            raise RequestTimeoutError() from None
        except AppError:
            raise
        except (UnexpectedModelBehavior,UsageLimitExceeded):
            return self._final(controlled_answer())
        except Exception:
            raise DependencyError() from None

    def _production_model(self):
        if getattr(self.settings,'llm_provider','openai') == 'ollama':
            from pydantic_ai.models.ollama import OllamaModel
            from pydantic_ai.providers.ollama import OllamaProvider
            model_name=self.settings.model_id.removeprefix('ollama:')
            provider=OllamaProvider(base_url=self.settings.ollama_base_url.rstrip('/')+'/v1',api_key='ollama')
            return OllamaModel(model_name,provider=provider)
        if self.settings.model_id.startswith('openai:') and self.settings.openai_api_key is not None:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            return OpenAIChatModel(self.settings.model_id.split(':',1)[1],provider=OpenAIProvider(api_key=self.settings.openai_api_key.get_secret_value()))
        return self.settings.model_id

    async def _answer(self,principal,request):
        # One immutable snapshot supplies both retrieval and deterministic tools for this request.
        snapshot=self.snapshot or await asyncio.to_thread(load_active_snapshot,self.settings)
        repository=snapshot.repository()
        # Confirm existence in the active dataset before retrieval so an
        # unknown but well-formed ID gets a clear 404 instead of a RAG refusal.
        repository.lookup(principal, request.user_id)
        tools,task=route_question(request.question)
        # Monthly revenue is intentionally excluded from vector documents. It is
        # calculated from the authorized source row, so a revenue-only request
        # must not depend on retrieval returning unrelated context first.
        if tools and all(tool.__name__=='revenue_summary' for tool in tools) and self.model is None and not self.synthetic_demo:
            context=build_context(request.question,principal,request.user_id,[],[],self.settings)
            ledger=EvidenceLedger(user_id=request.user_id)
            deps=AgentDependencies(principal,request.user_id,context,repository,AnalyticsService(repository),ledger,self.telemetry)
            answer=await self._extractive_answer(request.question,tools,deps,context)
            if answer is not None:
                return answer
            return controlled_answer('insufficient_evidence','missing_field')
        retriever=self.retriever or Retriever(self.settings,snapshot)
        with self._stage('retrieval') as retrieval_meta:
            documents=await asyncio.to_thread(retriever.retrieve,principal,request.user_id,request.question)
            # The Milvus collection is a compact discovery index. Customer-scoped
            # questions can still retrieve any row in the full SQLite source even
            # when that customer's summary was intentionally not embedded.
            if not documents and getattr(snapshot,'document_count',0)<getattr(snapshot,'row_count',0):
                record=await asyncio.to_thread(repository.lookup,principal,request.user_id)
                documents=[RetrievedEvidence(build_document(record,include_all_fields=True),1.0)]
            if retrieval_meta is not None: retrieval_meta['document_count']=len(documents)
            with self._stage('context') as context_meta:
                # Reduce distractors by exposing only the requested domain's RAG fields.
                # Revenue values stay out of the vector index; its typed tool computes them.
                selected_domains={_DOMAIN_BY_TOOL[tool.__name__] for tool in tools if tool.__name__ in _DOMAIN_BY_TOOL}
                allowed_fields={field for domain in selected_domains for field in DOMAIN_FIELDS[domain]}
                if allowed_fields:
                    allowed_fields.add('UserID')
                focused=[]
                for item in documents:
                    document=item.document if isinstance(item,RetrievedEvidence) else item
                    if allowed_fields:
                        metadata={**document.metadata,'source_fields':sorted(set(document.metadata.get('source_fields',())) & allowed_fields)}
                        document=Document(id=document.id,page_content=document.page_content,metadata=metadata)
                    focused.append(RetrievedEvidence(document,item.score) if isinstance(item,RetrievedEvidence) else document)
                context=build_context(request.question,principal,request.user_id,focused,[],self.settings)
                if context_meta is not None: context_meta.update(context_chars=len(context.text),context_tokens=len(context.text))
        if not context.documents or not context.source_fields:
            return controlled_answer()
        # Claims can only cite values returned by an authorized fixed-domain tool.
        ledger=EvidenceLedger(user_id=request.user_id)
        deps=AgentDependencies(principal,request.user_id,context,repository,AnalyticsService(repository),ledger,self.telemetry)
        extractive=await self._extractive_answer(request.question,tools,deps,context) if self.model is None and not self.synthetic_demo else None
        if extractive is not None:
            return extractive
        instructions=(PROMPTS/'system.txt').read_text()
        instructions+='\n'+(PROMPTS/f'{task}.txt').read_text()
        instructions+='\nCall the available fixed-domain tool before producing claims. Only its returned values may support the answer.'
        agent=Agent(self.model or self._production_model(),output_type=AgentResult,deps_type=AgentDependencies,instructions=instructions,tools=tools,retries={'tools':0,'output':self.settings.output_retries})

        @agent.output_validator
        def verify(ctx:RunContext[AgentDependencies],output:AgentResult)->AgentResult:
            with self._stage('validation') as validation_meta:
                try:
                    validate_result(output,ctx.deps.context,ctx.deps.evidence)
                except ValidationFailure:
                    if validation_meta is not None:
                        validation_meta.update(validation_passed=False,error_code='validation_failed',claim_count=len(output.claims))
                    raise ModelRetry('Claims must exactly match authorized source values, UserID, field/metric names and units. Return insufficient_evidence when unsupported.') from None
                if validation_meta is not None:
                    validation_meta.update(validation_passed=True,claim_count=len(output.claims))
                return output

        prompt=json.dumps({'user_id':request.user_id,'question':request.question,'untrusted_context':context.text},ensure_ascii=True)
        with self._stage('generation') as generation_meta:
            result=await agent.run(prompt,deps=deps,model_settings={'temperature':0,'max_tokens':512,'parallel_tool_calls':False,'tool_choice':ToolOrOutput(function_tools=[tool.__name__ for tool in tools])},usage_limits=UsageLimits(request_limit=min(4,self.settings.model_request_limit),tool_calls_limit=min(len(tools)+1+self.settings.output_retries,self.settings.tool_call_limit)))
            if generation_meta is not None:
                usage=result.usage
                generation_meta.update(input_tokens=usage.input_tokens,output_tokens=usage.output_tokens,total_tokens=usage.total_tokens,request_count=usage.requests,tool_calls=usage.tool_calls,cost_known=False,model='synthetic-demo-deterministic' if self.synthetic_demo else self.settings.model_id,prompt_version='v1')
        with self._stage('validation') as validation_meta:
            answer=validate_result(result.output,context,ledger)
            if validation_meta is not None: validation_meta.update(validation_passed=True,claim_count=len(result.output.claims))
            return answer

    async def _extractive_answer(self,question,tools,deps,context):
        """Answer exact field/metric questions from typed read tools without LLM latency."""
        if not tools or (not context.documents and any(tool.__name__!='revenue_summary' for tool in tools)):
            return None
        normalized=' '.join(re.sub(r'[^a-z0-9]+',' ',question.casefold()).split())
        broad=any(phrase in normalized for phrase in ('summarize','summary','customer details','all details','tell me about'))
        ledger=deps.evidence
        claims=[]
        for tool in tools:
            if tool.__name__=='revenue_summary':
                result=await asyncio.to_thread(deps.services.revenue,deps.principal,deps.user_id)
                ledger.add_revenue(result)
                requested_metrics=(
                    ('monthly_settled_total_paise','monthly_settled_complete','monthly_settled_available_months')
                    if any(word in normalized for word in ('revenue','settled total','monthly amount')) or broad
                    else ('monthly_settled_complete','monthly_settled_available_months')
                )
                for metric in requested_metrics:
                    value=ledger.metrics.get(metric)
                    if value is not None and ledger.metric_fields.get(metric):
                        claims.append({'user_id':deps.user_id,'metric_id':metric,'value':value,'unit':METRIC_UNITS[metric]})
                continue
            domain=_DOMAIN_BY_TOOL.get(tool.__name__)
            if domain is None:
                continue
            result=await asyncio.to_thread(deps.services.domain,deps.principal,deps.user_id,domain)
            ledger.add_domain(result)
            requested=select_source_fields(question,result.values)
            if not requested and broad:
                requested=[key for key,value in result.values.items() if value is not None]
            for field in requested:
                value=result.values.get(field)
                if value is not None and field in ledger.values:
                    claims.append({'user_id':deps.user_id,'source_field':field,'value':value,'unit':FIELD_DICTIONARY[field].unit})
        if not claims:
            return None
        result=AgentResult(status='answered',claims=claims[:12])
        try:
            answer=validate_result(result,context,ledger)
        except ValidationFailure:
            return None
        if self.telemetry:
            from app.observability.telemetry import current_correlation_id
            self.telemetry.record('extractive_fast_path',current_correlation_id.get(),status=answer.status,tool_calls=len(tools),claim_count=len(result.claims),model='server-verified-fast-path',cost_known=False)
        return answer


def make_demo_model():
    """Explicit deterministic synthetic-fixture provider. This is not a live LLM.

    Exercises the same Pydantic AI typed tool loop, authorization, claim validator,
    and renderer as production. Never selected as a runtime failure fallback.
    """
    def demo(messages,info):
        prompt=None; returns=[]
        for message in messages:
            for part in message.parts:
                if isinstance(part,UserPromptPart) and isinstance(part.content,str):
                    try: prompt=json.loads(part.content)
                    except ValueError: pass
                if isinstance(part,ToolReturnPart): returns.append(part)
        if prompt is None: raise ValueError('Expected a structured demo request.')
        user_id=prompt['user_id']; question=prompt['question'].lower()
        if not returns:
            mapping={'revenue_summary':('revenue','monthly'),'payment_summary':('payment','invoice','settled','refund','chargeback'),'subscription_summary':('subscription','plan','cancel','renew'),'support_summary':('support','ticket'),'risk_summary':('risk',),'engagement_summary':('engagement','usage','activity','login'),'referrals_summary':('referral',),'feedback_summary':('feedback',),'rewards_summary':('reward',)}
            names=[name for name,words in mapping.items() if any(word in question for word in words)]
            names=names or ['customer_record']
            return ModelResponse(parts=[ToolCallPart(name,{'user_id':user_id},tool_call_id=f'demo-{i}') for i,name in enumerate(names[:8])])
        claims=[]
        for part in returns:
            data=part.content
            if hasattr(data,'model_dump'): data=data.model_dump()
            if isinstance(data,str): data=json.loads(data)
            if not isinstance(data,dict) or data.get('user_id')!=user_id: continue
            if 'monthly_paise' in data:
                metrics={'monthly_settled_total_paise':data['total_paise'],'monthly_settled_complete':data['complete'],'monthly_settled_available_months':len(data['available_months'])}
                for metric,value in metrics.items():
                    if value is not None and data.get('metric_provenance',{}).get(metric,{}).get('source_fields'):
                        claims.append({'user_id':user_id,'metric_id':metric,'value':value,'unit':METRIC_UNITS[metric]})
            else:
                for name,value in data.get('values',{}).items():
                    if value is not None and name in FIELD_DICTIONARY:
                        claims.append({'user_id':user_id,'source_field':name,'value':value,'unit':FIELD_DICTIONARY[name].unit})
        output={'status':'answered' if claims else 'insufficient_evidence','claims':claims[:100]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,output)])
    model=FunctionModel(demo,model_name='synthetic-demo-deterministic')
    model._synthetic_demo=True
    return model
