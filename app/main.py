"""Application factory. Production never silently falls back to demo services."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import re
import time
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import httpx

from app.api.dependencies import RateLimiter
from app.api.jobs import IngestionJobs
from app.api.routes import router
from app.config import Settings
from app.errors import AppError, DependencyError
from app.observability.telemetry import LangfuseAdapter, Telemetry, current_correlation_id

@dataclass
class AppServices:
    assistant: Any = None
    analytics: Any = None
    ingestion: Callable[[],Any] | None = None
    readiness: Callable[[],dict] | None = None
    telemetry: Telemetry | None = None
    demo: bool = False
    demo_token: str | None = None


def _production_services(settings):
    """Bootstrap ingestion remains available before an active index exists."""
    services=AppServices()
    def refresh(snapshot=None):
        from app.rag.store import load_active_snapshot
        from app.rag.retriever import Retriever
        from app.agent.orchestrator import Assistant
        from app.services.analytics import AnalyticsService
        snapshot=snapshot or load_active_snapshot(settings)
        assistant=Assistant(settings,snapshot=snapshot,retriever=Retriever(settings,snapshot),telemetry=services.telemetry)
        analytics=AnalyticsService(snapshot.repository())
        # Assignment is one immutable bundle reference for each request via refreshed services.
        services.assistant=assistant
        services.analytics=analytics
    def ingestion():
        if settings.llm_provider == 'openai' and not settings.openai_api_key:
            raise DependencyError()
        from app.rag.store import ingest
        result=ingest(settings)
        refresh(result.snapshot)
        return result
    services.ingestion=ingestion
    services.readiness=lambda:{**_provider_readiness(settings),'index':'ready' if services.assistant else 'unavailable'}
    if settings.llm_provider == 'ollama' or settings.openai_api_key:
        try:
            refresh()
        except Exception:
            pass
    return services


def _provider_readiness(settings):
    if getattr(settings,'llm_provider','openai') != 'ollama':
        state='ready' if settings.openai_api_key else 'unavailable'
        return {'provider':state,'embeddings':state if settings.embedding_provider=='openai' else 'ready'}
    try:
        response=httpx.get(settings.ollama_base_url.rstrip('/')+'/api/tags',timeout=2,trust_env=False)
        response.raise_for_status()
        available={model.get('name','') for model in response.json().get('models',[])}
        def installed(name):
            name=name.removeprefix('ollama:')
            return name in available or name+':latest' in available
        return {'provider':'ready' if installed(settings.model_id) else 'unavailable',
                'embeddings':'ready' if settings.embedding_provider!='ollama' or installed(settings.embedding_model) else 'unavailable'}
    except Exception:
        return {'provider':'unavailable','embeddings':'unavailable'}


def create_app(settings:Settings|None=None,services:AppServices|None=None,*,clock=time.monotonic)->FastAPI:
    settings=settings or Settings()
    services=services or _production_services(settings)
    telemetry=services.telemetry or Telemetry(LangfuseAdapter(settings) if settings.langfuse_public_key and settings.langfuse_secret_key else None)
    services.telemetry=telemetry
    if services.assistant is not None and hasattr(services.assistant,'telemetry'):
        services.assistant.telemetry=telemetry
    jobs=IngestionJobs(services.ingestion)
    @asynccontextmanager
    async def lifespan(application):
        yield
        # A thread cannot safely be killed during atomic ingestion; drain it on shutdown.
        await jobs.wait()
        telemetry.close()
    application=FastAPI(title='Customer Knowledge Assistant',version='0.1.0',lifespan=lifespan)
    application.state.settings=settings
    application.state.services=services
    application.state.telemetry=telemetry
    application.state.jobs=jobs
    application.state.limiter=RateLimiter(settings.rate_limit_per_minute,clock)

    @application.middleware('http')
    async def bounded_request(request,call_next):
        supplied=request.headers.get('x-correlation-id','')
        request.state.correlation_id=supplied if re.fullmatch(r'[A-Za-z0-9_-]{1,64}',supplied) else uuid4().hex
        correlation_token=current_correlation_id.set(request.state.correlation_id)
        if request.method in {'POST','PUT','PATCH'}:
            # Bound even chunked bodies before schema parsing; no raw payload logging.
            body=bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body)>16384:
                    response=JSONResponse({'error':{'code':'request_too_large','message':'The request body is too large.'}},status_code=413)
                    response.headers['x-correlation-id']=request.state.correlation_id
                    current_correlation_id.reset(correlation_token)
                    return response
            request._body=bytes(body)
        with telemetry.stage('request',request.state.correlation_id,{'mode':'demo' if services.demo else 'production'}):
            try:
                response=await call_next(request)
            except Exception:
                response=JSONResponse({'error':{'code':'dependency_unavailable','message':'A required service is unavailable.'}},status_code=503)
        response.headers['x-correlation-id']=request.state.correlation_id
        response.headers['cache-control']='no-store'
        response.headers['x-content-type-options']='nosniff'
        telemetry.record('final_status',request.state.correlation_id,status=str(response.status_code))
        current_correlation_id.reset(correlation_token)
        return response

    @application.exception_handler(AppError)
    async def controlled(request,error):
        headers={'WWW-Authenticate':'Bearer'} if error.status_code==401 else None
        if error.status_code==429:
            headers={'Retry-After':'60'}
        return JSONResponse({'error':{'code':error.code,'message':error.message}},status_code=error.status_code,headers=headers)
    @application.exception_handler(RequestValidationError)
    async def invalid(request,error):
        return JSONResponse({'error':{'code':'validation_failed','message':'The supplied data could not be validated.'}},status_code=422)
    @application.get('/',include_in_schema=False)
    async def home():
        return {'service':'Customer Knowledge Assistant API','frontend_url':'http://127.0.0.1:8501','docs':'/docs','health':'/health'}
    application.include_router(router)
    return application


# Uvicorn uses the configured factory; Streamlit is the user interface.
app=create_app
