import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import DependencyError
from app.main import AppServices, create_app
from app.schemas import CustomerAnswer, Evidence
from app.api.jobs import IngestionJobs
from app.security import Principal


class Assistant:
    async def answer(self, principal, request):
        return CustomerAnswer(status='answered', answer='Synthetic answer', evidence=[Evidence(user_id=request.user_id, source_fields=['UserID'], explanation='Synthetic evidence')])


@pytest.fixture
def client(credential_mapping):
    settings = Settings(credential_mapping=credential_mapping)
    return TestClient(create_app(settings, AppServices(assistant=Assistant(), demo=True)))


def test_health_ui_and_chat(client):
    health = client.get('/health')
    assert health.status_code == 200
    assert health.json()['mode'] == 'demo'
    assert 'customer' not in health.text
    assert health.headers['x-correlation-id']
    home=client.get('/').json()
    assert home['service']=='Customer Knowledge Assistant API'
    assert home['frontend_url']=='http://127.0.0.1:8501'
    assert 'synthetic-customer-token' not in str(home)
    response = client.post('/chat', headers={'Authorization':'Bearer synthetic-customer-token'}, json={'user_id':'U0000000001','question':'status?'})
    assert response.status_code == 200
    assert response.json()['evidence']


def test_openapi_declares_bearer_auth_for_protected_routes(client):
    schema = client.get('/openapi.json').json()
    assert schema['components']['securitySchemes']['HTTPBearer']['scheme'] == 'bearer'
    assert schema['paths']['/chat']['post']['security'] == [{'HTTPBearer': []}]
    assert schema['paths']['/ingest']['post']['security'] == [{'HTTPBearer': []}]
    assert 'security' not in schema['paths']['/health']['get']
    assert schema['paths']['/observability/langfuse']['get']['security'] == [{'HTTPBearer': []}]


def test_langfuse_status_reports_disabled_without_credentials(client):
    response = client.get('/observability/langfuse', headers={'Authorization':'Bearer synthetic-customer-token'})
    assert response.status_code == 200
    assert response.json()['status'] == 'disabled'


def test_langfuse_status_checks_configured_adapter(credential_mapping):
    from app.observability.telemetry import Telemetry
    class Adapter:
        def check_connection(self):
            return True
        def emit(self, event):
            pass
    services=AppServices(assistant=Assistant(),telemetry=Telemetry(Adapter()))
    client=TestClient(create_app(Settings(credential_mapping=credential_mapping),services))
    response=client.get('/observability/langfuse',headers={'Authorization':'Bearer synthetic-customer-token'})
    assert response.status_code==200
    assert response.json()['status']=='connected'


@pytest.mark.parametrize('path', ['/chat', '/ingest'])
def test_authentication_required(client, path):
    assert client.post(path, json={}).status_code == 401


def test_scope_validation_unknown_fields_and_dependency_errors(client):
    headers={'Authorization':'Bearer synthetic-customer-token'}
    assert client.post('/chat', headers=headers, json={'user_id':'U0000000002','question':'x'}).status_code == 403
    invalid=client.post('/chat', headers=headers, json={'user_id':'U0000000001','question':'secret raw text','extra':'SECRET'})
    assert invalid.status_code == 422
    assert 'SECRET' not in invalid.text and 'raw text' not in invalid.text
    assert client.get('/customers/U0000000001/summary', headers=headers).status_code == 503
    assert client.get('/customers/U0000000001/revenue', headers=headers).status_code == 503
    assert client.get('/ingest/missing', headers=headers).status_code == 403
    assert client.get('/ingest/missing', headers={'Authorization':'Bearer synthetic-admin-token'}).status_code == 404


def test_rate_limit_injected_clock(credential_mapping):
    now=[0.0]
    client=TestClient(create_app(Settings(credential_mapping=credential_mapping, rate_limit_per_minute=1), AppServices(assistant=Assistant()), clock=lambda:now[0]))
    headers={'Authorization':'Bearer synthetic-customer-token'}
    assert client.post('/chat', headers=headers, json={'user_id':'U0000000001','question':'x'}).status_code == 200
    assert client.post('/chat', headers=headers, json={'user_id':'U0000000001','question':'x'}).status_code == 429
    now[0]=61
    assert client.post('/chat', headers=headers, json={'user_id':'U0000000001','question':'x'}).status_code == 200


def test_timeout_and_safe_provider_failure(credential_mapping):
    class Broken:
        async def answer(self, *_):
            raise RuntimeError('SECRET PROVIDER KEY')
    settings=Settings(credential_mapping=credential_mapping, request_timeout_seconds=.01)
    with TestClient(create_app(settings, AppServices(assistant=Broken()))) as client:
        response=client.post('/chat', headers={'Authorization':'Bearer synthetic-customer-token'},json={'user_id':'U0000000001','question':'x'})
        assert response.status_code == 503 and 'SECRET' not in response.text
    class Slow:
        async def answer(self,*_):
            await asyncio.sleep(1)
    with TestClient(create_app(settings, AppServices(assistant=Slow()))) as client:
        assert client.post('/chat',headers={'Authorization':'Bearer synthetic-customer-token'},json={'user_id':'U0000000001','question':'x'}).status_code == 504


@pytest.mark.asyncio
async def test_ingestion_single_worker_and_exception_releases_lock():
    release=threading.Event()
    def ingest():
        release.wait(1)
        raise RuntimeError('SECRET')
    jobs=IngestionJobs(ingest)
    principal=Principal(subject='admin',customer_ids=set(),ingest_admin=True)
    first=await jobs.start(principal)
    from app.errors import ConflictError
    with pytest.raises(ConflictError):
        await jobs.start(principal)
    release.set()
    await jobs.wait()
    assert jobs.get(principal,first.job_id).status == 'failed'
    assert 'SECRET' not in jobs.get(principal,first.job_id).model_dump_json()
    await jobs.start(principal)
    await jobs.wait()

@pytest.mark.asyncio
async def test_cancelled_job_keeps_lock_until_thread_exits():
    entered=threading.Event(); release=threading.Event()
    def ingest():
        entered.set(); release.wait(2)
    jobs=IngestionJobs(ingest)
    principal=Principal(subject='admin',customer_ids=set(),ingest_admin=True)
    first=await jobs.start(principal)
    await asyncio.to_thread(entered.wait,1)
    jobs.task.cancel()
    await asyncio.sleep(.01)
    from app.errors import ConflictError
    with pytest.raises(ConflictError):
        await jobs.start(principal)
    release.set()
    await jobs.wait()
    assert jobs.get(principal,first.job_id).status == 'failed'


def test_readiness_does_not_report_unready_dependencies_as_ready(credential_mapping):
    client=TestClient(create_app(Settings(credential_mapping=credential_mapping), AppServices(readiness=lambda:{'index':'unavailable'})))
    assert client.get('/health').status_code == 503


def test_bounded_body_and_untrusted_correlation_id(client):
    response=client.post('/chat',content='x'*16385,headers={'x-correlation-id':'unsafe!'})
    assert response.status_code == 413
    assert response.headers['x-correlation-id'] != 'unsafe!'


def test_real_analytics_routes_preserve_values_and_scope(tmp_path,credential_mapping):
    from app.data.ingestion import stage_dataset
    from app.data.repository import SourceRepository
    from app.services.analytics import AnalyticsService
    from tests.test_data_quality import configured
    service=AnalyticsService(SourceRepository(stage_dataset(configured(tmp_path)).sqlite_path))
    client=TestClient(create_app(Settings(credential_mapping=credential_mapping),AppServices(assistant=Assistant(),analytics=service)))
    headers={'Authorization':'Bearer synthetic-customer-token'}
    summary=client.get('/customers/U0000000001/summary',headers=headers)
    assert summary.status_code==200
    assert summary.json()['domains']['payments']['values']['PaymentAttemptCount']==7
    revenue=client.get('/customers/U0000000001/revenue',headers=headers)
    assert revenue.status_code==200
    assert revenue.json()['total_paise']==9007199254741000
    assert revenue.json()['complete'] is False
    assert client.get('/customers/U0000000002/revenue',headers=headers).status_code==403


def test_ingest_job_http_lifecycle_and_conflict(credential_mapping):
    entered=threading.Event(); release=threading.Event()
    def ingest():
        entered.set(); release.wait(2)
    with TestClient(create_app(Settings(credential_mapping=credential_mapping),AppServices(ingestion=ingest))) as client:
        headers={'Authorization':'Bearer synthetic-admin-token'}
        response=client.post('/ingest',headers=headers)
        assert response.status_code==202
        job_id=response.json()['job_id']
        assert entered.wait(1)
        assert client.get('/ingest/'+job_id,headers=headers).json()['status']=='running'
        assert client.post('/ingest',headers=headers).status_code==409
        release.set()


def test_production_bootstrap_ingestion_refreshes_snapshot_without_demo(tmp_path,credential_mapping,monkeypatch):
    from types import SimpleNamespace
    from app.rag import store
    from app.rag.embeddings import embedding_signature
    from app.data.ingestion import stage_dataset
    from tests.test_data_quality import configured
    settings=configured(tmp_path).model_copy(update={'openai_api_key':__import__('pydantic').SecretStr('synthetic-not-used')})
    staged=stage_dataset(settings)
    snapshot=store.ActiveSnapshot(version=staged.version,sqlite_path=staged.sqlite_path,collection_name='synthetic_bootstrap',schema_version=settings.schema_version,source_sha256=staged.source_sha256,embedding_signature=embedding_signature(settings),row_count=staged.row_count,document_count=staged.row_count)
    def missing(*_args,**_kwargs):
        raise DependencyError()
    monkeypatch.setattr(store,'load_active_snapshot',missing)
    monkeypatch.setattr(store,'ingest',lambda configured:SimpleNamespace(snapshot=snapshot))
    from app.main import _production_services
    services=_production_services(settings)
    assert services.assistant is None and services.ingestion is not None
    assert services.readiness()['index']=='unavailable'
    services.ingestion()
    assert services.demo is False
    assert services.assistant.snapshot is snapshot
    assert services.analytics.repository.sqlite_path==snapshot.sqlite_path
    assert services.readiness()['index']=='ready'


def test_ollama_readiness_checks_installed_chat_and_embedding_models(monkeypatch):
    from types import SimpleNamespace
    from app.main import _provider_readiness
    import httpx
    settings=SimpleNamespace(llm_provider='ollama',ollama_base_url='http://localhost:11434',model_id='ollama:qwen2.5:3b',embedding_provider='ollama',embedding_model='nomic-embed-text',openai_api_key=None)
    monkeypatch.setattr(httpx,'get',lambda *args,**kwargs:httpx.Response(200,json={'models':[{'name':'qwen2.5:3b'},{'name':'nomic-embed-text:latest'}]},request=httpx.Request('GET','http://localhost:11434/api/tags')))
    assert _provider_readiness(settings)=={'provider':'ready','embeddings':'ready'}
    monkeypatch.setattr(httpx,'get',lambda *args,**kwargs:httpx.Response(200,json={'models':[]},request=httpx.Request('GET','http://localhost:11434/api/tags')))
    assert _provider_readiness(settings)=={'provider':'unavailable','embeddings':'unavailable'}
