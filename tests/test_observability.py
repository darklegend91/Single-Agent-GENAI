from app.observability.telemetry import RecordingAdapter, Telemetry


def test_telemetry_allowlist_pseudonym_and_failures():
    adapter=RecordingAdapter()
    telemetry=Telemetry(adapter)
    with telemetry.stage('retrieval','correlation', {'query':'SECRET','record':'RAW','api_key':'PASSWORD','user_id':'U0000000001','document_count':2}):
        pass
    event=adapter.events[0]
    assert event['name']=='retrieval' and event['metadata']['document_count']==2
    assert event['duration_ms'] >= 0
    assert 'SECRET' not in str(event) and 'RAW' not in str(event) and 'PASSWORD' not in str(event)
    assert 'U0000000001' not in str(event)
    assert telemetry.pseudonym('subject') == telemetry.pseudonym('subject')
    assert telemetry.pseudonym('subject') != telemetry.pseudonym('different')


def test_export_degrades_without_breaking_request():
    class Broken:
        def emit(self,event):
            raise RuntimeError('secret')
    telemetry=Telemetry(Broken())
    with telemetry.stage('validation','id',{}):
        pass
    assert telemetry.degraded is True


def test_langfuse_adapter_groups_safe_observations_and_scores_without_inputs():
    from contextlib import contextmanager
    from app.observability.telemetry import LangfuseAdapter
    class Client:
        def __init__(self):
            self.observations=[]; self.scores=[]
        @contextmanager
        def start_as_current_observation(self,**kwargs):
            self.observations.append(kwargs)
            yield
        def create_score(self,**kwargs):
            self.scores.append(kwargs)
    adapter=object.__new__(LangfuseAdapter)
    adapter.client=Client()
    adapter.emit({'name':'generation','correlation_id':'same','metadata':{'input_tokens':10,'output_tokens':5,'total_tokens':15,'tool_calls':2,'request_count':1,'cost_known':False},'duration_ms':1.0})
    adapter.emit({'name':'validation','correlation_id':'same','metadata':{'validation_passed':True},'duration_ms':2.0})
    adapter.emit({'name':'request','correlation_id':'same','metadata':{'mode':'production'},'duration_ms':42.0})
    adapter.emit({'name':'final_status','correlation_id':'same','metadata':{'status':'answered','evidence_count':1},'duration_ms':0.0})
    observations=adapter.client.observations
    assert observations[0]['trace_context']['trace_id'] == observations[1]['trace_context']['trace_id']
    assert observations[0]['usage_details']=={'input_tokens':10,'output_tokens':5,'total_tokens':15}
    assert all('input' not in item and 'output' not in item for item in observations)
    scores={score['name']:score for score in adapter.client.scores}
    assert scores['claim_validation']['value']==1
    assert scores['model_total_tokens']['value']==15.0
    assert scores['agent_tool_calls']['value']==2.0
    assert scores['request_latency_ms']['value']==42.0
    assert scores['answer_status']['value']=='answered'
    assert scores['answer_has_source_evidence']['value']==1
    assert all(score['trace_id']==observations[0]['trace_context']['trace_id'] for score in adapter.client.scores)
    assert all('same' not in str(observation) for observation in observations)


import os
import pytest

@pytest.mark.live
@pytest.mark.skipif(os.environ.get('KA_RUN_LIVE_LANGFUSE')!='1',reason='Live Langfuse export requires explicit opt-in and project credentials.')
def test_opt_in_langfuse_export(credential_mapping):
    from app.config import Settings
    from app.observability.telemetry import LangfuseAdapter
    settings=Settings(credential_mapping=credential_mapping)
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        pytest.skip('Langfuse credentials unavailable; no live export was verified.')
    adapter=LangfuseAdapter(settings)
    assert adapter.client.auth_check()
    telemetry=Telemetry(adapter)
    with telemetry.stage('live_synthetic_export_check','synthetic-export-check',{'validation_passed':True,'mode':'synthetic'}):
        pass
    telemetry.close()
    assert not telemetry.degraded
