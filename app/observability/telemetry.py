"""Manual allowlisted tracing; customer text never enters this boundary."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import hmac
import secrets
import time
from typing import Any

current_correlation_id = ContextVar("correlation_id", default="local")
MEASURED_STAGES = frozenset({'request','authentication','authorization','routing','retrieval','context','tools','generation','validation'})


def langfuse_trace_id(correlation_id:str)->str:
    """Create a stable, opaque Langfuse trace identifier from the request ID."""
    return hashlib.sha256(correlation_id.encode()).hexdigest()[:32]

ALLOWED = frozenset({'status','document_count','context_chars','context_tokens','input_tokens','output_tokens','total_tokens','request_count','tool_calls','claim_count','evidence_count','retry_count','validation_passed','cost_known','cost_usd','model','dataset_version','prompt_version','principal_id','customer_id','error_code','mode','duration_ms','complete','available_months'})

class RecordingAdapter:
    def __init__(self):
        self.events: list[dict[str, Any]] = []
    def emit(self,event):
        self.events.append(event)

class LangfuseAdapter:
    """Uses the pinned SDK's manual API; never enable automatic model instrumentation."""
    def __init__(self,settings):
        from langfuse import Langfuse
        self.client=Langfuse(public_key=settings.langfuse_public_key.get_secret_value(),secret_key=settings.langfuse_secret_key.get_secret_value(),base_url=settings.langfuse_base_url)
    def emit(self,event):
        metadata={**event['metadata'],'duration_ms':event['duration_ms']}
        usage={k:event['metadata'][k] for k in ('input_tokens','output_tokens','total_tokens') if k in event['metadata'] and isinstance(event['metadata'][k],int)}
        trace_id=langfuse_trace_id(event['correlation_id'])
        kwargs={'trace_context':{'trace_id':trace_id},'name':event['name'],'as_type':'generation' if event['name']=='generation' else 'span','metadata':metadata}
        if usage and kwargs['as_type']=='generation':
            kwargs['usage_details']=usage
        with self.client.start_as_current_observation(**kwargs):
            pass
        duration=event['duration_ms']
        if event['name'] in MEASURED_STAGES and isinstance(duration,(int,float)) and duration >= 0:
            self._score(trace_id,f"{event['name']}_latency_ms",float(duration),'NUMERIC')
        if event['name']=='generation':
            for source,name in (('total_tokens','model_total_tokens'),('tool_calls','agent_tool_calls'),('request_count','model_requests')):
                value=event['metadata'].get(source)
                if isinstance(value,int) and value >= 0:
                    self._score(trace_id,name,float(value),'NUMERIC')
        if event['name']=='extractive_fast_path':
            value=event['metadata'].get('tool_calls')
            if isinstance(value,int) and value >= 0:
                self._score(trace_id,'agent_tool_calls',float(value),'NUMERIC')
        passed=event['metadata'].get('validation_passed')
        if type(passed) is bool:
            self._score(trace_id,'claim_validation',int(passed),'BOOLEAN')
        if event['name']=='final_status':
            status=event['metadata'].get('status')
            if status in {'answered','insufficient_evidence','refused'}:
                self._score(trace_id,'answer_status',status,'CATEGORICAL')
                evidence_count=event['metadata'].get('evidence_count')
                if isinstance(evidence_count,int):
                    self._score(trace_id,'answer_has_source_evidence',int(evidence_count>0),'BOOLEAN')

    def _score(self,trace_id,name,value,data_type):
        self.client.create_score(name=name,value=value,data_type=data_type,trace_id=trace_id)
    def check_connection(self):
        """Verify project credentials without creating a trace or exposing keys."""
        return bool(self.client.auth_check())
    def close(self):
        self.client.flush()

class Telemetry:
    def __init__(self,adapter=None,*,salt:bytes|None=None):
        self.adapter=adapter
        self.salt=salt or secrets.token_bytes(32)
        self.degraded=False
    def pseudonym(self,value:str)->str:
        return hmac.new(self.salt,str(value).encode(),hashlib.sha256).hexdigest()[:24]
    def _metadata(self,metadata):
        safe={}
        for key,value in (metadata or {}).items():
            if key not in ALLOWED or not (value is None or isinstance(value,(str,int,float,bool))):
                continue
            if isinstance(value,str):
                value=value[:128]
            if key in {'principal_id','customer_id'} and value is not None:
                value=self.pseudonym(value)
            safe[key]=value
        return safe
    def record(self,name,correlation_id,**safe_metadata):
        event={'name':str(name)[:64],'correlation_id':str(correlation_id)[:64],'metadata':self._metadata(safe_metadata),'duration_ms':float(safe_metadata.get('duration_ms',0))}
        if self.adapter:
            try:
                self.adapter.emit(event)
            except Exception:
                self.degraded=True
    @contextmanager
    def stage(self,name,correlation_id,safe_metadata=None):
        started=time.perf_counter()
        metadata=dict(safe_metadata or {})
        try:
            yield metadata
        except BaseException:
            metadata['status']='failed'
            raise
        finally:
            metadata['duration_ms']=(time.perf_counter()-started)*1000
            self.record(name,correlation_id,**metadata)
    def close(self):
        if self.adapter and hasattr(self.adapter,'close'):
            try:
                self.adapter.close()
            except Exception:
                self.degraded=True
