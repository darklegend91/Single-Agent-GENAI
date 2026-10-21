"""Real SQLite/manifest lifecycle with explicit external-vector injection."""
import json
from pathlib import Path
import pytest
from app.config import Settings
from app.errors import DependencyError
from app.rag.store import ingest, load_active_snapshot


class MemoryVectors:
    def __init__(self, batch_size=1): self.collections={}; self.fail=False; self.batch_size=batch_size; self.max_batch=0
    def collection_for_version(self,version): return 'test_'+version
    def stage(self, documents, version, embedding_signature):
        name=self.collection_for_version(version); self.collections[name]=[]; batch=[]
        for document in documents:
            batch.append(document)
            if len(batch)==self.batch_size:
                self.max_batch=max(self.max_batch,len(batch)); self.collections[name].extend(batch); batch=[]
                if self.fail: raise DependencyError()
        self.max_batch=max(self.max_batch,len(batch)); self.collections[name].extend(batch)
        return name
    def validate(self, snapshot):
        if len(self.collections.get(snapshot.collection_name,[]))!=snapshot.document_count: raise DependencyError()
    def discard(self, collection_name): self.collections.pop(collection_name,None)


def config(tmp_path):
    return Settings(credential_mapping={'token':{'subject':'test','customer_ids':['U0000000001']}},dataset_path=Path('tests/fixtures/customers.csv'),staging_dir=tmp_path/'staging',active_manifest_path=tmp_path/'active.json',ingestion_batch_size=1)


def test_success_repeat_and_explicit_sample_limit(tmp_path):
    cfg=config(tmp_path); store=MemoryVectors()
    result=ingest(cfg,limit=1,vector_store=store)
    assert result.sample_limit==1 and result.row_count==1 and result.reused is False
    original=cfg.active_manifest_path.read_bytes()
    repeat=ingest(cfg,limit=1,vector_store=store)
    assert repeat.reused and repeat.snapshot==result.snapshot
    assert cfg.active_manifest_path.read_bytes()==original and len(store.collections)==1
    full=ingest(cfg,vector_store=store)
    assert full.row_count==2 and full.sample_limit is None and full.snapshot.version!=result.snapshot.version
    assert store.max_batch<=cfg.ingestion_batch_size
    assert load_active_snapshot(cfg,vector_store=store)==full.snapshot
    assert result.snapshot.repository().lookup(next(iter(cfg.credential_mapping.values())),'U0000000001').provenance['UserID'].dataset_version==result.snapshot.version


def test_full_source_can_use_compact_vector_index(tmp_path):
    cfg=config(tmp_path).model_copy(update={'vector_index_limit':1})
    store=MemoryVectors()
    result=ingest(cfg,vector_store=store)
    assert result.row_count==2
    assert result.snapshot.document_count==1
    assert len(store.collections[result.snapshot.collection_name])==1
    assert load_active_snapshot(cfg,vector_store=store)==result.snapshot


@pytest.mark.parametrize('change',[{'schema_version':'2'},{'embedding_model':'new'},{'timestamp_formats':{'RegistrationAtUTC':'%Y%m%d%H%M%S'}}])
def test_idempotence_covers_config(tmp_path,change):
    cfg=config(tmp_path); store=MemoryVectors(); original=ingest(cfg,vector_store=store)
    newer=ingest(cfg.model_copy(update=change),vector_store=store)
    assert not newer.reused and newer.snapshot.version!=original.snapshot.version


@pytest.mark.parametrize('boundary',['sqlite','vectors','manifest'])
def test_failed_staging_preserves_active_manifest(tmp_path,monkeypatch,boundary):
    import app.rag.store as module
    cfg=config(tmp_path); store=MemoryVectors(); old=ingest(cfg,limit=1,vector_store=store)
    original=cfg.active_manifest_path.read_bytes()
    if boundary=='sqlite':
        def fail(*args,**kwargs): raise DependencyError()
        monkeypatch.setattr(module,'stage_dataset',fail)
    elif boundary=='vectors': store.fail=True
    else:
        def fail(*args,**kwargs): raise OSError('publication unavailable')
        monkeypatch.setattr(module.os,'replace',fail)
    with pytest.raises(DependencyError): ingest(cfg,vector_store=store)
    assert cfg.active_manifest_path.read_bytes()==original
    assert old.snapshot.sqlite_path.is_file() and old.snapshot.collection_name in store.collections
    assert list(cfg.staging_dir.glob('*.sqlite3'))==[old.snapshot.sqlite_path]
    assert list(store.collections)==[old.snapshot.collection_name]


def test_manifest_sqlite_mismatch_rejects_readiness(tmp_path):
    import sqlite3
    cfg=config(tmp_path); store=MemoryVectors(); result=ingest(cfg,vector_store=store)
    with sqlite3.connect(result.snapshot.sqlite_path) as conn: conn.execute("UPDATE dataset_metadata SET value='wrong' WHERE key='version'")
    with pytest.raises(DependencyError): load_active_snapshot(cfg,vector_store=store)


def test_opt_in_real_milvus(tmp_path):
    import os
    if os.environ.get('KA_RUN_LIVE_MILVUS')!='1': pytest.skip('Opt-in live Milvus disabled; set KA_RUN_LIVE_MILVUS=1 with Milvus and OpenAI credentials')
    if not os.environ.get('KA_MILVUS_URI') or not os.environ.get('KA_OPENAI_API_KEY'): pytest.skip('Real Milvus/provider credentials absent')
    cfg=config(tmp_path).model_copy(update={'milvus_uri':os.environ['KA_MILVUS_URI'],'openai_api_key':os.environ['KA_OPENAI_API_KEY']})
    from pydantic import SecretStr
    cfg=cfg.model_copy(update={'openai_api_key':SecretStr(os.environ['KA_OPENAI_API_KEY'])})
    result=ingest(cfg,limit=1)
    from app.rag.retriever import Retriever
    principal=next(iter(cfg.credential_mapping.values()))
    hits=Retriever(cfg,result.snapshot).retrieve(principal,'U0000000001','What is the plan price?')
    assert hits and hits[0].document.metadata['user_id']=='U0000000001'


def test_failure_after_sqlite_completion_removes_only_new_artifacts(tmp_path,monkeypatch):
    import app.rag.store as module
    cfg=config(tmp_path); store=MemoryVectors(); old=ingest(cfg,limit=1,vector_store=store)
    original=cfg.active_manifest_path.read_bytes()
    def fail(snapshot): raise DependencyError()
    monkeypatch.setattr(store,'validate',fail)
    with pytest.raises(DependencyError): ingest(cfg,vector_store=store)
    assert cfg.active_manifest_path.read_bytes()==original
    assert list(cfg.staging_dir.glob('*.sqlite3'))==[old.snapshot.sqlite_path]
    assert list(store.collections)==[old.snapshot.collection_name]


def test_manifest_timestamp_tampering_cannot_match_differently_parsed_sqlite(tmp_path):
    cfg=config(tmp_path); store=MemoryVectors(); result=ingest(cfg,vector_store=store)
    changed={'RegistrationAtUTC':'%Y%m%d%H%M%S'}
    raw=json.loads(cfg.active_manifest_path.read_text()); raw['timestamp_formats']=changed
    cfg.active_manifest_path.write_text(json.dumps(raw))
    with pytest.raises(DependencyError): load_active_snapshot(cfg.model_copy(update={'timestamp_formats':changed}),vector_store=store)


def test_missing_or_wrong_index_rejects_readiness(tmp_path):
    cfg=config(tmp_path); store=MemoryVectors(); result=ingest(cfg,vector_store=store)
    store.collections.clear()
    with pytest.raises(DependencyError): load_active_snapshot(cfg,vector_store=store)


def test_real_adapter_batches_documents_without_materializing_input(tmp_path,monkeypatch):
    from app.rag.store import VectorStore
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    class EmbeddingStub(Embeddings):
        def embed_documents(self,texts): return [[0.0,1.0] for _ in texts]
        def embed_query(self,text): return [0.0,1.0]
    class Client:
        exists=False
        def has_collection(self,name): return self.exists
        def flush(self,**kwargs): pass
        def load_collection(self,**kwargs): pass
        def close(self): pass
    client=Client(); batches=[]; consumed=[]
    class ExternalWriter:
        def add_documents(self,documents,ids):
            assert len(consumed)==sum(batches)+len(documents)
            assert ids==[d.id for d in documents]
            batches.append(len(documents)); client.exists=True
    cfg=config(tmp_path).model_copy(update={'ingestion_batch_size':2})
    store=VectorStore(cfg,embeddings=EmbeddingStub())
    monkeypatch.setattr(store,'_client',lambda:client)
    monkeypatch.setattr(store,'_store',lambda *args:ExternalWriter())
    def source():
        for i in range(5):
            consumed.append(i); yield Document(id=str(i),page_content='synthetic')
    store.stage(source(),'v1','a'*64)
    assert batches==[2,2,1]


def test_abandoned_unpublished_artifacts_cleaned_but_active_snapshot_preserved(tmp_path):
    cfg=config(tmp_path); store=MemoryVectors(); old=ingest(cfg,limit=1,vector_store=store)
    abandoned='a'*32; stale=cfg.staging_dir/(abandoned+'.sqlite3'); stale.write_bytes(b'abandoned')
    store.collections[store.collection_for_version(abandoned)]=[]
    (cfg.staging_dir/('pending-'+abandoned+'.json')).write_text(json.dumps({'version':abandoned,'collection_name':store.collection_for_version(abandoned)}))
    (cfg.staging_dir/('pending-'+old.snapshot.version+'.json')).write_text(json.dumps({'version':old.snapshot.version,'collection_name':old.snapshot.collection_name}))
    again=ingest(cfg,limit=1,vector_store=store)
    assert again.reused and not stale.exists()
    assert list(store.collections)==[old.snapshot.collection_name]
    assert old.snapshot.sqlite_path.exists()
    assert not list(cfg.staging_dir.glob('pending-*.json'))


def test_publication_that_committed_before_error_keeps_new_active_artifacts(tmp_path,monkeypatch):
    import app.rag.store as module
    cfg=config(tmp_path); store=MemoryVectors(); old=ingest(cfg,limit=1,vector_store=store)
    original=module._publish
    def commit_then_fail(settings,snapshot):
        original(settings,snapshot)
        raise OSError('process failure after atomic publication')
    monkeypatch.setattr(module,'_publish',commit_then_fail)
    with pytest.raises(DependencyError): ingest(cfg,vector_store=store)
    active=load_active_snapshot(cfg,vector_store=store)
    assert active.version!=old.snapshot.version and active.row_count==2
    assert old.snapshot.sqlite_path.exists()
