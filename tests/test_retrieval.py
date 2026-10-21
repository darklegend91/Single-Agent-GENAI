"""Identity checks must happen before top-K and again after vector retrieval."""
from dataclasses import replace
import pytest
from langchain_core.documents import Document
from app.config import Settings
from app.errors import AuthorizationError, DependencyError
from app.security import Principal
from app.rag.documents import build_document
from app.rag.retriever import Retriever
from app.rag.store import ActiveSnapshot, VectorStore


def settings(tmp_path):
    return Settings(credential_mapping={'token': {'subject':'test','customer_ids':['U0000000001']}}, active_manifest_path=tmp_path/'active.json')


def snapshot(tmp_path, settings):
    from app.rag.embeddings import embedding_signature
    return ActiveSnapshot(version='v1', sqlite_path=tmp_path/'v1.sqlite3', collection_name='customer_v1', schema_version='1', source_sha256='a'*64, embedding_signature=embedding_signature(settings), row_count=1, document_count=1, sample_limit=None, timestamp_formats={})


class SearchStore:
    def __init__(self, hits): self.hits=hits; self.calls=[]
    def search(self, snapshot, question, *, k, expr):
        self.calls.append((k,expr)); return self.hits


def doc(user='U0000000001', **extra):
    return Document(page_content='customer fact', metadata={'user_id':user,'dataset_version':'v1','schema_version':'1','source_sha256':'a'*64, 'embedding_signature':'', 'source_fields':['UserID'], **extra})


def test_filter_before_top_k_and_cosine_threshold(tmp_path):
    cfg=settings(tmp_path); snap=snapshot(tmp_path,cfg)
    hit=doc(embedding_signature=snap.embedding_signature)
    store=SearchStore([(hit,0.9),(hit,0.49)])
    results=Retriever(cfg,snap,store).retrieve(next(iter(cfg.credential_mapping.values())),'U0000000001','customer?')
    assert len(results)==1 and results[0].score==0.9
    assert store.calls==[(3,'user_id == "U0000000001"')]


@pytest.mark.parametrize('changed', [{'user_id':'U0000000002'},{'dataset_version':'v2'},{'schema_version':'2'},{'source_sha256':'b'*64},{'embedding_signature':'old'}])
def test_foreign_or_mismatched_metadata_fails_closed(tmp_path,changed):
    cfg=settings(tmp_path); snap=snapshot(tmp_path,cfg)
    with pytest.raises(DependencyError):
        Retriever(cfg,snap,SearchStore([(doc(embedding_signature=snap.embedding_signature,**{k:v for k,v in changed.items() if k!='embedding_signature'}),0.9)]) if 'embedding_signature' not in changed else SearchStore([(doc(embedding_signature='old'),0.9)])).retrieve(next(iter(cfg.credential_mapping.values())),'U0000000001','customer?')


def test_authorization_precedes_external_search(tmp_path):
    cfg=settings(tmp_path); snap=snapshot(tmp_path,cfg); store=SearchStore([])
    with pytest.raises(AuthorizationError):
        Retriever(cfg,snap,store).retrieve(Principal(subject='admin',customer_ids=frozenset(),ingest_admin=True),'U0000000001','customer?')
    assert store.calls==[]


def test_embedding_configuration_change_rejected(tmp_path):
    cfg=settings(tmp_path); snap=snapshot(tmp_path,cfg)
    with pytest.raises(DependencyError): Retriever(cfg.model_copy(update={'embedding_model':'changed'}),snap,SearchStore([]))


def test_milvus_client_uses_server_compatible_keepalive(tmp_path):
    args = VectorStore(settings(tmp_path), embeddings=object())._args()
    assert args['grpc_options'] == {
        'grpc.keepalive_time_ms': 300_000,
        'grpc.keepalive_timeout_ms': 10_000,
        'grpc.keepalive_permit_without_calls': True,
    }
