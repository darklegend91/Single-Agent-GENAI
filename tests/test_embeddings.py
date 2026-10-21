"""Explicit native embedding providers, dimensions and controlled readiness failures."""
import sys
from types import SimpleNamespace
import pytest
from langchain_core.embeddings import Embeddings
from app.config import Settings
from app.errors import DependencyError
from app.rag.embeddings import create_embeddings, embedding_signature


def config(**overrides):
    settings=Settings(_env_file=None,credential_mapping={'test-token':{'subject':'test','customer_ids':['U0000000001']}})
    return settings.model_copy(update={'embedding_provider':'ollama','embedding_model':'nomic-embed-text','embedding_dimensions':3,'ollama_base_url':'http://127.0.0.1:11434',**overrides})


class EmbeddingBackend(Embeddings):
    def __init__(self,vectors=None,error=None): self.vectors=vectors if vectors is not None else [[0.1,0.2,0.3]]; self.error=error; self.texts=[]
    def embed_documents(self,texts):
        self.texts.extend(texts)
        if self.error: raise self.error
        return self.vectors
    def embed_query(self,text):
        self.texts.append(text)
        if self.error: raise self.error
        return self.vectors[0]


def test_native_ollama_embeddings_need_no_cloud_credential(monkeypatch):
    backend=EmbeddingBackend(); calls=[]
    def factory(**kwargs): calls.append(kwargs); return backend
    monkeypatch.setitem(sys.modules,'langchain_ollama',SimpleNamespace(OllamaEmbeddings=factory))
    embeddings=create_embeddings(config())
    assert embeddings.embed_query('semantic query')==[0.1,0.2,0.3]
    assert calls==[{'model':'nomic-embed-text','base_url':'http://127.0.0.1:11434','client_kwargs':{'timeout':120.0}}]


@pytest.mark.parametrize('vectors',[[[0.1,0.2]],[[0.1,float('nan'),0.3]],[[0.0,0.0,0.0]],[[True,0.2,0.3]],[]])
def test_provider_malformed_vectors_fail_before_indexing(monkeypatch,vectors):
    monkeypatch.setitem(sys.modules,'langchain_ollama',SimpleNamespace(OllamaEmbeddings=lambda **kwargs:EmbeddingBackend(vectors)))
    embeddings=create_embeddings(config())
    with pytest.raises(DependencyError): embeddings.embed_documents(['customer summary'])


def test_native_embedding_failure_is_controlled_and_never_cloud_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules,'langchain_ollama',SimpleNamespace(OllamaEmbeddings=lambda **kwargs:EmbeddingBackend(error=ConnectionError('native endpoint unavailable'))))
    embeddings=create_embeddings(config())
    with pytest.raises(DependencyError): embeddings.embed_query('query')


def test_signature_separates_native_endpoints():
    assert embedding_signature(config())!=embedding_signature(config(ollama_base_url='http://127.0.0.1:11435'))


def test_readiness_checks_actual_vector_dimension_without_customer_content():
    from app.rag.embeddings import check_embedding_readiness
    backend=EmbeddingBackend()
    check_embedding_readiness(config(),embeddings=backend)
    assert backend.texts==['embedding readiness probe']
    with pytest.raises(DependencyError): check_embedding_readiness(config(embedding_dimensions=4),embeddings=backend)


def test_openai_is_only_used_when_explicitly_configured():
    with pytest.raises(DependencyError): create_embeddings(config(embedding_provider='openai'))


@pytest.mark.parametrize('uri,expected',[('/tmp/customer-local.db','FLAT'),('http://localhost:19530','HNSW')])
def test_milvus_lite_uses_supported_flat_index(monkeypatch,uri,expected):
    from app.rag.store import VectorStore
    import langchain_milvus
    captured=[]
    monkeypatch.setattr(langchain_milvus,'Milvus',lambda *args,**kwargs:captured.append(kwargs))
    store=VectorStore(config(milvus_uri=uri),embeddings=EmbeddingBackend())
    store._store('summary_collection')
    assert captured[0]['index_params']['index_type']==expected
    assert captured[0]['index_params']['metric_type']=='COSINE'
    if expected=='FLAT': assert captured[0]['search_params']['params']=={}
