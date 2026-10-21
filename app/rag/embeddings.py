"""Explicit native Ollama or optional OpenAI embeddings with validated dimensions."""
import hashlib
import json
import math
from langchain_core.embeddings import Embeddings
from app.config import Settings
from app.errors import DependencyError


def embedding_signature(settings: Settings) -> str:
    config={'provider':settings.embedding_provider,'model':settings.embedding_model,'dimensions':settings.embedding_dimensions,'metric':'COSINE','document_format':'customer-json-v2-summary'}
    if settings.embedding_provider=='ollama':
        config['base_url']=settings.ollama_base_url.rstrip('/')
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class ValidatedEmbeddings(Embeddings):
    """Reject wrong model dimensions and invalid vectors before Milvus receives them."""
    def __init__(self,backend:Embeddings,dimensions:int):
        self.backend=backend
        self.dimensions=dimensions

    def _validate(self,vectors,expected_count):
        if not isinstance(vectors,list) or len(vectors)!=expected_count:
            raise DependencyError()
        for vector in vectors:
            if not isinstance(vector,list) or len(vector)!=self.dimensions:
                raise DependencyError()
            if any(type(value) not in (int,float) or not math.isfinite(value) for value in vector) or not any(vector):
                raise DependencyError()
        return vectors

    def embed_documents(self,texts:list[str])->list[list[float]]:
        try:
            return self._validate(self.backend.embed_documents(texts),len(texts))
        except Exception:
            raise DependencyError() from None

    def embed_query(self,text:str)->list[float]:
        try:
            return self._validate([self.backend.embed_query(text)],1)[0]
        except Exception:
            raise DependencyError() from None


def create_embeddings(settings: Settings) -> Embeddings:
    """Use only the selected provider; provider failures never select another one."""
    try:
        if settings.embedding_provider=='ollama':
            from langchain_ollama import OllamaEmbeddings
            backend=OllamaEmbeddings(model=settings.embedding_model,base_url=settings.ollama_base_url,client_kwargs={'timeout':settings.request_timeout_seconds})
        elif settings.embedding_provider=='openai':
            if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value().strip():
                raise DependencyError()
            from langchain_openai import OpenAIEmbeddings
            backend=OpenAIEmbeddings(model=settings.embedding_model,dimensions=settings.embedding_dimensions,api_key=settings.openai_api_key,request_timeout=settings.request_timeout_seconds,max_retries=0)
        else:
            raise DependencyError()
        return ValidatedEmbeddings(backend,settings.embedding_dimensions)
    except Exception:
        raise DependencyError() from None


def check_embedding_readiness(settings:Settings,*,embeddings:Embeddings|None=None)->None:
    """Probe the actual selected endpoint/model with fixed noncustomer content."""
    backend=embeddings if embeddings is not None else create_embeddings(settings)
    ValidatedEmbeddings(backend,settings.embedding_dimensions).embed_query('embedding readiness probe')
