"""Authorize, apply Milvus expr before top-K, and distrust every returned row."""
from dataclasses import dataclass
import math
from langchain_core.documents import Document
from app.config import Settings
from app.errors import DependencyError, ValidationFailure
from app.rag.store import ActiveSnapshot, VectorStore, validate_snapshot_settings
from app.security import Principal, authorize


@dataclass(frozen=True)
class RetrievedEvidence:
    document: Document
    score: float


class Retriever:
    def __init__(self,settings:Settings,snapshot:ActiveSnapshot,vector_store=None):
        validate_snapshot_settings(snapshot,settings)
        self.settings=settings; self.snapshot=snapshot
        self.vector_store=vector_store if vector_store is not None else VectorStore(settings)

    def retrieve(self,principal:Principal,user_id:str,question:str)->list[RetrievedEvidence]:
        authorize(principal,user_id)
        if not isinstance(question,str) or not question.strip() or len(question)>self.settings.question_max_chars: raise ValidationFailure()
        # authorize validates the UserID grammar: interpolation cannot inject expr syntax.
        hits=self.vector_store.search(self.snapshot,question,k=self.settings.top_k,expr=f'user_id == "{user_id}"')
        results=[]
        for document,score in hits:
            expected={'user_id':user_id,'dataset_version':self.snapshot.version,'schema_version':self.snapshot.schema_version,'source_sha256':self.snapshot.source_sha256,'embedding_signature':self.snapshot.embedding_signature}
            if any(document.metadata.get(key)!=value for key,value in expected.items()): raise DependencyError()
            if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not -1<=score<=1: raise DependencyError()
            if score>=self.settings.relevance_threshold: results.append(RetrievedEvidence(document,score))
        return sorted(results,key=lambda result:result.score,reverse=True)[:self.settings.top_k]
