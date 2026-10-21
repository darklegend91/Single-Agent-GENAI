"""Allowlist source facts and bounded deterministic results as untrusted JSON data."""
from dataclasses import dataclass
import json
from langchain_core.documents import Document
from app.data.dictionary import FIELD_DICTIONARY
from app.errors import DependencyError
from app.rag.retriever import RetrievedEvidence
from app.security import authorize


@dataclass(frozen=True)
class ContextBundle:
    text: str
    documents: tuple[Document,...]
    source_fields: frozenset[str]
    limitations: tuple[str,...]


def build_context(question,principal,user_id,documents,results,settings)->ContextBundle:
    authorize(principal,user_id)
    # ASCII serialization makes a byte a conservative token upper bound; both budgets hold.
    budget=min(settings.context_max_chars,settings.context_max_tokens)
    chunks=[]; accepted=[]; fields=set(); limitations=set()
    for candidate in documents:
        if len(accepted)>=settings.max_documents: break
        document=candidate.document if isinstance(candidate,RetrievedEvidence) else candidate
        if not isinstance(document,Document) or document.metadata.get('user_id')!=user_id: raise DependencyError()
        try:
            payload=json.loads(document.page_content)
            if payload.get('user_id')!=user_id or not isinstance(payload.get('fields'),dict): raise DependencyError()
            allowed=set(document.metadata.get('source_fields',[])) & FIELD_DICTIONARY.keys()
            values={key:value for key,value in payload['fields'].items() if key in allowed and (value is None or type(value) in (str,int,bool))}
        except (ValueError,TypeError): raise DependencyError() from None
        # Add complete JSON scalar facts only; never cut a quoted value or provenance marker.
        selected={}
        for key,value in values.items():
            proposed=json.dumps({'user_id':user_id,'fields':{**selected,key:value}},sort_keys=True,separators=(',',':'),ensure_ascii=True)
            if len('\n'.join([*chunks,proposed]))<=budget: selected[key]=value
        if selected:
            content=json.dumps({'user_id':user_id,'fields':selected},sort_keys=True,separators=(',',':'),ensure_ascii=True)
            chunks.append(content); fields.update(selected)
            accepted.append(Document(id=document.id,page_content=content,metadata={**document.metadata,'source_fields':sorted(selected)}))
    # Deterministic results are intentionally not stringified wholesale. Source and
    # metric tools remain separate structured dependencies; arbitrary tool prose is excluded.
    if not accepted: limitations.add('no_relevant_evidence')
    return ContextBundle(text='\n'.join(chunks),documents=tuple(accepted),source_fields=frozenset(fields),limitations=tuple(sorted(limitations)))
