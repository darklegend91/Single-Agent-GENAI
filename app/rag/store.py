"""Stage both stores, then publish their immutable pair in one atomic manifest.

Previous published versions remain available to requests that already froze a
snapshot. A failed attempt owns and deletes only its unpublished artifacts.
"""
import fcntl
import json
import os
import re
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol
from uuid import uuid4

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from app.config import Settings
from app.data.ingestion import stage_dataset
from app.data.quality import source_checksum
from app.data.repository import SourceRepository
from app.errors import AppError, DependencyError, ValidationFailure
from app.rag.documents import build_document
from app.rag.embeddings import create_embeddings, embedding_signature


class ActiveSnapshot(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True)
    version: str = Field(pattern=r'\A[A-Za-z0-9][A-Za-z0-9_-]{0,127}\z')
    sqlite_path: Path
    collection_name: str = Field(pattern=r'\A[A-Za-z_][A-Za-z0-9_]{0,127}\z')
    schema_version: str
    source_sha256: str = Field(pattern=r'\A[0-9a-f]{64}\z')
    embedding_signature: str = Field(pattern=r'\A[0-9a-f]{64}\z')
    row_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    sample_limit: int | None = Field(default=None,gt=0)
    timestamp_formats: Mapping[str,str] = Field(default_factory=dict)
    metric: Literal['COSINE']='COSINE'

    @field_validator('timestamp_formats')
    @classmethod
    def immutable_formats(cls,value): return MappingProxyType(dict(value))

    @field_serializer('timestamp_formats')
    def serialize_formats(self,value): return dict(value)

    def repository(self) -> SourceRepository:
        """Use this same frozen snapshot for all source tools in a request."""
        return SourceRepository(self.sqlite_path)


class IngestionResult(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True)
    snapshot: ActiveSnapshot
    row_count: int
    sample_limit: int | None
    reused: bool=False


class VectorBackend(Protocol):
    def collection_for_version(self,version:str)->str: ...
    def stage(self,documents:Iterable[Document],version:str,embedding_signature:str)->str: ...
    def search(self,snapshot:ActiveSnapshot,question:str,*,k:int,expr:str)->list[tuple[Document,float]]: ...
    def validate(self,snapshot:ActiveSnapshot)->None: ...
    def discard(self,collection_name:str)->None: ...


class VectorStore:
    """LangChain Milvus adapter. COSINE is similarity: larger scores are better."""
    def __init__(self,settings:Settings,*,embeddings=None):
        self.settings=settings
        self.embeddings=embeddings if embeddings is not None else create_embeddings(settings)

    def _args(self):
        # PyMilvus defaults to a 10s keepalive. Milvus Lite rejects that cadence
        # as too_many_pings during long Ollama embedding batches, so use a
        # conservative interval for local and server connections alike.
        args={'uri':self.settings.milvus_uri,'grpc_options':{'grpc.keepalive_time_ms':300_000,'grpc.keepalive_timeout_ms':10_000,'grpc.keepalive_permit_without_calls':True}}
        if self.settings.milvus_token is not None: args['token']=self.settings.milvus_token.get_secret_value()
        return args

    def collection_for_version(self,version):
        # Keep within Milvus' name bound even for the longest configured prefix.
        return self.settings.milvus_collection[:85]+'_'+version

    def _client(self):
        from pymilvus import MilvusClient
        return MilvusClient(**self._args(),timeout=self.settings.request_timeout_seconds)

    def _store(self,name,description=''):
        from langchain_milvus import Milvus
        # Milvus Lite supports FLAT; server deployments use HNSW for scale.
        lite=self.settings.milvus_uri.endswith('.db') and '://' not in self.settings.milvus_uri
        index_params={'index_type':'FLAT','metric_type':'COSINE','params':{}} if lite else {'index_type':'HNSW','metric_type':'COSINE','params':{'M':16,'efConstruction':200}}
        search_params={'metric_type':'COSINE','params':{} if lite else {'ef':max(64,self.settings.top_k)}}
        # langchain-milvus 0.4 and pymilvus 3.0 both pass insert timeout;
        # setting Milvus.timeout duplicates that keyword on pymilvus 3.x.
        return Milvus(self.embeddings,collection_name=name,collection_description=description,connection_args=self._args(),auto_id=False,enable_dynamic_field=True,consistency_level='Strong',index_params=index_params,search_params=search_params)

    def stage(self,documents,version,embedding_signature):
        name=self.collection_for_version(version)
        description=json.dumps({'dataset_version':version,'embedding_signature':embedding_signature,'metric':'COSINE'},sort_keys=True)
        try:
            client=self._client()
            try:
                if client.has_collection(name): raise DependencyError()
            finally: client.close()
            store=self._store(name,description)
            batch=[]
            for document in documents:
                batch.append(document)
                if len(batch)>=self.settings.ingestion_batch_size:
                    store.add_documents(batch,ids=[d.id for d in batch]); batch=[]
            if batch: store.add_documents(batch,ids=[d.id for d in batch])
            client=self._client()
            try:
                if not client.has_collection(name): raise DependencyError()  # empty sources never become ready
                client.flush(collection_name=name,timeout=self.settings.request_timeout_seconds)
                client.load_collection(collection_name=name,timeout=self.settings.request_timeout_seconds)
            finally: client.close()
            return name
        except AppError: raise
        except Exception: raise DependencyError() from None

    def validate(self,snapshot):
        client=None
        try:
            client=self._client()
            info=client.describe_collection(snapshot.collection_name)
            description=json.loads(info.get('description') or '{}')
            expected_description={'dataset_version':snapshot.version,'embedding_signature':snapshot.embedding_signature,'metric':'COSINE'}
            # Milvus Lite currently omits collection descriptions; local snapshots
            # still bind the same values through the atomic manifest and SQLite.
            if description and description!=expected_description: raise DependencyError()
            if not description and not (self.settings.milvus_uri.endswith('.db') and '://' not in self.settings.milvus_uri): raise DependencyError()
            vector=next(field for field in info['fields'] if field['name']=='vector')
            if int(vector['params']['dim'])!=self.settings.embedding_dimensions: raise DependencyError()
            indexes=client.list_indexes(snapshot.collection_name,field_name='vector')
            if not indexes or any(client.describe_index(snapshot.collection_name,index)['metric_type']!='COSINE' for index in indexes): raise DependencyError()
            stats=client.get_collection_stats(snapshot.collection_name)
            if int(stats['row_count'])!=snapshot.document_count: raise DependencyError()
        except AppError: raise
        except Exception: raise DependencyError() from None
        finally:
            if client is not None: client.close()

    def search(self,snapshot,question,*,k,expr):
        try: return self._store(snapshot.collection_name).similarity_search_with_score(question,k=k,expr=expr,timeout=self.settings.request_timeout_seconds)
        except Exception: raise DependencyError() from None

    def discard(self,collection_name):
        client=None
        try:
            client=self._client()
            if client.has_collection(collection_name): client.drop_collection(collection_name)
        except Exception: raise DependencyError() from None
        finally:
            if client is not None: client.close()


def validate_snapshot_settings(snapshot:ActiveSnapshot,settings:Settings):
    if snapshot.embedding_signature!=embedding_signature(settings) or snapshot.schema_version!=settings.schema_version or dict(snapshot.timestamp_formats)!=settings.timestamp_formats or snapshot.metric!='COSINE' or not 1<=snapshot.document_count<=snapshot.row_count:
        raise DependencyError()


def _validate_sqlite(snapshot,settings):
    path=snapshot.sqlite_path
    if not path.is_absolute() or path.resolve().parent!=settings.staging_dir.resolve() or path.name!=snapshot.version+'.sqlite3': raise DependencyError()
    connection=None
    try:
        connection=sqlite3.connect(f'{path.as_uri()}?mode=ro',uri=True)
        metadata=dict(connection.execute('SELECT key,value FROM dataset_metadata'))
        wanted={'version':snapshot.version,'schema_version':snapshot.schema_version,'source_sha256':snapshot.source_sha256,'row_count':str(snapshot.row_count),'sample_limit':'' if snapshot.sample_limit is None else str(snapshot.sample_limit),'state':'staged','timestamp_formats':json.dumps(dict(snapshot.timestamp_formats),sort_keys=True,separators=(',',':')),'embedding_signature':snapshot.embedding_signature}
        if any(metadata.get(key)!=value for key,value in wanted.items()) or connection.execute('SELECT COUNT(*) FROM customers').fetchone()[0]!=snapshot.row_count:
            raise DependencyError()
    except (sqlite3.Error,OSError): raise DependencyError() from None
    finally:
        if connection is not None: connection.close()


def _seal_sqlite(staged,settings):
    """Bind parsing/embedding config to the source store before vector publication."""
    try:
        with sqlite3.connect(staged.sqlite_path) as connection:
            connection.executemany('INSERT INTO dataset_metadata VALUES (?,?)',[
                ('timestamp_formats',json.dumps(settings.timestamp_formats,sort_keys=True,separators=(',',':'))),
                ('embedding_signature',embedding_signature(settings)),
            ])
    except sqlite3.Error: raise DependencyError() from None


def load_active_snapshot(settings:Settings,*,vector_store:VectorBackend|None=None)->ActiveSnapshot:
    """Read exactly once at request start; share the result across all tools."""
    try: snapshot=ActiveSnapshot.model_validate_json(settings.active_manifest_path.read_bytes())
    except (OSError,ValueError): raise DependencyError() from None
    validate_snapshot_settings(snapshot,settings)
    _validate_sqlite(snapshot,settings)
    store=vector_store if vector_store is not None else VectorStore(settings)
    store.validate(snapshot)
    return snapshot


@contextmanager
def _ingestion_lock(settings):
    settings.active_manifest_path.parent.mkdir(parents=True,exist_ok=True)
    with settings.active_manifest_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


def _publish(settings,snapshot):
    temporary=settings.active_manifest_path.with_name('.'+settings.active_manifest_path.name+'.'+snapshot.version+'.tmp')
    try:
        with temporary.open('x') as stream:
            stream.write(snapshot.model_dump_json()); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,settings.active_manifest_path)
    finally: temporary.unlink(missing_ok=True)


def _active_version(settings):
    if not settings.active_manifest_path.exists(): return None
    try: return ActiveSnapshot.model_validate_json(settings.active_manifest_path.read_bytes()).version
    except (OSError,ValueError): raise DependencyError() from None


def _clean_abandoned(settings,store,active_version):
    """Only journaled unpublished attempts; never published historical snapshots."""
    settings.staging_dir.mkdir(parents=True,exist_ok=True)
    for marker in settings.staging_dir.glob('pending-*.json'):
        try:
            pending=json.loads(marker.read_text())
            version=pending['version']
            if not re.fullmatch(r'[0-9a-f]{32}',version) or marker.name!='pending-'+version+'.json': continue
            if version==active_version:
                marker.unlink(missing_ok=True); continue
            collection_name=store.collection_for_version(version)
            if pending.get('collection_name')!=collection_name: continue
            store.discard(collection_name)
            (settings.staging_dir/(version+'.sqlite3')).unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
        except (ValueError,KeyError,TypeError): continue


def ingest(settings:Settings,limit:int|None=None,*,vector_store:VectorBackend|None=None)->IngestionResult:
    if limit is not None and (type(limit) is not int or limit<1): raise ValidationFailure()
    store=vector_store if vector_store is not None else VectorStore(settings)
    try:
        with _ingestion_lock(settings):
            _clean_abandoned(settings,store,_active_version(settings))
            checksum=source_checksum(settings.dataset_path)
            if settings.active_manifest_path.exists():
                # Re-ingestion with changed configuration is legal, but cannot reuse the old index.
                try: active=ActiveSnapshot.model_validate_json(settings.active_manifest_path.read_bytes())
                except (OSError,ValueError): raise DependencyError() from None
                expected_document_count=min(active.row_count,settings.vector_index_limit) if settings.vector_index_limit is not None else active.row_count
                if active.source_sha256==checksum and active.sample_limit==limit and active.schema_version==settings.schema_version and active.embedding_signature==embedding_signature(settings) and active.document_count==expected_document_count and dict(active.timestamp_formats)==settings.timestamp_formats:
                    active=load_active_snapshot(settings,vector_store=store)
                    return IngestionResult(snapshot=active,row_count=active.row_count,sample_limit=limit,reused=True)
            version=uuid4().hex
            collection_name=store.collection_for_version(version)
            staged=None
            marker=settings.staging_dir/('pending-'+version+'.json')
            marker.write_text(json.dumps({'version':version,'collection_name':collection_name}))
            try:
                staged=stage_dataset(settings,limit=limit,version=version)
                _seal_sqlite(staged,settings)
                signature=embedding_signature(settings)
                def documents():
                    records=SourceRepository(staged.sqlite_path).iter_records(settings.ingestion_batch_size)
                    if settings.vector_index_limit is not None:
                        from itertools import islice
                        records=islice(records,settings.vector_index_limit)
                    for record in records:
                        document=build_document(record)
                        document.metadata['embedding_signature']=signature
                        yield document
                collection_name=store.stage(documents(),version,signature)
                document_count=min(staged.row_count,settings.vector_index_limit) if settings.vector_index_limit is not None else staged.row_count
                snapshot=ActiveSnapshot(version=version,sqlite_path=staged.sqlite_path,collection_name=collection_name,schema_version=staged.schema_version,source_sha256=staged.source_sha256,embedding_signature=signature,row_count=staged.row_count,document_count=document_count,sample_limit=limit,timestamp_formats=settings.timestamp_formats)
                if snapshot.source_sha256!=checksum or source_checksum(settings.dataset_path)!=checksum: raise DependencyError()
                _validate_sqlite(snapshot,settings); store.validate(snapshot)
                _publish(settings,snapshot)
                marker.unlink(missing_ok=True)
                return IngestionResult(snapshot=snapshot,row_count=snapshot.row_count,sample_limit=limit)
            except BaseException:
                # The atomic commit may have happened immediately before a process
                # error. Never delete artifacts now referenced by the active pointer.
                if _active_version(settings)!=version:
                    if staged is not None: staged.sqlite_path.unlink(missing_ok=True)
                    try:
                        store.discard(collection_name)
                        marker.unlink(missing_ok=True)
                    except Exception: pass  # leave journal for a later cleanup attempt
                raise
    except AppError: raise
    except Exception: raise DependencyError() from None
