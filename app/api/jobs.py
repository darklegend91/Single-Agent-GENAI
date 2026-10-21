"""Single-process ingestion coordinator; publication belongs to the ingestion pipeline."""
import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from app.errors import AuthorizationError, ConflictError, DependencyError, NotFoundError


class JobStatus(BaseModel):
    model_config=ConfigDict(frozen=True)
    job_id:str
    status:str
    created_at:str
    error_code:str|None=None

class IngestionJobs:
    def __init__(self,worker):
        self.worker=worker
        self.jobs={}
        self.task=None
    @staticmethod
    def _authorize(principal):
        if not principal.ingest_admin:
            raise AuthorizationError()
    async def start(self,principal):
        self._authorize(principal)
        if self.worker is None:
            raise DependencyError()
        if self.task is not None and not self.task.done():
            raise ConflictError()
        job=JobStatus(job_id=uuid4().hex,status='running',created_at=datetime.now(timezone.utc).isoformat())
        if len(self.jobs)>=1000:
            self.jobs.pop(next(iter(self.jobs)))
        self.jobs[job.job_id]=job
        self.task=asyncio.create_task(self._run(job))
        return job
    async def _run(self,job):
        worker=asyncio.create_task(asyncio.to_thread(self.worker))
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Cancellation does not terminate a Python worker thread. Keep this job
            # active until publication/rollback finishes, preventing overlapping jobs.
            try:
                await asyncio.shield(worker)
            except BaseException:
                pass
            self.jobs[job.job_id]=job.model_copy(update={'status':'failed','error_code':'ingestion_cancelled'})
        except BaseException:
            self.jobs[job.job_id]=job.model_copy(update={'status':'failed','error_code':'ingestion_failed'})
        else:
            self.jobs[job.job_id]=job.model_copy(update={'status':'succeeded'})
    def get(self,principal,job_id):
        self._authorize(principal)
        if job_id not in self.jobs:
            raise NotFoundError()
        return self.jobs[job_id]
    async def wait(self):
        if self.task is not None:
            await asyncio.shield(self.task)
