from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import principal
from app.errors import DependencyError
from app.schemas import ChatRequest, CustomerAnswer, UserID
from app.security import Principal, authorize

router=APIRouter()
Identity=Annotated[Principal,Depends(principal)]

@router.get('/health')
async def health(request:Request):
    services=request.app.state.services
    try:
        components=services.readiness() if services.readiness else {'services':'ready' if services.assistant is not None else 'unavailable'}
        ready=bool(components) and all(value=='ready' for value in components.values())
    except Exception:
        components={'services':'unavailable'}
        ready=False
    from fastapi.responses import JSONResponse
    return JSONResponse({'status':'ready' if ready else 'not_ready','mode':'demo' if services.demo else 'production','components':components,'telemetry':'degraded' if request.app.state.telemetry.degraded else 'configured' if request.app.state.telemetry.adapter else 'disabled'},status_code=200 if ready else 503)

@router.get('/observability/langfuse')
async def langfuse_status(request:Request,identity:Identity):
    """Check Langfuse project connectivity without sending customer content."""
    telemetry=request.app.state.telemetry
    adapter=telemetry.adapter
    if adapter is None:
        return {'status':'disabled','message':'Langfuse keys are not configured.'}
    checker=getattr(adapter,'check_connection',None)
    if checker is None:
        return {'status':'configured','message':'Telemetry adapter is configured; connection check is unavailable.'}
    try:
        import asyncio
        connected=await asyncio.to_thread(checker)
    except Exception:
        telemetry.degraded=True
        connected=False
    if connected:
        return {'status':'connected','message':'Langfuse accepted the configured project credentials.'}
    telemetry.degraded=True
    return {'status':'unreachable','message':'Langfuse did not accept the project credentials or could not be reached. Check the keys and base URL.'}

@router.post('/chat',response_model=CustomerAnswer)
async def chat(body:ChatRequest,request:Request,identity:Identity):
    with request.app.state.telemetry.stage('authorization',request.state.correlation_id,{'principal_id':identity.subject}):
        authorize(identity,body.user_id)
    if request.app.state.services.assistant is None:
        raise DependencyError()
    import asyncio
    from app.errors import AppError,RequestTimeoutError
    try:
        async with asyncio.timeout(request.app.state.settings.request_timeout_seconds):
            return await request.app.state.services.assistant.answer(identity,body)
    except TimeoutError:
        raise RequestTimeoutError() from None
    except AppError:
        raise
    except Exception:
        raise DependencyError() from None

@router.post('/ingest',status_code=202)
async def ingest(request:Request,identity:Identity):
    return await request.app.state.jobs.start(identity)

@router.get('/ingest/{job_id}')
async def job_status(job_id:str,request:Request,identity:Identity):
    return request.app.state.jobs.get(identity,job_id)

@router.get('/customers/{user_id}/summary')
async def summary(user_id:UserID,request:Request,identity:Identity):
    authorize(identity,user_id)
    analytics=request.app.state.services.analytics
    if analytics is None:
        raise DependencyError()
    return await __import__('asyncio').to_thread(analytics.summary,identity,user_id)

@router.get('/customers/{user_id}/revenue')
async def revenue(user_id:UserID,request:Request,identity:Identity):
    authorize(identity,user_id)
    analytics=request.app.state.services.analytics
    if analytics is None:
        raise DependencyError()
    return await __import__('asyncio').to_thread(analytics.revenue,identity,user_id)
