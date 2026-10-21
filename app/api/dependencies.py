"""Authentication, scope-independent request limits, and server-side services."""
from collections import deque
import time
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.errors import AuthenticationError, RateLimitError
from app.security import authenticate

bearer=HTTPBearer(auto_error=False)

class RateLimiter:
    """One-process sliding window. Entries never contain bearer credentials."""
    def __init__(self,limit=60,clock=time.monotonic):
        self.limit=limit
        self.clock=clock
        self.windows={}
    def check(self,subject):
        now=self.clock()
        # Prune inactive subjects so the dictionary remains bounded by active identities.
        for key in list(self.windows):
            if not self.windows[key] or self.windows[key][-1] <= now-60:
                del self.windows[key]
        window=self.windows.setdefault(subject,deque())
        while window and window[0] <= now-60:
            window.popleft()
        if len(window)>=self.limit:
            raise RateLimitError()
        window.append(now)

async def principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
):
    with request.app.state.telemetry.stage('authentication',request.state.correlation_id,{}):
        if credentials is None or credentials.scheme.lower()!='bearer':
            raise AuthenticationError()
        identity=authenticate(credentials.credentials,request.app.state.settings)
        request.app.state.limiter.check(identity.subject)
    return identity
