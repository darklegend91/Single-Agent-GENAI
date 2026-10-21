"""Server-controlled identities and explicit per-customer authorization."""

import hmac
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, StrictBool

from app.errors import AuthenticationError, AuthorizationError
from app.schemas import NonemptyText, UserID

if TYPE_CHECKING:
    from app.config import Settings


class Principal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    subject: NonemptyText
    customer_ids: frozenset[UserID]
    ingest_admin: StrictBool = False
    # Explicitly enabled only by the loopback local runner. The source repository
    # still verifies that each requested ID exists in the active dataset.
    dataset_scope: StrictBool = False


def authenticate(token: str, settings: "Settings") -> Principal:
    """Compare all configured credentials, using constant-time byte comparison."""
    if not isinstance(token, str) or not token or len(token) > 4096:
        raise AuthenticationError()
    try:
        supplied = token.encode("utf-8")
    except UnicodeEncodeError:
        raise AuthenticationError() from None
    principal = None
    for credential, configured_principal in settings.credential_mapping.items():
        if hmac.compare_digest(supplied, credential.encode("utf-8")):
            principal = configured_principal
    if principal is None:
        raise AuthenticationError()
    return principal


def authorize(principal: Principal, user_id: str) -> None:
    """Ingestion administration grants no implicit customer read permissions."""
    if user_id in principal.customer_ids:
        return
    if principal.dataset_scope and re.fullmatch(r"U[0-9]{10}", user_id, re.ASCII):
        return
    else:
        raise AuthorizationError()
