"""
Operator (admin) authentication for state-changing API endpoints.

A single shared secret is read from the LOADLY_ADMIN_KEY environment variable at
request time and must be presented in the X-Admin-Key header. The design fails closed:
when the key is unset, empty, or shorter than MIN_ADMIN_KEY_LENGTH, every admin
endpoint answers 503 rather than running unauthenticated.

The key value is never logged or echoed back.
"""

import hashlib
import hmac
import logging
import os
from typing import Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

ADMIN_KEY_ENV = "LOADLY_ADMIN_KEY"
ADMIN_KEY_HEADER = "X-Admin-Key"
MIN_ADMIN_KEY_LENGTH = 24

ADMIN_DISABLED_DETAIL = f"admin endpoints disabled: {ADMIN_KEY_ENV} not configured"
ADMIN_UNAUTHORIZED_DETAIL = "Invalid or missing admin key."

KEY_STATUS_OK = "ok"
KEY_STATUS_MISSING = "missing"
KEY_STATUS_TOO_SHORT = "too_short"

# auto_error=False: a missing header must yield our 401 (not FastAPI's default 403),
# and the "not configured" check must run before any header check.
_admin_key_header = APIKeyHeader(
    name=ADMIN_KEY_HEADER,
    auto_error=False,
    description="Operator admin key (LOADLY_ADMIN_KEY) required for state-changing endpoints.",
)


def _configured_admin_key() -> str:
    return os.getenv(ADMIN_KEY_ENV, "").strip()


def admin_key_status() -> str:
    """Return the configuration status of the admin key without exposing its value."""
    key = _configured_admin_key()
    if not key:
        return KEY_STATUS_MISSING
    if len(key) < MIN_ADMIN_KEY_LENGTH:
        return KEY_STATUS_TOO_SHORT
    return KEY_STATUS_OK


def _digest(value: str) -> bytes:
    # Hashing first makes the comparison length-independent; compare_digest on the
    # fixed-size digests is constant time.
    return hashlib.sha256(value.encode("utf-8")).digest()


def require_admin_key(
    request: Request,
    provided_key: Optional[str] = Security(_admin_key_header),
) -> None:
    """FastAPI dependency guarding admin endpoints. Raises 503 or 401; returns None on success."""
    key_status = admin_key_status()
    if key_status != KEY_STATUS_OK:
        # Weak keys are treated exactly like an unset key: fail closed.
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=ADMIN_DISABLED_DETAIL)

    if not provided_key or not hmac.compare_digest(_digest(provided_key), _digest(_configured_admin_key())):
        client_host = request.client.host if request.client else "unknown"
        logger.warning("Rejected admin request to %s from %s", request.url.path, client_host)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=ADMIN_UNAUTHORIZED_DETAIL)
