"""FastAPI auth dependencies — cookie JWT + API key dual-auth."""

from __future__ import annotations

import hmac
import uuid

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.config import get_settings
from rehab_os.core.auth import ACCESS_COOKIE, decode_token
from rehab_os.core.database import get_db
from rehab_os.core.models import Provider


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Provider:
    """Resolve the authenticated provider.

    Priority:
    1. rehab_access cookie → decode JWT → load Provider
    2. API key (Bearer / X-API-Key) + X-Provider-Id header → load Provider
    3. Raise 401
    """
    # --- Path 1: JWT cookie ---
    token = request.cookies.get(ACCESS_COOKIE)
    if token:
        claims = decode_token(token)
        if claims and claims.get("type") == "access":
            provider_id = claims.get("sub")
            if provider_id:
                try:
                    pid = uuid.UUID(provider_id)
                except ValueError:
                    raise HTTPException(status_code=401, detail="Invalid token subject")
                result = await db.execute(
                    select(Provider).where(Provider.id == pid, Provider.active.is_(True))
                )
                provider = result.scalar_one_or_none()
                if provider:
                    return provider
        # Cookie present but invalid → fall through to API key check

    # --- Path 2: API key ---
    settings = get_settings()
    if settings.api_key:
        auth_header = request.headers.get("Authorization")
        api_key_header = request.headers.get("X-API-Key")

        provided_key = None
        if auth_header and auth_header.startswith("Bearer "):
            provided_key = auth_header[7:]
        elif api_key_header:
            provided_key = api_key_header

        if provided_key and hmac.compare_digest(provided_key, settings.api_key):
            # Machine client must supply X-Provider-Id
            provider_id_str = request.headers.get("X-Provider-Id")
            if provider_id_str:
                try:
                    pid = uuid.UUID(provider_id_str)
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid X-Provider-Id")
                result = await db.execute(
                    select(Provider).where(Provider.id == pid, Provider.active.is_(True))
                )
                provider = result.scalar_one_or_none()
                if provider:
                    return provider
            else:
                # API key valid but no provider specified — return first active owner/admin
                result = await db.execute(
                    select(Provider).where(Provider.active.is_(True)).limit(1)
                )
                provider = result.scalar_one_or_none()
                if provider:
                    return provider

    raise HTTPException(status_code=401, detail="Not authenticated")


async def require_admin(
    current_user: Provider = Depends(get_current_user),
) -> Provider:
    """Require the current user to be an owner or admin."""
    if current_user.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
