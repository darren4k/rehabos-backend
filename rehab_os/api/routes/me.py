"""Current-user endpoint — returns the logged-in provider profile.

For now (no auth system), accepts provider_id as a query param.
When auth is added, this will read from the JWT/session.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.core.database import get_db
from rehab_os.core.models import Provider
from rehab_os.core.schemas import OrganizationRead, ProviderRead

router = APIRouter()


class MeResponse(ProviderRead):
    """Extended provider read with organization details."""
    organization_name: Optional[str] = None


@router.get("/me", response_model=MeResponse)
async def get_current_user(
    provider_id: Optional[uuid.UUID] = Query(None, description="Provider ID (temp until auth)"),
    db: AsyncSession = Depends(get_db),
) -> MeResponse:
    """Return the current user's provider profile.

    Until auth is wired, pass ?provider_id=<uuid>.
    If omitted, returns the first active provider (dev convenience).
    """
    if provider_id:
        q = select(Provider).where(Provider.id == provider_id)
    else:
        # Dev fallback: first active provider
        q = select(Provider).where(Provider.active.is_(True)).limit(1)

    result = await db.execute(q)
    provider = result.scalar_one_or_none()
    if not provider:
        raise HTTPException(status_code=404, detail="No provider found")

    org_name = None
    if provider.organization:
        org_name = provider.organization.name

    return MeResponse(
        id=provider.id,
        first_name=provider.first_name,
        last_name=provider.last_name,
        credentials=provider.credentials,
        npi=provider.npi,
        discipline=provider.discipline,
        role=provider.role,
        email=provider.email,
        organization_id=provider.organization_id,
        active=provider.active,
        organization_name=org_name,
    )
