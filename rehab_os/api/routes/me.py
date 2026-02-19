"""Current-user endpoint — returns the logged-in provider profile."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends

from rehab_os.api.dependencies import get_current_user
from rehab_os.core.models import Provider
from rehab_os.core.schemas import ProviderRead

router = APIRouter()


class MeResponse(ProviderRead):
    """Extended provider read with organization details."""
    organization_name: Optional[str] = None
    must_change_password: bool = False


@router.get("/me", response_model=MeResponse)
async def get_me(
    current_user: Provider = Depends(get_current_user),
) -> MeResponse:
    """Return the current user's provider profile."""
    org_name = None
    if current_user.organization:
        org_name = current_user.organization.name

    return MeResponse(
        id=current_user.id,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        credentials=current_user.credentials,
        npi=current_user.npi,
        discipline=current_user.discipline,
        role=current_user.role,
        email=current_user.email,
        organization_id=current_user.organization_id,
        active=current_user.active,
        organization_name=org_name,
        must_change_password=current_user.must_change_password,
    )
