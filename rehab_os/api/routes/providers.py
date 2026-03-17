"""Provider Search API — NPI lookup via NPPES registry.

Endpoints:
  GET  /providers/search?q=John+Smith&state=NJ&discipline=PT
  GET  /providers/npi/{npi}
  GET  /providers/verify/{npi}
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from rehab_os.api.dependencies import get_current_user
from rehab_os.core.models import Provider
from rehab_os.integrations.nppes_client import (
    EntityType,
    ProviderStatus,
    get_nppes_client,
)

router = APIRouter(prefix="/providers", tags=["providers"])
logger = logging.getLogger(__name__)


class ProviderSearchResult(BaseModel):
    npi: str
    name: str
    first_name: str = ""
    last_name: str = ""
    credential: str = ""
    specialty: str = ""
    discipline: Optional[str] = None
    is_rehab_provider: bool = False
    phone: str = ""
    fax: str = ""
    city: str = ""
    state: str = ""
    address: str = ""
    status: str = "verified"
    enumeration_date: Optional[str] = None


class ProviderSearchResponse(BaseModel):
    results: list[ProviderSearchResult] = Field(default_factory=list)
    count: int = 0
    query: str = ""


class ProviderVerifyResponse(BaseModel):
    npi: str
    is_valid: bool
    provider: Optional[ProviderSearchResult] = None
    message: str = ""


@router.get("/search", response_model=ProviderSearchResponse)
async def search_providers(
    q: str = Query("", description="Provider name (first last)"),
    state: Optional[str] = Query(None, description="2-letter state code"),
    discipline: Optional[str] = Query(None, description="PT, OT, or SLP"),
    specialty: Optional[str] = Query(None, description="Taxonomy description"),
    entity_type: Optional[str] = Query(None, description="individual or organization"),
    limit: int = Query(10, ge=1, le=50),
    current_user: Provider = Depends(get_current_user),
):
    """Search NPPES for providers by name, state, discipline.

    Examples:
      /providers/search?q=John+Smith&state=NJ
      /providers/search?q=Smith&discipline=PT&state=CA
      /providers/search?specialty=Physical+Therapist&state=NY
    """
    client = get_nppes_client()

    if not q and not specialty and not state:
        return ProviderSearchResponse(query=q)

    # If discipline specified, use rehab-specific search
    if discipline and discipline.upper() in ("PT", "OT", "SLP"):
        providers = client.search_rehab_providers(
            name=q or None,
            state=state,
            discipline=discipline.upper(),
            limit=limit,
        )
    else:
        # General search
        first_name = None
        last_name = None

        if q:
            parts = q.strip().split()
            prefixes = {"dr", "dr.", "doctor"}
            parts = [p for p in parts if p.lower() not in prefixes]
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = parts[-1]
            elif len(parts) == 1:
                last_name = parts[0] + "*"

        etype = None
        if entity_type == "individual":
            etype = EntityType.INDIVIDUAL
        elif entity_type == "organization":
            etype = EntityType.ORGANIZATION

        providers = client.search_providers(
            first_name=first_name,
            last_name=last_name,
            state=state,
            specialty=specialty,
            entity_type=etype or EntityType.INDIVIDUAL,
            limit=limit,
        )
        # Filter deactivated
        providers = [p for p in providers if p.status != ProviderStatus.DEACTIVATED]

    results = [ProviderSearchResult(**p.to_dict()) for p in providers]

    return ProviderSearchResponse(
        results=results,
        count=len(results),
        query=q,
    )


@router.get("/npi/{npi}", response_model=ProviderSearchResult)
async def lookup_npi(
    npi: str,
    current_user: Provider = Depends(get_current_user),
):
    """Look up a single provider by NPI."""
    if not npi or len(npi) != 10 or not npi.isdigit():
        raise HTTPException(400, "NPI must be a 10-digit number")

    client = get_nppes_client()
    provider = client.lookup_npi(npi)

    if not provider:
        raise HTTPException(404, f"NPI {npi} not found in NPPES registry")

    return ProviderSearchResult(**provider.to_dict())


@router.get("/verify/{npi}", response_model=ProviderVerifyResponse)
async def verify_npi(
    npi: str,
    current_user: Provider = Depends(get_current_user),
):
    """Verify an NPI is valid and active."""
    if not npi or len(npi) != 10 or not npi.isdigit():
        raise HTTPException(400, "NPI must be a 10-digit number")

    client = get_nppes_client()
    is_valid, provider = client.verify_provider(npi)

    result = ProviderVerifyResponse(npi=npi, is_valid=is_valid)

    if provider:
        result.provider = ProviderSearchResult(**provider.to_dict())
        if provider.status == ProviderStatus.DEACTIVATED:
            result.message = f"NPI {npi} is deactivated (since {provider.deactivation_date})"
        else:
            result.message = f"NPI {npi} is active — {provider.display_name}"
    else:
        result.message = f"NPI {npi} not found in NPPES registry"

    return result
