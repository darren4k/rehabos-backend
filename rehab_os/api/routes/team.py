"""Team management API — therapist roster and per-therapist metrics."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.core.database import get_db
from rehab_os.core.models import AppointmentDB, ClinicalNote, Patient, Provider
from rehab_os.core.schemas import ProviderRead

router = APIRouter(prefix="/team")


class TherapistSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    first_name: str
    last_name: str
    discipline: str
    credentials: Optional[str] = None
    role: str = "therapist"
    active_patients: int = 0
    visits_this_week: int = 0
    unsigned_notes: int = 0


class TeamResponse(BaseModel):
    therapists: list[TherapistSummary]
    total: int


@router.get("", response_model=TeamResponse)
async def get_team(
    organization_id: Optional[uuid.UUID] = Query(None),
    discipline: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> TeamResponse:
    """List therapists in the organization with per-therapist KPI counts."""

    # Fetch providers
    q = select(Provider).where(Provider.active.is_(True))
    if organization_id:
        q = q.where(Provider.organization_id == organization_id)
    if discipline:
        q = q.where(Provider.discipline == discipline)
    q = q.order_by(Provider.last_name, Provider.first_name)

    result = await db.execute(q)
    providers = result.scalars().all()

    # Compute per-therapist metrics
    now = datetime.now(timezone.utc)
    from rehab_os.api.routes.dashboard import _week_range
    week_start, week_end = _week_range()

    summaries: list[TherapistSummary] = []
    for p in providers:
        # Active patients where this provider is primary therapist
        active_pts = (
            await db.execute(
                select(func.count(Patient.id)).where(
                    Patient.active.is_(True),
                    Patient.primary_therapist_id == p.id,
                )
            )
        ).scalar_one()

        # Visits this week
        visits = (
            await db.execute(
                select(func.count(AppointmentDB.id)).where(
                    AppointmentDB.provider_id == p.id,
                    AppointmentDB.start_time >= week_start,
                    AppointmentDB.start_time <= week_end,
                    AppointmentDB.status != "cancelled",
                )
            )
        ).scalar_one()

        # Unsigned notes
        unsigned = (
            await db.execute(
                select(func.count(ClinicalNote.id)).where(
                    ClinicalNote.therapist_id == p.id,
                    ClinicalNote.status == "draft",
                )
            )
        ).scalar_one()

        summaries.append(
            TherapistSummary(
                id=p.id,
                first_name=p.first_name,
                last_name=p.last_name,
                discipline=p.discipline,
                credentials=p.credentials,
                role=p.role,
                active_patients=active_pts,
                visits_this_week=visits,
                unsigned_notes=unsigned,
            )
        )

    return TeamResponse(therapists=summaries, total=len(summaries))


@router.get("/{provider_id}", response_model=ProviderRead)
async def get_therapist(
    provider_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> ProviderRead:
    """Get a single therapist's profile."""
    result = await db.execute(select(Provider).where(Provider.id == provider_id))
    provider = result.scalar_one()
    return ProviderRead.model_validate(provider)
