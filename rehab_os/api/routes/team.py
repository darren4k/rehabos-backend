"""Team management API — therapist roster and per-therapist metrics."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select, literal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from rehab_os.api.dependencies import get_current_user
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
    current_user: Provider = Depends(get_current_user),
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

    if not providers:
        return TeamResponse(therapists=[], total=0)

    # Batch-compute per-therapist metrics with 3 aggregate queries (not 3N)
    from rehab_os.api.routes.dashboard import _week_range
    week_start, week_end = _week_range()

    provider_ids = [p.id for p in providers]

    # Active patients per therapist
    pts_result = await db.execute(
        select(
            Patient.primary_therapist_id,
            func.count(Patient.id).label("cnt"),
        )
        .where(Patient.active.is_(True), Patient.primary_therapist_id.in_(provider_ids))
        .group_by(Patient.primary_therapist_id)
    )
    active_pts_map = {row[0]: row[1] for row in pts_result.all()}

    # Visits this week per therapist
    visits_result = await db.execute(
        select(
            AppointmentDB.provider_id,
            func.count(AppointmentDB.id).label("cnt"),
        )
        .where(
            AppointmentDB.provider_id.in_(provider_ids),
            AppointmentDB.start_time >= week_start,
            AppointmentDB.start_time <= week_end,
            AppointmentDB.status != "cancelled",
        )
        .group_by(AppointmentDB.provider_id)
    )
    visits_map = {row[0]: row[1] for row in visits_result.all()}

    # Unsigned notes per therapist
    unsigned_result = await db.execute(
        select(
            ClinicalNote.therapist_id,
            func.count(ClinicalNote.id).label("cnt"),
        )
        .where(
            ClinicalNote.therapist_id.in_(provider_ids),
            ClinicalNote.status == "draft",
        )
        .group_by(ClinicalNote.therapist_id)
    )
    unsigned_map = {row[0]: row[1] for row in unsigned_result.all()}

    summaries = [
        TherapistSummary(
            id=p.id,
            first_name=p.first_name,
            last_name=p.last_name,
            discipline=p.discipline,
            credentials=p.credentials,
            role=p.role,
            active_patients=active_pts_map.get(p.id, 0),
            visits_this_week=visits_map.get(p.id, 0),
            unsigned_notes=unsigned_map.get(p.id, 0),
        )
        for p in providers
    ]

    return TeamResponse(therapists=summaries, total=len(summaries))


@router.get("/{provider_id}", response_model=ProviderRead)
async def get_therapist(
    provider_id: uuid.UUID,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProviderRead:
    """Get a single therapist's profile."""
    result = await db.execute(select(Provider).where(Provider.id == provider_id))
    provider = result.scalar_one()
    return ProviderRead.model_validate(provider)
