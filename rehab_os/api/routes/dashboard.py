"""Dashboard metrics API — lightweight aggregate queries for the RehabOS dashboard."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from rehab_os.core.database import get_db
from rehab_os.core.models import AppointmentDB, ClinicalNote, Patient, Provider

router = APIRouter(prefix="/dashboard")


class ComplianceMetrics(BaseModel):
    poc_not_signed: int = 0
    notes_pending_cosign: int = 0
    unsigned_daily_notes: int = 0
    incomplete_docs: int = 0


class DashboardMetrics(BaseModel):
    patients_under_care: int = 0
    visits_this_week: int = 0
    unsigned_notes: int = 0
    no_show_rate: float = 0.0
    compliance: ComplianceMetrics = ComplianceMetrics()


class TherapistSummary(BaseModel):
    id: uuid.UUID
    first_name: str
    last_name: str
    discipline: str
    credentials: Optional[str] = None
    active_patients: int = 0
    visits_this_week: int = 0
    unsigned_notes: int = 0


def _week_range() -> tuple[datetime, datetime]:
    """Return Monday 00:00 and Sunday 23:59 for the current ISO week."""
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    return (
        datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc),
        datetime(sunday.year, sunday.month, sunday.day, 23, 59, 59, tzinfo=timezone.utc),
    )


def _apply_scope(
    stmt: Select,
    model,
    scope: str,
    therapist_id: uuid.UUID | None,
    org_id: uuid.UUID | None,
) -> Select:
    """Apply scope filtering to a SELECT statement.

    - scope='me' + therapist_id → filter by that therapist
    - scope='org' + org_id → filter by org
    - scope='org' without org_id → no filter (show all)
    """
    if scope == "me" and therapist_id is not None:
        if hasattr(model, "therapist_id"):
            stmt = stmt.where(model.therapist_id == therapist_id)
        elif hasattr(model, "provider_id"):
            stmt = stmt.where(model.provider_id == therapist_id)
        elif hasattr(model, "primary_therapist_id"):
            stmt = stmt.where(model.primary_therapist_id == therapist_id)
    elif scope == "org" and org_id is not None:
        if hasattr(model, "organization_id"):
            stmt = stmt.where(model.organization_id == org_id)
    return stmt


@router.get("/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    scope: str = Query("org", regex="^(me|org)$"),
    therapist_id: Optional[uuid.UUID] = Query(None),
    organization_id: Optional[uuid.UUID] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> DashboardMetrics:
    """Single-call aggregate metrics for the dashboard KPI strip and compliance card.

    Query params:
      - scope: 'me' (single therapist) or 'org' (whole organization, default)
      - therapist_id: required when scope='me'
      - organization_id: optional org filter when scope='org'
    """

    # 1. Active patients
    q = select(func.count(Patient.id)).where(Patient.active.is_(True))
    q = _apply_scope(q, Patient, scope, therapist_id, organization_id)
    active_count = (await db.execute(q)).scalar_one()

    # 2. Visits this week (any status except cancelled)
    week_start, week_end = _week_range()
    q = select(func.count(AppointmentDB.id)).where(
        AppointmentDB.start_time >= week_start,
        AppointmentDB.start_time <= week_end,
        AppointmentDB.status != "cancelled",
    )
    q = _apply_scope(q, AppointmentDB, scope, therapist_id, organization_id)
    visits_count = (await db.execute(q)).scalar_one()

    # 3. No-show rate (last 30 days)
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    q_total = select(func.count(AppointmentDB.id)).where(
        AppointmentDB.start_time >= thirty_days_ago,
        AppointmentDB.status.in_(["completed", "no_show"]),
    )
    q_total = _apply_scope(q_total, AppointmentDB, scope, therapist_id, organization_id)
    total_appts_30d = (await db.execute(q_total)).scalar_one()

    q_ns = select(func.count(AppointmentDB.id)).where(
        AppointmentDB.start_time >= thirty_days_ago,
        AppointmentDB.status == "no_show",
    )
    q_ns = _apply_scope(q_ns, AppointmentDB, scope, therapist_id, organization_id)
    no_shows_30d = (await db.execute(q_ns)).scalar_one()
    no_show_rate = round((no_shows_30d / total_appts_30d) * 100, 1) if total_appts_30d > 0 else 0.0

    # 4. Unsigned notes (status = 'draft')
    q = select(func.count(ClinicalNote.id)).where(ClinicalNote.status == "draft")
    q = _apply_scope(q, ClinicalNote, scope, therapist_id, organization_id)
    unsigned_count = (await db.execute(q)).scalar_one()

    # 5. Compliance breakdown by note type
    def _note_count(note_type: str) -> Select:
        s = select(func.count(ClinicalNote.id)).where(
            ClinicalNote.status == "draft",
            ClinicalNote.note_type == note_type,
        )
        return _apply_scope(s, ClinicalNote, scope, therapist_id, organization_id)

    poc_unsigned = (await db.execute(_note_count("evaluation"))).scalar_one()
    cosign_pending = (await db.execute(_note_count("progress_note"))).scalar_one()
    daily_unsigned = (await db.execute(_note_count("daily_note"))).scalar_one()

    # Incomplete docs: draft notes missing any SOAP section
    q_inc = select(func.count(ClinicalNote.id)).where(
        ClinicalNote.status == "draft",
        (
            ClinicalNote.soap_subjective.is_(None)
            | ClinicalNote.soap_objective.is_(None)
            | ClinicalNote.soap_assessment.is_(None)
            | ClinicalNote.soap_plan.is_(None)
        ),
    )
    q_inc = _apply_scope(q_inc, ClinicalNote, scope, therapist_id, organization_id)
    incomplete = (await db.execute(q_inc)).scalar_one()

    return DashboardMetrics(
        patients_under_care=active_count,
        visits_this_week=visits_count,
        unsigned_notes=unsigned_count,
        no_show_rate=no_show_rate,
        compliance=ComplianceMetrics(
            poc_not_signed=poc_unsigned,
            notes_pending_cosign=cosign_pending,
            unsigned_daily_notes=daily_unsigned,
            incomplete_docs=incomplete,
        ),
    )
