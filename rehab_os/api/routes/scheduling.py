"""Scheduling API endpoints — DB-backed with conflict detection and insurance checks."""

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.api.dependencies import get_current_user
from rehab_os.core.database import get_db
from rehab_os.core.models import AppointmentDB, Encounter, Insurance, Provider
from rehab_os.core.repository import (
    AppointmentRepository,
    EncounterRepository,
    InsuranceRepository,
    ProviderAvailabilityRepository,
)
from rehab_os.scheduling.models import (
    Appointment,
    AppointmentStatus,
    ScheduleRequest,
    ScheduleResult,
    TimeSlot,
)
from rehab_os.scheduling.scheduler import SchedulingService
from rehab_os.scheduling.optimizer import RouteOptimizer

router = APIRouter(prefix="/scheduling")

_scheduler = SchedulingService()
_optimizer = RouteOptimizer()


# ---------------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------------

class AppointmentCreate(BaseModel):
    patient_id: str
    provider_id: str
    start_time: datetime
    end_time: datetime
    discipline: str = "PT"
    encounter_type: str = "treatment"
    location: str | None = None
    notes: str | None = None


class AppointmentUpdate(BaseModel):
    start_time: datetime | None = None
    end_time: datetime | None = None
    location: str | None = None
    notes: str | None = None
    status: str | None = None
    encounter_type: str | None = None


class AppointmentResponse(BaseModel):
    id: str
    patient_id: str
    provider_id: str
    encounter_id: str | None = None
    start_time: datetime
    end_time: datetime
    location: str | None = None
    discipline: str
    encounter_type: str
    status: str
    cancel_reason: str | None = None
    notes: str | None = None
    is_auto_scheduled: bool = False
    insurance_warning: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CancelRequest(BaseModel):
    reason: str | None = None


class AutoScheduleResponse(BaseModel):
    appointments: list[AppointmentResponse] = []
    conflicts: list[str] = []
    optimization_notes: list[str] = []
    next_available: datetime | None = None


class RouteRequest(BaseModel):
    appointment_ids: list[str]
    provider_home_address: str | None = None


class InsuranceCheckResponse(BaseModel):
    authorized: bool = True
    authorized_visits: int | None = None
    visits_used: int = 0
    remaining: int | None = None
    warning: str | None = None
    expired: bool = False


class AvailabilityRuleIn(BaseModel):
    day_of_week: int = Field(ge=0, le=6)
    start_hour: int = Field(ge=0, le=23, default=8)
    end_hour: int = Field(ge=1, le=24, default=17)
    slot_duration_minutes: int = Field(ge=15, default=45)
    is_available: bool = True


class AvailabilityRuleResponse(BaseModel):
    id: str
    provider_id: str
    day_of_week: int
    start_hour: int
    end_hour: int
    slot_duration_minutes: int
    is_available: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _appt_to_response(appt: AppointmentDB, insurance_warning: str | None = None) -> AppointmentResponse:
    return AppointmentResponse(
        id=str(appt.id),
        patient_id=str(appt.patient_id),
        provider_id=str(appt.provider_id),
        encounter_id=str(appt.encounter_id) if appt.encounter_id else None,
        start_time=appt.start_time,
        end_time=appt.end_time,
        location=appt.location,
        discipline=appt.discipline,
        encounter_type=appt.encounter_type,
        status=appt.status,
        cancel_reason=appt.cancel_reason,
        notes=appt.notes,
        is_auto_scheduled=appt.is_auto_scheduled,
        insurance_warning=insurance_warning,
        created_at=appt.created_at,
        updated_at=appt.updated_at,
    )


def _appt_to_pydantic(appt: AppointmentDB) -> Appointment:
    """Convert DB model to scheduling Pydantic model (for optimizer compatibility)."""
    return Appointment(
        id=str(appt.id),
        patient_id=str(appt.patient_id),
        provider_id=str(appt.provider_id),
        time_slot=TimeSlot(
            start_time=appt.start_time,
            end_time=appt.end_time,
            provider_id=str(appt.provider_id),
            location=appt.location,
        ),
        discipline=appt.discipline,
        encounter_type=appt.encounter_type,
        status=AppointmentStatus(appt.status) if appt.status in AppointmentStatus.__members__.values() else AppointmentStatus.SCHEDULED,
        notes=appt.notes,
    )


async def _check_insurance(
    session: AsyncSession, patient_id: uuid.UUID, discipline: str
) -> str | None:
    """Check insurance authorization, return warning string or None."""
    ins_repo = InsuranceRepository(session)
    records = await ins_repo.get_by_patient(patient_id)
    primary = next((r for r in records if r.is_primary), None)
    if not primary:
        return None
    if primary.authorized_visits is None:
        return None
    remaining = primary.authorized_visits - primary.visits_used
    warnings: list[str] = []
    if primary.expiry_date and primary.expiry_date < date.today():
        warnings.append("Insurance authorization has expired")
    if remaining <= 0:
        warnings.append(f"Authorized visits exhausted ({primary.visits_used}/{primary.authorized_visits})")
    elif remaining <= 3:
        warnings.append(f"Only {remaining} authorized visits remaining")
    return "; ".join(warnings) if warnings else None


def _parse_uuid(value: str, name: str = "ID") -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {name}: {value}")


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------

@router.get("/providers/{provider_id}/availability", response_model=list[TimeSlot])
async def get_provider_availability(
    provider_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[TimeSlot]:
    """Return available slots for a provider over a date range."""
    pid = _parse_uuid(provider_id, "provider_id")
    appt_repo = AppointmentRepository(db)
    avail_repo = ProviderAvailabilityRepository(db)

    # Get booked appointments as Pydantic models
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc)
    booked_db = await appt_repo.list_by_provider_date_range(pid, start_dt, end_dt)
    existing = [_appt_to_pydantic(a) for a in booked_db]

    # Get availability rules
    rules_db = await avail_repo.get_by_provider(pid)
    rules = [
        {
            "day_of_week": r.day_of_week,
            "start_hour": r.start_hour,
            "end_hour": r.end_hour,
            "slot_duration_minutes": r.slot_duration_minutes,
            "is_available": r.is_available,
        }
        for r in rules_db
    ] if rules_db else None

    return _scheduler.find_available_slots(
        provider_id, (start_date, end_date), existing,
        availability_rules=rules,
    )


# ---------------------------------------------------------------------------
# Provider hours (availability rules CRUD)
# ---------------------------------------------------------------------------

@router.get("/providers/{provider_id}/hours", response_model=list[AvailabilityRuleResponse])
async def get_provider_hours(
    provider_id: str,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[AvailabilityRuleResponse]:
    pid = _parse_uuid(provider_id, "provider_id")
    repo = ProviderAvailabilityRepository(db)
    rules = await repo.get_by_provider(pid)
    return [
        AvailabilityRuleResponse(
            id=str(r.id), provider_id=str(r.provider_id),
            day_of_week=r.day_of_week, start_hour=r.start_hour,
            end_hour=r.end_hour, slot_duration_minutes=r.slot_duration_minutes,
            is_available=r.is_available,
        )
        for r in rules
    ]


@router.put("/providers/{provider_id}/hours", response_model=list[AvailabilityRuleResponse])
async def set_provider_hours(
    provider_id: str,
    rules: list[AvailabilityRuleIn],
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[AvailabilityRuleResponse]:
    pid = _parse_uuid(provider_id, "provider_id")
    repo = ProviderAvailabilityRepository(db)
    results = []
    for rule in rules:
        r = await repo.upsert(
            pid, rule.day_of_week,
            start_hour=rule.start_hour, end_hour=rule.end_hour,
            slot_duration_minutes=rule.slot_duration_minutes,
            is_available=rule.is_available,
        )
        results.append(AvailabilityRuleResponse(
            id=str(r.id), provider_id=str(r.provider_id),
            day_of_week=r.day_of_week, start_hour=r.start_hour,
            end_hour=r.end_hour, slot_duration_minutes=r.slot_duration_minutes,
            is_available=r.is_available,
        ))
    return results


# ---------------------------------------------------------------------------
# Book / list / get / update / cancel appointments
# ---------------------------------------------------------------------------

@router.post("/appointments", response_model=AppointmentResponse, status_code=201)
async def book_appointment(
    body: AppointmentCreate,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AppointmentResponse:
    """Book a single appointment with conflict detection and insurance warning."""
    patient_id = _parse_uuid(body.patient_id, "patient_id")
    provider_id = _parse_uuid(body.provider_id, "provider_id")
    repo = AppointmentRepository(db)

    # Conflict detection
    has_conflict = await repo.check_conflict(provider_id, body.start_time, body.end_time)
    if has_conflict:
        raise HTTPException(status_code=409, detail="Time slot conflicts with an existing appointment")

    appt = await repo.create(
        patient_id=patient_id,
        provider_id=provider_id,
        start_time=body.start_time,
        end_time=body.end_time,
        discipline=body.discipline,
        encounter_type=body.encounter_type,
        location=body.location,
        notes=body.notes,
    )

    warning = await _check_insurance(db, patient_id, body.discipline)
    return _appt_to_response(appt, insurance_warning=warning)


@router.get("/appointments", response_model=list[AppointmentResponse])
async def list_appointments(
    patient_id: Optional[str] = Query(None),
    provider_id: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[AppointmentResponse]:
    """List appointments with filters."""
    repo = AppointmentRepository(db)

    if provider_id and date_from and date_to:
        pid = _parse_uuid(provider_id, "provider_id")
        start = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc) if "T" in date_from else datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc) if "T" in date_to else datetime.strptime(date_to, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        appts = await repo.list_by_provider_date_range(pid, start, end)
    elif patient_id:
        pid = _parse_uuid(patient_id, "patient_id")
        appts = await repo.list_by_patient(pid)
    else:
        # Fallback: list recent (today) or by date
        if date_from:
            dt = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            pid_val = _parse_uuid(provider_id, "provider_id") if provider_id else None
            appts = await repo.list_by_date(dt, provider_id=pid_val)
        else:
            dt = datetime.now(timezone.utc)
            appts = await repo.list_by_date(dt)

    results = [_appt_to_response(a) for a in appts]
    if status:
        results = [r for r in results if r.status == status]
    return results


@router.get("/appointments/{appointment_id}", response_model=AppointmentResponse)
async def get_appointment(
    appointment_id: str,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AppointmentResponse:
    """Get a single appointment."""
    aid = _parse_uuid(appointment_id, "appointment_id")
    repo = AppointmentRepository(db)
    appt = await repo.get_by_id(aid)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return _appt_to_response(appt)


@router.put("/appointments/{appointment_id}", response_model=AppointmentResponse)
async def update_appointment(
    appointment_id: str,
    body: AppointmentUpdate,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AppointmentResponse:
    """Update an appointment. Validates conflicts on time changes."""
    aid = _parse_uuid(appointment_id, "appointment_id")
    repo = AppointmentRepository(db)

    appt = await repo.get_by_id(aid)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    # If time is changing, check for conflicts
    new_start = body.start_time or appt.start_time
    new_end = body.end_time or appt.end_time
    if body.start_time or body.end_time:
        has_conflict = await repo.check_conflict(appt.provider_id, new_start, new_end, exclude_id=aid)
        if has_conflict:
            raise HTTPException(status_code=409, detail="New time conflicts with an existing appointment")

    update_data = {k: v for k, v in body.model_dump().items() if v is not None}
    updated = await repo.update(aid, **update_data)
    return _appt_to_response(updated)


@router.delete("/appointments/{appointment_id}", response_model=AppointmentResponse)
async def cancel_appointment(
    appointment_id: str,
    body: CancelRequest | None = None,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AppointmentResponse:
    """Cancel an appointment (soft delete via status)."""
    aid = _parse_uuid(appointment_id, "appointment_id")
    repo = AppointmentRepository(db)
    reason = body.reason if body else None
    appt = await repo.cancel(aid, reason)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return _appt_to_response(appt)


# ---------------------------------------------------------------------------
# Check-in / Complete
# ---------------------------------------------------------------------------

@router.post("/appointments/{appointment_id}/check-in", response_model=AppointmentResponse)
async def check_in_appointment(
    appointment_id: str,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AppointmentResponse:
    """Check in: set status=checked_in and create a linked Encounter."""
    aid = _parse_uuid(appointment_id, "appointment_id")
    appt_repo = AppointmentRepository(db)
    enc_repo = EncounterRepository(db)

    appt = await appt_repo.get_by_id(aid)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if appt.status not in ("scheduled", "confirmed"):
        raise HTTPException(status_code=400, detail=f"Cannot check in from status '{appt.status}'")

    # Create linked encounter
    encounter = await enc_repo.create(
        patient_id=appt.patient_id,
        provider_id=appt.provider_id,
        encounter_date=appt.start_time,
        discipline=appt.discipline,
        encounter_type=appt.encounter_type,
        status="in_progress",
    )

    appt.status = "checked_in"
    appt.encounter_id = encounter.id
    appt.updated_at = datetime.now(timezone.utc)
    await db.flush()

    return _appt_to_response(appt)


@router.post("/appointments/{appointment_id}/complete", response_model=AppointmentResponse)
async def complete_appointment(
    appointment_id: str,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AppointmentResponse:
    """Complete: set status=completed, finalize encounter, increment insurance visits_used."""
    aid = _parse_uuid(appointment_id, "appointment_id")
    appt_repo = AppointmentRepository(db)
    enc_repo = EncounterRepository(db)
    ins_repo = InsuranceRepository(db)

    appt = await appt_repo.get_by_id(aid)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if appt.status != "checked_in":
        raise HTTPException(status_code=400, detail=f"Cannot complete from status '{appt.status}'")

    appt.status = "completed"
    appt.updated_at = datetime.now(timezone.utc)

    # Update encounter status
    if appt.encounter_id:
        await enc_repo.update_status(appt.encounter_id, "completed")

    # Increment insurance visits_used
    records = await ins_repo.get_by_patient(appt.patient_id)
    primary = next((r for r in records if r.is_primary), None)
    if primary:
        await ins_repo.update_visits_used(primary.id, primary.visits_used + 1)

    await db.flush()
    return _appt_to_response(appt)


# ---------------------------------------------------------------------------
# Insurance check
# ---------------------------------------------------------------------------

@router.get("/insurance-check", response_model=InsuranceCheckResponse)
async def insurance_check(
    patient_id: str = Query(...),
    discipline: str = Query("PT"),
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> InsuranceCheckResponse:
    """Check insurance authorization for a patient + discipline."""
    pid = _parse_uuid(patient_id, "patient_id")
    ins_repo = InsuranceRepository(db)
    records = await ins_repo.get_by_patient(pid)
    primary = next((r for r in records if r.is_primary), None)

    if not primary:
        return InsuranceCheckResponse(authorized=True, warning="No insurance on file")

    expired = bool(primary.expiry_date and primary.expiry_date < date.today())
    if primary.authorized_visits is None:
        return InsuranceCheckResponse(
            authorized=True,
            visits_used=primary.visits_used,
            expired=expired,
            warning="Insurance authorization expired" if expired else None,
        )

    remaining = primary.authorized_visits - primary.visits_used
    warnings: list[str] = []
    if expired:
        warnings.append("Insurance authorization has expired")
    if remaining <= 0:
        warnings.append(f"Authorized visits exhausted ({primary.visits_used}/{primary.authorized_visits})")
    elif remaining <= 3:
        warnings.append(f"Only {remaining} authorized visits remaining")

    return InsuranceCheckResponse(
        authorized=remaining > 0 and not expired,
        authorized_visits=primary.authorized_visits,
        visits_used=primary.visits_used,
        remaining=max(remaining, 0),
        expired=expired,
        warning="; ".join(warnings) if warnings else None,
    )


# ---------------------------------------------------------------------------
# Auto-schedule
# ---------------------------------------------------------------------------

@router.post("/auto-schedule", response_model=AutoScheduleResponse)
async def auto_schedule(
    request: ScheduleRequest,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AutoScheduleResponse:
    """Auto-schedule a series of appointments and persist to DB."""
    provider_id_str = request.provider_id or "default-provider"
    appt_repo = AppointmentRepository(db)
    avail_repo = ProviderAvailabilityRepository(db)

    # Get existing appointments as Pydantic models
    start = datetime.now(timezone.utc).date()
    end = start + timedelta(weeks=request.duration_weeks + 2)

    try:
        pid = uuid.UUID(provider_id_str)
        start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc)
        booked_db = await appt_repo.list_by_provider_date_range(pid, start_dt, end_dt)
        existing = [_appt_to_pydantic(a) for a in booked_db]

        rules_db = await avail_repo.get_by_provider(pid)
        rules = [
            {
                "day_of_week": r.day_of_week,
                "start_hour": r.start_hour,
                "end_hour": r.end_hour,
                "slot_duration_minutes": r.slot_duration_minutes,
                "is_available": r.is_available,
            }
            for r in rules_db
        ] if rules_db else None
    except ValueError:
        existing = []
        rules = None

    available = _scheduler.find_available_slots(
        provider_id_str, (start, end), existing,
        availability_rules=rules,
    )
    result = _scheduler.auto_schedule(request, available)

    # Persist to DB and collect flat responses
    patient_id = _parse_uuid(request.patient_id, "patient_id")
    insurance_warning = await _check_insurance(db, patient_id, request.discipline)
    flat_appointments: list[AppointmentResponse] = []

    for appt in result.appointments:
        try:
            prov_id = uuid.UUID(appt.provider_id)
        except ValueError:
            continue
        db_appt = await appt_repo.create(
            patient_id=patient_id,
            provider_id=prov_id,
            start_time=appt.time_slot.start_time.replace(tzinfo=timezone.utc) if appt.time_slot.start_time.tzinfo is None else appt.time_slot.start_time,
            end_time=appt.time_slot.end_time.replace(tzinfo=timezone.utc) if appt.time_slot.end_time.tzinfo is None else appt.time_slot.end_time,
            discipline=appt.discipline,
            encounter_type=appt.encounter_type,
            location=appt.time_slot.location,
            is_auto_scheduled=True,
        )
        flat_appointments.append(_appt_to_response(db_appt, insurance_warning))

    optimization_notes = list(result.optimization_notes)
    if insurance_warning:
        optimization_notes.append(f"Insurance warning: {insurance_warning}")

    return AutoScheduleResponse(
        appointments=flat_appointments,
        conflicts=result.conflicts,
        optimization_notes=optimization_notes,
        next_available=result.next_available,
    )


# ---------------------------------------------------------------------------
# Route optimization
# ---------------------------------------------------------------------------

@router.post("/optimize-route", response_model=list[AppointmentResponse])
async def optimize_route(
    req: RouteRequest,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[AppointmentResponse]:
    """Optimize daily route order for home-health appointments."""
    repo = AppointmentRepository(db)
    appts: list[Appointment] = []
    for aid_str in req.appointment_ids:
        aid = _parse_uuid(aid_str, "appointment_id")
        appt = await repo.get_by_id(aid)
        if not appt:
            raise HTTPException(status_code=404, detail=f"Appointment {aid_str} not found")
        appts.append(_appt_to_pydantic(appt))

    optimized = _optimizer.optimize_daily_route(appts, req.provider_home_address)

    # Return as AppointmentResponse by re-fetching (to get full DB fields)
    result = []
    for pydantic_appt in optimized:
        db_appt = await repo.get_by_id(uuid.UUID(pydantic_appt.id))
        if db_appt:
            result.append(_appt_to_response(db_appt))
    return result
