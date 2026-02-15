"""Scheduling API endpoints.

Appointments are stored **in-memory** (dict keyed by id).
Production would use the core DB via SQLAlchemy / async session.
"""

import uuid
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

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

# ---------------------------------------------------------------------------
# In-memory store  (replace with DB in production)
# ---------------------------------------------------------------------------
_appointments: dict[str, Appointment] = {}

_scheduler = SchedulingService()
_optimizer = RouteOptimizer()


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------

@router.get("/providers/{provider_id}/availability", response_model=list[TimeSlot])
async def get_provider_availability(
    provider_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
) -> list[TimeSlot]:
    """Return available slots for a provider over a date range."""
    existing = list(_appointments.values())
    return _scheduler.find_available_slots(provider_id, (start_date, end_date), existing)


# ---------------------------------------------------------------------------
# Book / list / update appointments
# ---------------------------------------------------------------------------

@router.post("/appointments", response_model=Appointment, status_code=201)
async def book_appointment(appointment: Appointment) -> Appointment:
    """Book a single appointment."""
    if not appointment.id:
        appointment.id = str(uuid.uuid4())
    _appointments[appointment.id] = appointment
    return appointment


@router.get("/appointments", response_model=list[Appointment])
async def list_appointments(
    patient_id: Optional[str] = Query(None),
    provider_id: Optional[str] = Query(None),
    date: Optional[str] = Query(None, alias="date"),
) -> list[Appointment]:
    """List appointments with optional filters."""
    results = list(_appointments.values())
    if patient_id:
        results = [a for a in results if a.patient_id == patient_id]
    if provider_id:
        results = [a for a in results if a.provider_id == provider_id]
    if date:
        d = datetime.strptime(date, "%Y-%m-%d").date()
        results = [a for a in results if a.time_slot.start_time.date() == d]
    return results


@router.put("/appointments/{appointment_id}", response_model=Appointment)
async def update_appointment(appointment_id: str, update: dict) -> Appointment:
    """Update an appointment (reschedule, cancel, etc.)."""
    if appointment_id not in _appointments:
        raise HTTPException(status_code=404, detail="Appointment not found")
    appt = _appointments[appointment_id]
    data = appt.model_dump()
    data.update(update)
    data["updated_at"] = datetime.utcnow()
    updated = Appointment(**data)
    _appointments[appointment_id] = updated
    return updated


# ---------------------------------------------------------------------------
# Auto-schedule
# ---------------------------------------------------------------------------

@router.post("/auto-schedule", response_model=ScheduleResult)
async def auto_schedule(request: ScheduleRequest) -> ScheduleResult:
    """Auto-schedule a series of appointments."""
    provider_id = request.provider_id or "default-provider"
    existing = list(_appointments.values())

    start = datetime.utcnow().date()
    end = start + __import__("datetime").timedelta(weeks=request.duration_weeks + 2)
    available = _scheduler.find_available_slots(provider_id, (start, end), existing)

    result = _scheduler.auto_schedule(request, available)

    # Persist scheduled appointments
    for appt in result.appointments:
        _appointments[appt.id] = appt

    return result


# ---------------------------------------------------------------------------
# Route optimization
# ---------------------------------------------------------------------------

class _RouteRequest(__import__("pydantic").BaseModel):
    appointment_ids: list[str]
    provider_home_address: Optional[str] = None


@router.post("/optimize-route", response_model=list[Appointment])
async def optimize_route(req: _RouteRequest) -> list[Appointment]:
    """Optimize daily route order for home-health appointments."""
    appts = []
    for aid in req.appointment_ids:
        if aid not in _appointments:
            raise HTTPException(status_code=404, detail=f"Appointment {aid} not found")
        appts.append(_appointments[aid])
    return _optimizer.optimize_daily_route(appts, req.provider_home_address)
