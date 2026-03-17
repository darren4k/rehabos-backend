"""Patient portal API routes -- magic-link auth, appointments, HEP, progress, messaging.

Patients authenticate via magic-link tokens (no password). A link is sent
to the patient's email/phone; clicking it generates a 7-day JWT with
type="patient". This is separate from provider JWT auth.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.config import get_settings
from rehab_os.core.database import get_db
from rehab_os.core.models import (
    AppointmentDB,
    ClinicalNote,
    Insurance,
    Patient,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portal")

# ---------------------------------------------------------------------------
# In-memory stores (TODO: move to DB/Redis)
# ---------------------------------------------------------------------------

# magic_token -> {patient_id, created_at, used}
_magic_tokens: dict[str, dict[str, Any]] = {}

# patient_id -> [message dicts]
_messages: dict[str, list[dict[str, Any]]] = defaultdict(list)

# exercise_id -> {patient_id, exercise data, completed}
_exercises: dict[str, dict[str, Any]] = {}

MAGIC_TOKEN_EXPIRY_MINUTES = 15


# ---------------------------------------------------------------------------
# Patient JWT helpers
# ---------------------------------------------------------------------------

def create_patient_token(patient_id: str) -> str:
    """Create a 7-day JWT for a patient portal session."""
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(days=7)
    payload = {
        "sub": patient_id,
        "type": "patient",
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _decode_patient_token(token: str) -> dict | None:
    """Decode a patient JWT. Returns claims or None."""
    settings = get_settings()
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if claims.get("type") != "patient":
            return None
        return claims
    except JWTError:
        return None


async def get_current_patient(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Patient:
    """Resolve the authenticated patient from Bearer token or cookie."""
    # Check Authorization header
    token = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
    # Fallback to cookie
    if not token:
        token = request.cookies.get("rehab_patient")
    if not token:
        raise HTTPException(status_code=401, detail="Patient not authenticated")

    claims = _decode_patient_token(token)
    if not claims:
        raise HTTPException(status_code=401, detail="Invalid or expired patient token")

    patient_id = claims.get("sub")
    if not patient_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        pid = uuid.UUID(patient_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid patient ID in token")

    result = await db.execute(
        select(Patient).where(Patient.id == pid, Patient.active.is_(True))
    )
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AccessRequest(BaseModel):
    """Request a magic link -- provide email or phone."""
    email: str | None = None
    phone: str | None = None


class VerifyRequest(BaseModel):
    """Verify a magic-link token."""
    token: str


class VerifyResponse(BaseModel):
    access_token: str
    patient_id: str
    first_name: str
    last_name: str
    expires_in_days: int = 7


class PatientProfile(BaseModel):
    id: str
    first_name: str
    last_name: str
    dob: str
    sex: str
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    insurance: list[dict[str, Any]] = []


class AppointmentAction(BaseModel):
    reason: str | None = None


class MessageRequest(BaseModel):
    content: str
    subject: str | None = None


class ExerciseCompleteRequest(BaseModel):
    sets_completed: int | None = None
    reps_completed: int | None = None
    pain_during: int | None = Field(None, ge=0, le=10)
    notes: str | None = None


# ---------------------------------------------------------------------------
# Public endpoints (no patient auth required)
# ---------------------------------------------------------------------------

@router.post("/request-access")
async def request_access(
    body: AccessRequest,
    db: AsyncSession = Depends(get_db),
):
    """Send a magic link to the patient's email or phone.

    In production this would send an actual email/SMS. For now it returns
    the token directly for development/testing purposes.
    """
    if not body.email and not body.phone:
        raise HTTPException(status_code=400, detail="Email or phone required")

    # Look up patient by email or phone
    if body.email:
        result = await db.execute(
            select(Patient).where(Patient.active.is_(True))
        )
        # Since email is encrypted, we need to check all patients
        patients = result.scalars().all()
        patient = next((p for p in patients if p.email == body.email), None)
    else:
        result = await db.execute(
            select(Patient).where(Patient.active.is_(True))
        )
        patients = result.scalars().all()
        patient = next((p for p in patients if p.phone == body.phone), None)

    if not patient:
        # Don't reveal whether patient exists (security)
        return {"message": "If an account exists, a magic link has been sent."}

    # Generate magic token
    magic_token = secrets.token_urlsafe(32)
    _magic_tokens[magic_token] = {
        "patient_id": str(patient.id),
        "created_at": datetime.now(timezone.utc),
        "used": False,
    }

    # TODO: Send actual email/SMS in production
    logger.info("Magic link generated for patient %s", patient.id)

    return {
        "message": "If an account exists, a magic link has been sent.",
        # DEV ONLY -- remove in production
        "_dev_token": magic_token,
    }


@router.post("/verify", response_model=VerifyResponse)
async def verify_magic_link(
    body: VerifyRequest,
    db: AsyncSession = Depends(get_db),
):
    """Verify a magic-link token and return a patient session JWT."""
    token_data = _magic_tokens.get(body.token)
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid or expired magic link")

    # Check expiry
    created = token_data["created_at"]
    if datetime.now(timezone.utc) - created > timedelta(minutes=MAGIC_TOKEN_EXPIRY_MINUTES):
        _magic_tokens.pop(body.token, None)
        raise HTTPException(status_code=401, detail="Magic link expired")

    # Check if already used
    if token_data["used"]:
        raise HTTPException(status_code=401, detail="Magic link already used")

    # Mark as used
    token_data["used"] = True

    patient_id = token_data["patient_id"]

    # Load patient
    try:
        pid = uuid.UUID(patient_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid patient reference")

    result = await db.execute(
        select(Patient).where(Patient.id == pid, Patient.active.is_(True))
    )
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Generate session token
    access_token = create_patient_token(patient_id)

    return VerifyResponse(
        access_token=access_token,
        patient_id=patient_id,
        first_name=patient.first_name,
        last_name=patient.last_name,
    )


# ---------------------------------------------------------------------------
# Authenticated patient endpoints
# ---------------------------------------------------------------------------

@router.get("/me", response_model=PatientProfile)
async def get_profile(
    patient: Patient = Depends(get_current_patient),
    db: AsyncSession = Depends(get_db),
):
    """Get the authenticated patient's profile and insurance info."""
    # Load insurance
    ins_result = await db.execute(
        select(Insurance).where(Insurance.patient_id == patient.id)
    )
    insurance_records = ins_result.scalars().all()

    return PatientProfile(
        id=str(patient.id),
        first_name=patient.first_name,
        last_name=patient.last_name,
        dob=patient.dob.isoformat() if patient.dob else "",
        sex=patient.sex or "",
        phone=patient.phone,
        email=patient.email,
        address=patient.address,
        insurance=[
            {
                "payer_name": ins.payer_name,
                "member_id": ins.member_id,
                "group_id": ins.group_id,
                "authorized_visits": ins.authorized_visits,
                "visits_used": ins.visits_used,
                "is_primary": ins.is_primary,
            }
            for ins in insurance_records
        ],
    )


@router.get("/appointments")
async def get_appointments(
    patient: Patient = Depends(get_current_patient),
    db: AsyncSession = Depends(get_db),
):
    """Get upcoming appointments for the authenticated patient."""
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(AppointmentDB)
        .where(
            AppointmentDB.patient_id == patient.id,
            AppointmentDB.start_time >= now,
            AppointmentDB.status.in_(["scheduled", "confirmed"]),
        )
        .order_by(AppointmentDB.start_time)
    )
    appointments = result.scalars().all()
    return [
        {
            "id": str(a.id),
            "start_time": a.start_time.isoformat(),
            "end_time": a.end_time.isoformat(),
            "discipline": a.discipline,
            "encounter_type": a.encounter_type,
            "status": a.status,
            "location": a.location,
            "notes": a.notes,
        }
        for a in appointments
    ]


@router.post("/appointments/{appointment_id}/confirm")
async def confirm_appointment(
    appointment_id: str,
    patient: Patient = Depends(get_current_patient),
    db: AsyncSession = Depends(get_db),
):
    """Confirm an upcoming appointment."""
    try:
        aid = uuid.UUID(appointment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid appointment ID")

    result = await db.execute(
        select(AppointmentDB).where(
            AppointmentDB.id == aid,
            AppointmentDB.patient_id == patient.id,
        )
    )
    appt = result.scalar_one_or_none()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    appt.status = "confirmed"
    await db.commit()
    return {"status": "confirmed", "appointment_id": appointment_id}


@router.post("/appointments/{appointment_id}/cancel")
async def cancel_appointment(
    appointment_id: str,
    body: AppointmentAction,
    patient: Patient = Depends(get_current_patient),
    db: AsyncSession = Depends(get_db),
):
    """Cancel an appointment with an optional reason."""
    try:
        aid = uuid.UUID(appointment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid appointment ID")

    result = await db.execute(
        select(AppointmentDB).where(
            AppointmentDB.id == aid,
            AppointmentDB.patient_id == patient.id,
        )
    )
    appt = result.scalar_one_or_none()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    appt.status = "cancelled"
    appt.cancel_reason = body.reason
    await db.commit()
    return {"status": "cancelled", "appointment_id": appointment_id, "reason": body.reason}


@router.get("/exercises")
async def get_exercises(
    patient: Patient = Depends(get_current_patient),
):
    """Get the patient's current home exercise program (HEP)."""
    patient_id = str(patient.id)
    exercises = [
        {**ex, "id": eid}
        for eid, ex in _exercises.items()
        if ex.get("patient_id") == patient_id
    ]
    return exercises


@router.post("/exercises/{exercise_id}/complete")
async def complete_exercise(
    exercise_id: str,
    body: ExerciseCompleteRequest,
    patient: Patient = Depends(get_current_patient),
):
    """Mark an exercise as completed with optional details."""
    ex = _exercises.get(exercise_id)
    if not ex or ex.get("patient_id") != str(patient.id):
        raise HTTPException(status_code=404, detail="Exercise not found")

    ex["completed"] = True
    ex["completed_at"] = datetime.now(timezone.utc).isoformat()
    if body.sets_completed is not None:
        ex["sets_completed"] = body.sets_completed
    if body.reps_completed is not None:
        ex["reps_completed"] = body.reps_completed
    if body.pain_during is not None:
        ex["pain_during"] = body.pain_during
    if body.notes:
        ex["completion_notes"] = body.notes

    return {"status": "completed", "exercise_id": exercise_id}


@router.get("/progress")
async def get_progress(
    patient: Patient = Depends(get_current_patient),
    db: AsyncSession = Depends(get_db),
):
    """Get outcome measure progress data for charts."""
    from rehab_os.clinical.outcomes import OutcomeTracker

    tracker = OutcomeTracker()
    patient_id = str(patient.id)

    # Get summary from outcome tracker
    summary = tracker.get_functional_summary(patient_id)

    # Also get flow sheet summary
    from rehab_os.clinical.flow_sheets import get_flow_sheet_service

    fs_svc = get_flow_sheet_service()
    flow_summary = fs_svc.get_summary(patient_id, "PT")

    return {
        "outcome_measures": summary,
        "flow_sheet_summary": flow_summary,
    }


@router.get("/messages")
async def get_messages(
    patient: Patient = Depends(get_current_patient),
):
    """Get messages from the care team."""
    patient_id = str(patient.id)
    messages = _messages.get(patient_id, [])
    return messages


@router.post("/messages")
async def send_message(
    body: MessageRequest,
    patient: Patient = Depends(get_current_patient),
):
    """Send a message to the care team."""
    patient_id = str(patient.id)
    message = {
        "id": str(uuid.uuid4()),
        "from": "patient",
        "patient_id": patient_id,
        "content": body.content,
        "subject": body.subject,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "read": False,
    }
    _messages[patient_id].append(message)
    return {"status": "sent", "message_id": message["id"]}


@router.get("/documents")
async def get_documents(
    patient: Patient = Depends(get_current_patient),
    db: AsyncSession = Depends(get_db),
):
    """Get shared documents (clinical notes, care plan) visible to the patient."""
    result = await db.execute(
        select(ClinicalNote)
        .where(
            ClinicalNote.patient_id == patient.id,
            ClinicalNote.status == "final",
        )
        .order_by(ClinicalNote.note_date.desc())
        .limit(20)
    )
    notes = result.scalars().all()
    return [
        {
            "id": str(n.id),
            "type": n.note_type,
            "date": n.note_date.isoformat() if n.note_date else None,
            "discipline": n.discipline,
            "therapist_name": n.therapist_name,
            # Only expose assessment and plan to patient (not full SOAP)
            "summary": n.soap_assessment[:200] if n.soap_assessment else None,
        }
        for n in notes
    ]
