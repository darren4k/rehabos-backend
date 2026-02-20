"""Export endpoints — PDF generation for clinical notes."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from rehab_os.api.audit import log_phi_access
from rehab_os.api.dependencies import get_current_user
from rehab_os.core.database import get_db
from rehab_os.core.models import Provider
from rehab_os.core.repository import ClinicalNoteRepository
from rehab_os.export.pdf_generator import generate_note_pdf

router = APIRouter(prefix="/notes", tags=["export"])


# --- Request model for POST (unsaved notes) ---

_VALID_NOTE_TYPES = Literal[
    "evaluation", "daily_note", "progress_note", "recertification", "discharge_summary"
]


class NoteExportRequest(BaseModel):
    note_type: _VALID_NOTE_TYPES = "daily_note"
    note_date: date
    discipline: str = "pt"
    therapist_name: Optional[str] = None
    soap_subjective: Optional[str] = None
    soap_objective: Optional[str] = None
    soap_assessment: Optional[str] = None
    soap_plan: Optional[str] = None
    structured_data: Optional[dict] = None
    compliance_score: Optional[float] = None
    compliance_warnings: Optional[list[str]] = None
    clinic_name: str = "RehabOS Clinic"


def _safe_filename(raw: str) -> str:
    """Sanitize a string for use in Content-Disposition filename."""
    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', raw)


def _pdf_response(pdf_bytes: bytes, filename: str) -> Response:
    safe = _safe_filename(filename)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe}"'},
    )


@router.get("/{note_id}/export/pdf")
async def export_note_pdf(note_id: uuid.UUID, current_user: Provider = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Generate a PDF from a persisted clinical note."""
    repo = ClinicalNoteRepository(db)
    note = await repo.get_by_id(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Normalize compliance_warnings: DB stores JSON (could be dict or list)
    warnings = note.compliance_warnings
    if isinstance(warnings, dict):
        warnings = list(warnings.values()) if warnings else []
    elif not isinstance(warnings, list):
        warnings = []

    try:
        pdf_bytes = generate_note_pdf(
            note_type=note.note_type,
            note_date=str(note.note_date),
            discipline=note.discipline,
            therapist_name=note.therapist_name,
            soap={
                "subjective": note.soap_subjective or "",
                "objective": note.soap_objective or "",
                "assessment": note.soap_assessment or "",
                "plan": note.soap_plan or "",
            },
            structured_data=note.structured_data,
            compliance_score=float(note.compliance_score) if note.compliance_score is not None else None,
            compliance_warnings=warnings,
        )
    except Exception as exc:
        logger.exception("PDF generation failed for note %s", note_id)
        raise HTTPException(status_code=500, detail="PDF generation failed")

    log_phi_access(
        user_id=str(current_user.id),
        action="export",
        resource_type="clinical_note",
        resource_id=str(note_id),
        ip_address="",
    )
    filename = f"note_{note.note_type}_{note.note_date}.pdf"
    return _pdf_response(pdf_bytes, filename)


@router.post("/export/pdf")
async def export_note_data_pdf(payload: NoteExportRequest, current_user: Provider = Depends(get_current_user)):
    """Generate a PDF from an unsaved note payload (e.g. from SkilledNotes or live sessions)."""
    try:
        pdf_bytes = generate_note_pdf(
            note_type=payload.note_type,
            note_date=str(payload.note_date),
            discipline=payload.discipline,
            therapist_name=payload.therapist_name,
            soap={
                "subjective": payload.soap_subjective or "",
                "objective": payload.soap_objective or "",
                "assessment": payload.soap_assessment or "",
                "plan": payload.soap_plan or "",
            },
            structured_data=payload.structured_data,
            compliance_score=payload.compliance_score,
            compliance_warnings=payload.compliance_warnings,
            clinic_name=payload.clinic_name,
        )
    except Exception as exc:
        logger.exception("PDF generation failed for POST export")
        raise HTTPException(status_code=500, detail="PDF generation failed")

    log_phi_access(
        user_id=str(current_user.id),
        action="export",
        resource_type="clinical_note",
        resource_id="",
        ip_address="",
    )
    filename = f"note_{payload.note_type}_{payload.note_date}.pdf"
    return _pdf_response(pdf_bytes, filename)
