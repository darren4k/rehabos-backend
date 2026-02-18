"""Export endpoints — PDF generation for clinical notes."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.core.database import get_db
from rehab_os.core.repository import ClinicalNoteRepository
from rehab_os.export.pdf_generator import generate_note_pdf

router = APIRouter(prefix="/notes", tags=["export"])


# --- Request model for POST (unsaved notes) ---

class NoteExportRequest(BaseModel):
    note_type: str = "daily_note"
    note_date: str
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


def _pdf_response(pdf_bytes: bytes, filename: str) -> Response:
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{note_id}/export/pdf")
async def export_note_pdf(note_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Generate a PDF from a persisted clinical note."""
    repo = ClinicalNoteRepository(db)
    note = await repo.get_by_id(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

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
        compliance_score=note.compliance_score,
        compliance_warnings=note.compliance_warnings,
    )

    filename = f"note_{note.note_type}_{note.note_date}.pdf"
    return _pdf_response(pdf_bytes, filename)


@router.post("/export/pdf")
async def export_note_data_pdf(payload: NoteExportRequest):
    """Generate a PDF from an unsaved note payload (e.g. from SkilledNotes or live sessions)."""
    pdf_bytes = generate_note_pdf(
        note_type=payload.note_type,
        note_date=payload.note_date,
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

    filename = f"note_{payload.note_type}_{payload.note_date}.pdf"
    return _pdf_response(pdf_bytes, filename)
