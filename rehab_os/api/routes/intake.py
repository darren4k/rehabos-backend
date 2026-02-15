"""API routes for intake processing."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.intake.pipeline import IntakePipeline

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
}

SOURCE_TYPES = [
    {"id": "referral", "label": "Physician Referral"},
    {"id": "prescription", "label": "Prescription / Script"},
    {"id": "transfer_summary", "label": "Transfer Summary"},
    {"id": "prior_auth", "label": "Prior Authorization"},
]


class TextIntakeRequest(BaseModel):
    """Request body for raw text intake."""

    text: str
    source_type: str = "referral"


async def _get_db_optional(request: Request) -> AsyncSession | None:
    """Get a database session if available, otherwise None."""
    try:
        from rehab_os.core.database import get_db

        async for session in get_db():
            return session
    except Exception:
        return None


def _get_pipeline(request: Request, db: AsyncSession | None = None) -> IntakePipeline:
    """Build an IntakePipeline from app state."""
    llm_router = request.app.state.llm_router
    session_memory = getattr(request.app.state, "session_memory", None)
    return IntakePipeline(llm=llm_router, session_memory=session_memory, db_session=db)


@router.post("/intake/upload")
async def upload_referral(
    request: Request,
    file: UploadFile = File(...),
    source_type: str = Form("referral"),
    referring_provider: Optional[str] = Form(None),
    db: AsyncSession | None = Depends(_get_db_optional),
):
    """Upload a referral PDF or image and extract a structured patient profile."""
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Allowed: {ALLOWED_CONTENT_TYPES}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    pipeline = _get_pipeline(request, db)
    metadata = {
        "source_type": source_type,
        "referring_provider": referring_provider,
    }

    try:
        result = await pipeline.process_referral(file_bytes, content_type, metadata)
        return result.model_dump()
    except Exception as e:
        logger.exception("Intake upload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Intake processing failed: {e}")


@router.post("/intake/text")
async def intake_from_text(request: Request, body: TextIntakeRequest, db: AsyncSession | None = Depends(_get_db_optional)):
    """Process raw text (copy-paste or typed) through the intake pipeline."""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    pipeline = _get_pipeline(request, db)

    try:
        result = await pipeline.process_raw_text(body.text, body.source_type)
        return result.model_dump()
    except Exception as e:
        logger.exception("Intake text processing failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Intake processing failed: {e}")


@router.get("/intake/templates")
async def get_templates():
    """Return a list of common referral source types."""
    return {"source_types": SOURCE_TYPES}
