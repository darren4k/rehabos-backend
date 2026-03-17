"""OASIS-E assessment API routes for RehabOS.

Provides endpoints for creating, updating, validating, and querying
OASIS-E home health assessments per CMS requirements.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from rehab_os.api.dependencies import get_current_user
from rehab_os.core.models import Provider
from rehab_os.clinical.oasis import (
    OASISAssessmentType,
    OASISService,
    OASIS_ITEMS,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/oasis",
    tags=["oasis"],
    dependencies=[Depends(get_current_user)],
)

# Module-level service instance (in production, inject via app.state)
_oasis_service = OASISService()


def _get_service() -> OASISService:
    return _oasis_service


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------

class CreateAssessmentRequest(BaseModel):
    assessment_type: OASISAssessmentType
    patient_id: str


class SaveProgressRequest(BaseModel):
    responses: dict[str, Any] = Field(
        ..., description="Map of OASIS item IDs to response values"
    )


class AutoPopulateRequest(BaseModel):
    encounter_data: dict[str, Any] = Field(
        ..., description="Encounter/SOAP data to map into OASIS items"
    )


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.post("/assessments", status_code=201)
async def create_assessment(
    req: CreateAssessmentRequest,
    user: Provider = Depends(get_current_user),
    service: OASISService = Depends(_get_service),
) -> dict:
    """Create a new OASIS-E assessment for a patient."""
    assessment = service.create_assessment(req.assessment_type, req.patient_id)
    return assessment.to_dict()


@router.get("/assessments/{assessment_id}")
async def get_assessment(
    assessment_id: str,
    user: Provider = Depends(get_current_user),
    service: OASISService = Depends(_get_service),
) -> dict:
    """Get an OASIS assessment with all responses."""
    assessment = service.get_assessment(assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment.to_dict()


@router.put("/assessments/{assessment_id}")
async def save_progress(
    assessment_id: str,
    req: SaveProgressRequest,
    user: Provider = Depends(get_current_user),
    service: OASISService = Depends(_get_service),
) -> dict:
    """Save partial responses for an in-progress assessment."""
    try:
        assessment = service.save_progress(assessment_id, req.responses)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return assessment.to_dict()


@router.post("/assessments/{assessment_id}/validate")
async def validate_assessment(
    assessment_id: str,
    user: Provider = Depends(get_current_user),
    service: OASISService = Depends(_get_service),
) -> dict:
    """Validate an assessment for CMS submission readiness."""
    try:
        issues = service.validate_for_submission(assessment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    assessment = service.get_assessment(assessment_id)
    return {
        "assessment_id": assessment_id,
        "valid": len([i for i in issues if i["severity"] == "error"]) == 0,
        "completion_pct": assessment.get_completion_pct() if assessment else 0,
        "issues": issues,
    }


@router.post("/assessments/{assessment_id}/auto-populate")
async def auto_populate(
    assessment_id: str,
    req: AutoPopulateRequest,
    user: Provider = Depends(get_current_user),
    service: OASISService = Depends(_get_service),
) -> dict:
    """Auto-fill OASIS items from encounter/SOAP data."""
    try:
        filled = await service.auto_populate_from_encounter(
            assessment_id, req.encounter_data
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "assessment_id": assessment_id,
        "auto_filled_count": len(filled),
        "auto_filled_items": filled,
    }


@router.get("/recert-due")
async def get_recert_due(
    user: Provider = Depends(get_current_user),
    service: OASISService = Depends(_get_service),
) -> dict:
    """Get patients due for 60-day recertification."""
    due = service.get_recert_due_patients()
    return {"patients_due": due, "count": len(due)}


@router.get("/items")
async def list_oasis_items(
    section: Optional[str] = None,
    user: Provider = Depends(get_current_user),
) -> dict:
    """List available OASIS-E items, optionally filtered by section."""
    items = []
    for item_id, item in OASIS_ITEMS.items():
        if section and item.section != section:
            continue
        items.append({
            "item_id": item.item_id,
            "section": item.section,
            "label": item.label,
            "description": item.description,
            "response_type": item.response_type,
            "options": item.options,
            "required": item.required,
            "skip_logic": item.skip_logic,
        })
    sections = sorted({item.section for item in OASIS_ITEMS.values()})
    return {"items": items, "count": len(items), "sections": sections}
