"""Consultation endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from rehab_os.api.rate_limit import SlidingWindowRateLimiter

from rehab_os.models.output import (
    ClinicalRequest,
    ConsultationResponse,
    DocumentationType,
)
from rehab_os.models.patient import CareSetting, Discipline, PatientContext

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limit: 10 requests per minute per API key (or IP)
_consult_limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=60)


class ConsultRequest(BaseModel):
    """API request for clinical consultation."""

    query: str = Field(..., description="Clinical question or consultation request")
    patient: Optional[PatientContext] = Field(
        None, description="Patient context (optional for simple queries)"
    )
    discipline: Discipline = Field(default=Discipline.PT)
    setting: CareSetting = Field(default=CareSetting.OUTPATIENT)
    task_type: str = Field(
        default="full_consult",
        description="Task type: full_consult, diagnosis_only, plan_only, evidence_search, safety_only",
    )
    include_documentation: bool = Field(default=False)
    documentation_type: Optional[DocumentationType] = None
    skip_qa: bool = Field(default=False, description="Skip QA review for faster response")


class QuickConsultRequest(BaseModel):
    """Simplified request for quick consultations."""

    query: str = Field(..., description="Clinical question")
    age: int = Field(default=50, ge=0, le=120)
    sex: str = Field(default="other", pattern="^(male|female|other)$")
    discipline: Discipline = Field(default=Discipline.PT)
    setting: CareSetting = Field(default=CareSetting.OUTPATIENT)


@router.post("/consult", response_model=ConsultationResponse)
async def create_consultation(
    request: Request,
    consult_request: ConsultRequest,
    _rate=Depends(_consult_limiter),
) -> ConsultationResponse:
    """Process a clinical consultation through the agent pipeline.

    This endpoint runs the full multi-agent pipeline:
    1. Safety screening (RedFlagAgent)
    2. Diagnosis classification (DiagnosisAgent)
    3. Evidence retrieval (EvidenceAgent)
    4. Treatment planning (PlanAgent)
    5. Outcome measure selection (OutcomeAgent)
    6. Documentation generation (optional)
    7. Quality assurance review (QALearningAgent)
    """
    orchestrator = request.app.state.orchestrator

    # Build patient context if not provided
    if not consult_request.patient:
        consult_request.patient = PatientContext(
            age=50,
            sex="other",
            chief_complaint=consult_request.query,
            discipline=consult_request.discipline,
            setting=consult_request.setting,
        )

    # Create internal request
    clinical_request = ClinicalRequest(
        query=consult_request.query,
        patient=consult_request.patient,
        discipline=consult_request.discipline,
        setting=consult_request.setting,
        task_type=consult_request.task_type,
        include_documentation=consult_request.include_documentation,
        documentation_type=consult_request.documentation_type,
    )

    try:
        response = await orchestrator.process(
            clinical_request,
            skip_qa=consult_request.skip_qa,
        )
        return response

    except Exception as e:
        logger.exception(f"Consultation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Consultation processing failed: {str(e)}",
        )


@router.post("/consult/quick", response_model=ConsultationResponse)
async def quick_consultation(
    request: Request,
    consult_request: QuickConsultRequest,
) -> ConsultationResponse:
    """Quick consultation with minimal input.

    Runs a faster pipeline with:
    - Safety screening
    - Basic diagnosis
    - Treatment plan
    - No documentation
    - No QA review
    """
    orchestrator = request.app.state.orchestrator

    patient = PatientContext(
        age=consult_request.age,
        sex=consult_request.sex,
        chief_complaint=consult_request.query,
        discipline=consult_request.discipline,
        setting=consult_request.setting,
    )

    clinical_request = ClinicalRequest(
        query=consult_request.query,
        patient=patient,
        discipline=consult_request.discipline,
        setting=consult_request.setting,
        task_type="full_consult",
        include_documentation=False,
    )

    try:
        response = await orchestrator.process(clinical_request, skip_qa=True)
        return response

    except Exception as e:
        logger.exception(f"Quick consultation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Consultation processing failed: {str(e)}",
        )


@router.post("/consult/safety")
async def safety_check(
    request: Request,
    consult_request: ConsultRequest,
) -> dict:
    """Run safety screening only.

    Returns red flags and safety assessment without full treatment planning.
    """
    orchestrator = request.app.state.orchestrator

    if not consult_request.patient:
        consult_request.patient = PatientContext(
            age=50,
            sex="other",
            chief_complaint=consult_request.query,
            discipline=consult_request.discipline,
            setting=consult_request.setting,
        )

    clinical_request = ClinicalRequest(
        query=consult_request.query,
        patient=consult_request.patient,
        discipline=consult_request.discipline,
        setting=consult_request.setting,
        task_type="safety_only",
    )

    try:
        response = await orchestrator.process(clinical_request)
        return {
            "safety": response.safety.model_dump(),
            "processing_notes": response.processing_notes,
        }

    except Exception as e:
        logger.exception(f"Safety check error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Safety check failed: {str(e)}",
        )
