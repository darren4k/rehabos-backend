"""Clinical intelligence API endpoints — drug interactions, symptom correlation, chronic management."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from rehab_os.api.dependencies import get_current_user
from rehab_os.core.models import Provider

from rehab_os.clinical.drug_checker import (
    DrugCheckResult,
    SideEffectCorrelation,
    check_drug_interactions,
    correlate_symptoms,
)
from rehab_os.clinical.chronic_management import (
    ClinicalAlert,
    check_for_alerts,
    store_clinical_snapshot,
    get_patient_snapshots,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/clinical-intelligence")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class DrugCheckRequest(BaseModel):
    medications: list[str]
    symptoms: Optional[list[str]] = None
    diagnoses: Optional[list[str]] = None


class SymptomCorrelateRequest(BaseModel):
    medications: list[str]
    symptoms: list[str]
    diagnoses: list[str]


class SnapshotRequest(BaseModel):
    patient_id: str
    medications: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    vitals: dict = Field(default_factory=dict)
    functional_status: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/drug-check", response_model=DrugCheckResult)
async def drug_check(body: DrugCheckRequest, request: Request, current_user: Provider = Depends(get_current_user)):
    """Check drug interactions and rehab-relevant side effects."""
    llm = request.app.state.llm_router
    result = await check_drug_interactions(body.medications, llm)

    # If symptoms/diagnoses provided, also correlate
    if body.symptoms and body.diagnoses:
        correlations = await correlate_symptoms(
            body.medications, body.symptoms, body.diagnoses, llm
        )
        # Merge correlations (avoid duplicates by medication+symptom key)
        existing = {(c.medication, c.symptom) for c in result.side_effect_correlations}
        for c in correlations:
            if (c.medication, c.symptom) not in existing:
                result.side_effect_correlations.append(c)

    return result


@router.post("/symptom-correlate", response_model=list[SideEffectCorrelation])
async def symptom_correlate(body: SymptomCorrelateRequest, request: Request, current_user: Provider = Depends(get_current_user)):
    """Correlate symptoms with medications or disease progression."""
    llm = request.app.state.llm_router
    return await correlate_symptoms(
        body.medications, body.symptoms, body.diagnoses, llm
    )


@router.post("/snapshot")
async def create_snapshot(body: SnapshotRequest, request: Request, current_user: Provider = Depends(get_current_user)):
    """Store clinical snapshot and check for alerts."""
    llm = request.app.state.llm_router
    memory = request.app.state.session_memory

    await store_clinical_snapshot(
        patient_id=body.patient_id,
        medications=body.medications,
        symptoms=body.symptoms,
        vitals=body.vitals,
        functional_status=body.functional_status,
        memory_service=memory,
    )

    current = body.model_dump()
    alerts = await check_for_alerts(body.patient_id, current, memory, llm)

    return {"stored": True, "alerts": [a.model_dump() for a in alerts]}


@router.get("/alerts/{patient_id}", response_model=list[ClinicalAlert])
async def get_alerts(patient_id: str, request: Request, current_user: Provider = Depends(get_current_user)):
    """Get clinical alerts from current + historical analysis."""
    llm = request.app.state.llm_router
    memory = request.app.state.session_memory

    # Build current snapshot from latest stored data
    snapshots = await get_patient_snapshots(patient_id, memory)
    current = snapshots[-1]["data"] if snapshots else {}

    return await check_for_alerts(patient_id, current, memory, llm)
