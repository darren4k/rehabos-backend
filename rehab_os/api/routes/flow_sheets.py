"""Flow sheet API routes for longitudinal clinical data tracking."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from rehab_os.api.dependencies import get_current_user
from rehab_os.clinical.flow_sheets import FlowSheetColumn, get_flow_sheet_service
from rehab_os.core.models import Provider

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/flow-sheets",
    dependencies=[Depends(get_current_user)],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class FlowSheetEntryRequest(BaseModel):
    encounter_id: str
    encounter_date: str = Field(..., description="ISO date string YYYY-MM-DD")
    data: dict[str, Any] = Field(..., description="Column key -> value mapping")


class FlowSheetEntryResponse(BaseModel):
    encounter_id: str
    encounter_date: str
    provider_id: str
    data: dict[str, Any]
    recorded_at: str


class FlowSheetColumnResponse(BaseModel):
    key: str
    label: str
    unit: str
    category: str
    body_region: str | None = None


class TrendDataPoint(BaseModel):
    date: str
    value: Any
    encounter_id: str
    provider_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/columns/{discipline}", response_model=list[FlowSheetColumnResponse])
async def get_columns(discipline: str):
    """Get available flow sheet columns for a discipline (PT/OT/SLP)."""
    svc = get_flow_sheet_service()
    try:
        columns = svc.get_columns(discipline)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [
        FlowSheetColumnResponse(
            key=c.key, label=c.label, unit=c.unit,
            category=c.category, body_region=c.body_region,
        )
        for c in columns
    ]


@router.get("/{patient_id}")
async def get_flow_sheet(
    patient_id: str,
    discipline: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
):
    """Get flow sheet data for a patient, optionally filtered by discipline and date range."""
    svc = get_flow_sheet_service()
    entries = svc.get_flow_sheet(patient_id, discipline, date_from, date_to)
    return [
        FlowSheetEntryResponse(
            encounter_id=e.encounter_id,
            encounter_date=e.encounter_date,
            provider_id=e.provider_id,
            data=e.data,
            recorded_at=e.recorded_at.isoformat(),
        )
        for e in entries
    ]


@router.post("/{patient_id}/entry", response_model=FlowSheetEntryResponse)
async def record_entry(
    patient_id: str,
    body: FlowSheetEntryRequest,
    current_user: Provider = Depends(get_current_user),
):
    """Record a new flow sheet entry for a patient encounter."""
    svc = get_flow_sheet_service()
    entry = svc.record_entry(
        patient_id=patient_id,
        encounter_id=body.encounter_id,
        encounter_date=body.encounter_date,
        provider_id=str(current_user.id),
        data=body.data,
    )
    return FlowSheetEntryResponse(
        encounter_id=entry.encounter_id,
        encounter_date=entry.encounter_date,
        provider_id=entry.provider_id,
        data=entry.data,
        recorded_at=entry.recorded_at.isoformat(),
    )


@router.get("/{patient_id}/trend/{column_key}", response_model=list[TrendDataPoint])
async def get_trend(patient_id: str, column_key: str):
    """Get trend data for a specific measure across encounters (for charting)."""
    svc = get_flow_sheet_service()
    data = svc.get_trending_data(patient_id, column_key)
    return data


@router.get("/{patient_id}/summary")
async def get_summary(patient_id: str, discipline: str = "PT"):
    """Get summary statistics for a patient's flow sheet."""
    svc = get_flow_sheet_service()
    try:
        return svc.get_summary(patient_id, discipline)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
