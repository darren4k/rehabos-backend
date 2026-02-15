"""Cross-namespace patient history API endpoint."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request

from rehab_os.memory.cross_namespace import (
    get_patient_history,
    EncounterRecord,
    DEFAULT_NAMESPACES,
)

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("/{patient_id}/history")
async def patient_history(
    patient_id: str,
    request: Request,
    namespace: Optional[str] = Query(None, description="Filter to a single namespace (e.g. 'rehab' or 'docpilot')"),
):
    """Return combined encounter history for a patient across all services."""
    memory_service = request.app.state.session_memory

    namespaces = [namespace] if namespace else list(DEFAULT_NAMESPACES)
    records = get_patient_history(memory_service, patient_id, namespaces)

    return {
        "patient_id": patient_id,
        "namespaces_queried": namespaces,
        "total": len(records),
        "encounters": [r.to_dict() for r in records],
    }
