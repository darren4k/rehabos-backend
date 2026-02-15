"""Chronic disease management via memU longitudinal tracking.

Stores clinical snapshots and detects trends (symptom worsening, vital sign
changes, functional decline, readmission risk) across visits.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from rehab_os.llm.base import Message, MessageRole

logger = logging.getLogger(__name__)

_NAMESPACE_PREFIX = "rehab:clinical:"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ClinicalAlert(BaseModel):
    alert_type: str = Field(description="symptom_worsening | drug_interaction | fall_risk | vital_trend | readmission_risk")
    severity: str = Field(description="info | warning | critical")
    description: str
    recommendation: str
    related_data: dict = Field(default_factory=dict)


class ClinicalSnapshot(BaseModel):
    patient_id: str
    timestamp: str
    medications: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    vitals: dict = Field(default_factory=dict)
    functional_status: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ALERT_SYSTEM = """\
You are a clinical decision support system for rehabilitation medicine.
Compare the CURRENT clinical snapshot with HISTORICAL data and identify:
1. Symptom worsening trends
2. Vital sign trends (e.g., progressive BP increase, HR changes)
3. Functional decline patterns
4. Readmission risk factors
5. Drug regimen changes needing therapy adjustment

Return ONLY valid JSON — a list of alert objects:
[{"alert_type":"symptom_worsening|drug_interaction|fall_risk|vital_trend|readmission_risk",
  "severity":"info|warning|critical",
  "description":"…",
  "recommendation":"…",
  "related_data":{}}]
If no alerts, return [].
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def store_clinical_snapshot(
    patient_id: str,
    medications: list[str],
    symptoms: list[str],
    vitals: dict,
    functional_status: dict,
    memory_service: Any,  # SessionMemoryService
) -> None:
    """Store a clinical snapshot in memU for longitudinal tracking."""
    snapshot = ClinicalSnapshot(
        patient_id=patient_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        medications=medications,
        symptoms=symptoms,
        vitals=vitals,
        functional_status=functional_status,
    )

    record = {
        "type": "clinical_snapshot",
        "data": snapshot.model_dump(),
        "tags": ["clinical", "snapshot", f"patient:{patient_id}"],
    }

    scoped_id = f"{_NAMESPACE_PREFIX}{patient_id}"

    if memory_service.is_memu_available:
        try:
            memory_service._persist_record(
                scoped_id,
                record,
                tags=["clinical_snapshot"],
            )
            logger.info("Clinical snapshot stored in memU for patient %s", patient_id)
        except Exception as e:
            logger.warning("memU persist failed, using cache: %s", e)
            _cache_snapshot(memory_service, patient_id, record)
    else:
        _cache_snapshot(memory_service, patient_id, record)


async def get_patient_snapshots(
    patient_id: str,
    memory_service: Any,
) -> list[dict]:
    """Retrieve historical clinical snapshots for a patient."""
    scoped_id = f"{_NAMESPACE_PREFIX}{patient_id}"

    if memory_service.is_memu_available:
        try:
            history = memory_service._retrieve_history(scoped_id)
            return [
                r for r in history
                if isinstance(r, dict) and r.get("type") == "clinical_snapshot"
            ]
        except Exception as e:
            logger.warning("memU retrieval failed: %s", e)

    # Fallback to cache
    cache_key = f"clinical:{patient_id}"
    return memory_service._cache.get(cache_key, [])


async def check_for_alerts(
    patient_id: str,
    current_snapshot: dict,
    memory_service: Any,  # SessionMemoryService
    llm: Any,  # LLMRouter
) -> list[ClinicalAlert]:
    """Compare current visit against historical data to detect clinical alerts."""
    historical = await get_patient_snapshots(patient_id, memory_service)

    if not historical and not current_snapshot:
        return []

    # Build context for LLM
    history_summary = json.dumps(historical[-10:], indent=2) if historical else "No prior visits."
    current_summary = json.dumps(current_snapshot, indent=2)

    messages = [
        Message(role=MessageRole.SYSTEM, content=_ALERT_SYSTEM),
        Message(
            role=MessageRole.USER,
            content=(
                f"CURRENT SNAPSHOT:\n{current_summary}\n\n"
                f"HISTORICAL DATA (most recent 10 visits):\n{history_summary}"
            ),
        ),
    ]

    try:
        resp = await llm.complete(messages, temperature=0.2, max_tokens=4096)
        raw = resp.content.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
        items = json.loads(raw.strip())
        return [ClinicalAlert(**item) for item in items]
    except Exception as e:
        logger.error("Alert check failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_snapshot(memory_service: Any, patient_id: str, record: dict) -> None:
    """Cache a snapshot in the in-memory fallback."""
    cache_key = f"clinical:{patient_id}"
    if cache_key not in memory_service._cache:
        memory_service._cache[cache_key] = []
    memory_service._cache[cache_key].append(record)
