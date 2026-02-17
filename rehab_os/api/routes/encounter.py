"""Encounter API — Brain-powered clinical documentation.

Replaces the rigid conversation script with an intelligent orchestrator
that understands clinical context and drives the documentation flow.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from rehab_os.encounter.brain import EncounterBrain
from rehab_os.encounter.state import EncounterPhase, EncounterState

router = APIRouter(prefix="/encounter", tags=["encounter"])
logger = logging.getLogger(__name__)

# In-memory encounter store (keyed by encounter_id)
# In production this would be Redis or DB-backed
_encounters: dict[str, EncounterState] = {}


# ── Request / Response Models ─────────────────────────────────────────────────


class TurnRequest(BaseModel):
    """A single therapist utterance."""

    encounter_id: Optional[str] = None
    utterance: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None


class TurnResponse(BaseModel):
    """Brain's response to a turn."""

    encounter_id: str
    response: str
    suggestions: list[str] = Field(default_factory=list)
    state: EncounterState
    phase: str
    completeness: float
    missing_critical: list[str] = Field(default_factory=list)
    missing_recommended: list[str] = Field(default_factory=list)
    ready_to_generate: bool = False


class GenerateRequest(BaseModel):
    """Request to generate a SOAP note from encounter state."""

    encounter_id: str
    force: bool = False  # generate even with missing critical items


class GenerateResponse(BaseModel):
    """Generated SOAP note with clinical intelligence."""

    encounter_id: str
    note_type: str
    content: str
    sections: dict[str, str]
    clinical_data: Optional[dict] = None
    compliance: dict = Field(default_factory=dict)
    billing_codes: list[dict] = Field(default_factory=list)
    word_count: int = 0
    # Clinical intelligence (populated after SOAP generation)
    defensibility: Optional[dict] = None
    drug_warnings: Optional[dict] = None
    evidence_suggestions: Optional[dict] = None
    clinical_alerts: list[dict] = Field(default_factory=list)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/turn", response_model=TurnResponse)
async def encounter_turn(req: TurnRequest, request: Request):
    """Process a therapist utterance and get an intelligent response.

    This is the main conversation endpoint. The Brain:
    1. Parses structured data from the utterance
    2. Updates the encounter state
    3. Builds a dynamic prompt with patient context
    4. Returns a smart, contextual response
    """
    llm_router = request.app.state.llm_router
    session_memory = getattr(request.app.state, "session_memory", None)

    # Get or create encounter state
    enc_id = req.encounter_id
    if enc_id and enc_id in _encounters:
        state = _encounters[enc_id]
    else:
        enc_id = enc_id or hashlib.sha256(
            datetime.now(timezone.utc).isoformat().encode()
        ).hexdigest()[:16]
        state = EncounterState(encounter_id=enc_id)

    # Inject patient context if provided
    if req.patient_id and not state.patient_id:
        state.patient_id = req.patient_id
    if req.patient_name and not state.patient_name:
        state.patient_name = req.patient_name

    # Process the turn
    brain = EncounterBrain(llm_router=llm_router, session_memory=session_memory)
    state, response, suggestions = await brain.process_turn(state, req.utterance)

    # Store updated state
    _encounters[enc_id] = state

    ready = state.phase == EncounterPhase.REVIEW or (
        len(state.missing_critical()) == 0 and state.turn_count > 3
    )

    return TurnResponse(
        encounter_id=enc_id,
        response=response,
        suggestions=suggestions,
        state=state,
        phase=state.phase.value,
        completeness=state.completeness_score(),
        missing_critical=state.missing_critical(),
        missing_recommended=state.missing_recommended(),
        ready_to_generate=ready,
    )


@router.post("/generate", response_model=GenerateResponse)
async def encounter_generate(req: GenerateRequest, request: Request):
    """Generate a SOAP note from the collected encounter state.

    Uses the structured data collected by the Brain (not raw transcript)
    to produce a complete, Medicare-compliant clinical note.
    After SOAP generation, runs clinical intelligence checks in parallel:
    - Medicare defensibility audit
    - Drug interaction check (if medications known)
    - Evidence-based treatment suggestions
    - Clinical alerts (trend detection)
    """
    if req.encounter_id not in _encounters:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Encounter not found")

    state = _encounters[req.encounter_id]

    # Check if we have enough data
    missing = state.missing_critical()
    if missing and not req.force:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=f"Missing critical items: {', '.join(missing)}. Set force=true to generate anyway.",
        )

    llm_router = request.app.state.llm_router
    session_memory = getattr(request.app.state, "session_memory", None)

    # 1. Generate SOAP note from structured state
    note = await _generate_soap_from_state(state, llm_router)

    # 2. Run clinical intelligence in parallel (non-blocking — failures don't kill the note)
    intel = await _run_clinical_intelligence(state, note, llm_router, session_memory)
    note.defensibility = intel.get("defensibility")
    note.drug_warnings = intel.get("drug_warnings")
    note.evidence_suggestions = intel.get("evidence_suggestions")
    note.clinical_alerts = intel.get("clinical_alerts", [])

    # Merge defensibility into compliance dict for backward compat
    if note.defensibility:
        note.compliance = {
            "defensibility_score": note.defensibility.get("overall_score", 0),
            "warnings": note.defensibility.get("warnings", []),
            "failures": note.defensibility.get("failures", []),
        }

    # Mark encounter complete
    state.phase = EncounterPhase.COMPLETE
    _encounters[req.encounter_id] = state

    return note


@router.get("/state/{encounter_id}")
async def get_encounter_state(encounter_id: str):
    """Get the current encounter state."""
    if encounter_id not in _encounters:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Encounter not found")
    return _encounters[encounter_id]


@router.delete("/state/{encounter_id}")
async def clear_encounter(encounter_id: str):
    """Clear an encounter state."""
    _encounters.pop(encounter_id, None)
    return {"status": "cleared", "encounter_id": encounter_id}


# ── Clinical Intelligence Pipeline ────────────────────────────────────────────


async def _run_clinical_intelligence(
    state: EncounterState,
    note: GenerateResponse,
    llm_router: Any,
    session_memory: Any,
) -> dict:
    """Run clinical intelligence checks in parallel after SOAP generation.

    Each check is independent — failures are logged but don't block the note.
    Returns a dict with keys: defensibility, drug_warnings, evidence_suggestions, clinical_alerts.
    """
    results: dict[str, Any] = {}

    # Build tasks list — only add checks that have enough data to be meaningful
    tasks: dict[str, Any] = {}

    # 1. Defensibility audit (always run — checks the generated SOAP)
    tasks["defensibility"] = _check_defensibility(state, note, llm_router)

    # 2. Drug interaction check (only if medications known)
    if state.history.medications:
        tasks["drug_warnings"] = _check_drugs(state.history.medications, llm_router)

    # 3. Evidence-based suggestions (if we have diagnosis or interventions)
    if state.history.diagnosis or state.objective.interventions:
        tasks["evidence_suggestions"] = _check_evidence(state, note, llm_router)

    # 4. Clinical alerts / trend detection (if patient has history)
    if state.patient_id and session_memory:
        tasks["clinical_alerts"] = _check_alerts(state, session_memory, llm_router)

    if not tasks:
        return results

    # Run all checks in parallel
    keys = list(tasks.keys())
    outcomes = await asyncio.gather(*tasks.values(), return_exceptions=True)

    for key, outcome in zip(keys, outcomes):
        if isinstance(outcome, Exception):
            logger.warning("Clinical intel '%s' failed: %s", key, outcome)
        else:
            results[key] = outcome

    return results


async def _check_defensibility(
    state: EncounterState, note: GenerateResponse, llm_router: Any
) -> dict:
    """Run Medicare defensibility audit on the generated SOAP note."""
    from rehab_os.clinical.defensibility import check_defensibility

    note_content = note.sections
    structured_data = note.clinical_data or {}
    patient_context = {
        "discipline": state.discipline,
        "note_type": state.note_type or "daily_note",
        "patient_name": state.patient_name,
        "diagnosis": state.history.diagnosis,
        "medications": state.history.medications,
    }

    result = await check_defensibility(
        note_content=note_content,
        note_type=state.note_type or "daily_note",
        structured_data=structured_data,
        patient_context=patient_context,
        llm=llm_router,
    )
    return result.model_dump()


async def _check_drugs(medications: list[str], llm_router: Any) -> dict:
    """Check for drug interactions and rehab-relevant side effects."""
    from rehab_os.clinical.drug_checker import check_drug_interactions

    result = await check_drug_interactions(medications=medications, llm=llm_router)
    return result.model_dump()


async def _check_evidence(
    state: EncounterState, note: GenerateResponse, llm_router: Any
) -> dict:
    """Suggest evidence-based treatments the therapist may have missed."""
    from rehab_os.clinical.evidence_engine import suggest_evidence_based_treatments

    current_interventions = [i.name for i in state.objective.interventions]
    functional_deficits = []
    if state.subjective.chief_complaint:
        functional_deficits.append(state.subjective.chief_complaint)
    patient_context = {
        "discipline": state.discipline,
        "patient_name": state.patient_name,
    }

    result = await suggest_evidence_based_treatments(
        diagnosis=state.history.diagnosis,
        current_interventions=current_interventions,
        functional_deficits=functional_deficits,
        patient_context=patient_context,
        note_type=state.note_type or "daily_note",
        llm=llm_router,
    )
    return result.model_dump()


async def _check_alerts(
    state: EncounterState, session_memory: Any, llm_router: Any
) -> list[dict]:
    """Detect clinical trends and alerts from longitudinal data."""
    from rehab_os.clinical.chronic_management import check_for_alerts

    current_snapshot = {
        "medications": state.history.medications,
        "symptoms": [state.subjective.chief_complaint] if state.subjective.chief_complaint else [],
        "vitals": state.objective.vitals.model_dump() if state.objective.vitals else {},
        "functional_status": {
            "pain_level": state.subjective.pain_level,
            "rom": [r.model_dump() for r in state.objective.rom],
            "interventions": [i.name for i in state.objective.interventions],
        },
    }

    alerts = await check_for_alerts(
        patient_id=state.patient_id,
        current_snapshot=current_snapshot,
        memory_service=session_memory,
        llm=llm_router,
    )
    return [a.model_dump() for a in alerts]


# ── SOAP Generation from Structured State ─────────────────────────────────────


SOAP_FROM_STATE_PROMPT = """You are a clinical documentation assistant generating a SOAP note from structured encounter data.

PATIENT: {patient_name}
NOTE TYPE: {note_type}
DATE: {date_of_service}
DISCIPLINE: {discipline}

COLLECTED DATA:
{collected_summary}

PATIENT HISTORY:
{history_summary}

Generate a complete, Medicare-compliant SOAP note. Return ONLY valid JSON:
{{
  "subjective": "...",
  "objective": "...",
  "assessment": "...",
  "plan": "...",
  "billing_codes": [
    {{"code": "97110", "description": "Therapeutic Exercise", "units": 1}}
  ]
}}

RULES:
- ONLY document what was collected. Use "[NOT DOCUMENTED]" for missing data.
- Upgrade casual language to skilled terminology (97110, 97116, etc.)
- Reference prior visit data for comparison when available.
- Include skilled justification in Assessment.
- Keep the therapist's clinical voice — not generic templates.
- Add brief Medicare defensibility language.
- Map interventions to CPT codes in billing_codes."""


async def _generate_soap_from_state(
    state: EncounterState, llm_router
) -> GenerateResponse:
    """Generate SOAP note from structured encounter state."""
    import json
    import re

    import httpx

    # Build collected summary
    collected = state.summary_for_prompt()

    # Build history summary
    history_parts = []
    if state.history.last_encounters:
        for enc in state.history.last_encounters[:3]:
            h = f"  {enc.get('date', '?')}: {enc.get('summary', 'No summary')}"
            if enc.get("pain_level") is not None:
                h += f" (pain {enc['pain_level']}/10)"
            history_parts.append(h)
    history_summary = "\n".join(history_parts) if history_parts else "No prior encounters available"

    if state.history.active_goals:
        history_summary += "\nActive Goals:\n"
        for g in state.history.active_goals[:3]:
            history_summary += f"  - {g.get('area', '?')}: {g.get('current', '?')} → {g.get('target', '?')}\n"

    prompt = SOAP_FROM_STATE_PROMPT.format(
        patient_name=state.patient_name or "Unknown",
        note_type=state.note_type or "daily_note",
        date_of_service=state.date_of_service or "today",
        discipline=state.discipline,
        collected_summary=collected,
        history_summary=history_summary,
    )

    # Use smart model (80b) via Ollama directly for full SOAP generation
    OLLAMA_URL = "http://192.168.68.127:11434"
    SMART_MODEL = "qwen3-next:80b"

    parsed = None
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": SMART_MODEL,
                    "stream": False,
                    "options": {"num_predict": 4096, "temperature": 0.3},
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Generate the SOAP note from the collected data above."},
                    ],
                },
            )
            if res.status_code == 200:
                raw = res.json().get("message", {}).get("content", "")
                raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    raw = raw.strip()
                parsed = json.loads(raw)
    except Exception as e:
        logger.error("SOAP generation via Ollama failed: %s", e)

    if parsed is None:
        parsed = _assemble_fallback_soap(state)

    sections = {
        "subjective": parsed.get("subjective", ""),
        "objective": parsed.get("objective", ""),
        "assessment": parsed.get("assessment", ""),
        "plan": parsed.get("plan", ""),
    }

    content_lines = []
    for section_name in ("subjective", "objective", "assessment", "plan"):
        val = sections.get(section_name, "")
        if val:
            content_lines.append(f"{section_name.upper()}:\n{val}")
    content = "\n\n".join(content_lines)

    # Build clinical_data from structured state
    clinical_data = {
        "rom": [r.model_dump() for r in state.objective.rom],
        "mmt": [m.model_dump() for m in state.objective.mmt],
        "standardized_tests": [t.model_dump() for t in state.objective.standardized_tests],
        "vitals": state.objective.vitals.model_dump() if state.objective.vitals else None,
        "interventions": [i.model_dump() for i in state.objective.interventions],
    }

    # Generate validated billing codes from structured state
    from rehab_os.billing.engine import generate_billing

    billing = generate_billing(
        interventions=[i.model_dump() for i in state.objective.interventions],
        diagnosis_list=state.history.diagnosis,
        chief_complaint=state.subjective.chief_complaint,
        pain_location=state.subjective.pain_location,
        note_type=state.note_type or "daily_note",
    )

    # Use validated billing codes; fall back to LLM-parsed if billing engine produced none
    billing_codes = [
        {
            "code": line.code,
            "description": line.description,
            "units": line.units,
            "minutes": line.minutes,
            "category": line.category,
        }
        for line in billing.cpt_lines
        if line.units > 0
    ] or parsed.get("billing_codes", [])

    # Add ICD-10 codes to clinical_data
    if billing.icd10_codes:
        clinical_data["icd10_codes"] = [
            {"code": s.code, "description": s.description, "confidence": s.confidence}
            for s in billing.icd10_codes
        ]

    # Add billing warnings to compliance
    billing_compliance = {}
    if billing.warnings:
        billing_compliance["billing_warnings"] = billing.warnings
    billing_compliance["total_units"] = billing.total_units
    billing_compliance["eight_minute_valid"] = billing.unit_validation.is_valid

    return GenerateResponse(
        encounter_id=state.encounter_id,
        note_type=state.note_type or "daily_note",
        content=content,
        sections=sections,
        clinical_data=clinical_data,
        compliance=billing_compliance,
        billing_codes=billing_codes,
        word_count=len(content.split()),
    )


def _assemble_fallback_soap(state: EncounterState) -> dict:
    """Assemble a basic SOAP from structured state when LLM is unavailable."""
    s_parts = []
    if state.subjective.chief_complaint:
        s_parts.append(f"Patient presents with {state.subjective.chief_complaint}.")
    if state.subjective.pain_level is not None:
        loc = f" ({state.subjective.pain_location})" if state.subjective.pain_location else ""
        s_parts.append(f"Pain rated {state.subjective.pain_level}/10{loc}.")
    if state.subjective.hep_compliance:
        s_parts.append(f"HEP: {state.subjective.hep_compliance}.")

    o_parts = []
    if state.objective.vitals:
        v = state.objective.vitals
        vitals = []
        if v.bp:
            vitals.append(f"BP {v.bp}")
        if v.pulse:
            vitals.append(f"HR {v.pulse}")
        if v.spo2:
            vitals.append(f"SpO2 {v.spo2}%")
        if vitals:
            o_parts.append(f"Vitals: {', '.join(vitals)}.")
    if state.objective.interventions:
        names = [i.name for i in state.objective.interventions]
        o_parts.append(f"Interventions: {', '.join(names)}.")
    if state.objective.rom:
        for r in state.objective.rom:
            o_parts.append(f"ROM: {r.side} {r.joint} {r.motion} {r.value}°.")
    if state.objective.tolerance:
        o_parts.append(f"Tolerance: {state.objective.tolerance}.")

    a_parts = []
    if state.assessment.progress:
        a_parts.append(state.assessment.progress)
    else:
        a_parts.append("[Assessment to be completed by therapist]")

    p_parts = []
    if state.plan.next_visit:
        p_parts.append(state.plan.next_visit)
    if state.plan.frequency:
        p_parts.append(state.plan.frequency)
    if not p_parts:
        p_parts.append("[Plan to be completed by therapist]")

    return {
        "subjective": " ".join(s_parts) or "[NOT DOCUMENTED]",
        "objective": " ".join(o_parts) or "[NOT DOCUMENTED]",
        "assessment": " ".join(a_parts),
        "plan": " ".join(p_parts),
        "billing_codes": [],
    }
