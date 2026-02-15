"""Drug interaction checker and symptom-medication correlator.

Uses the LLM as a clinical knowledge engine to identify drug-drug interactions,
rehabilitation-relevant side effects, fall-risk medications, and therapy timing
considerations.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from rehab_os.llm.base import Message, MessageRole

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: str = Field(description="minor | moderate | major | contraindicated")
    description: str
    clinical_significance: str
    recommendation: str


class SideEffectCorrelation(BaseModel):
    medication: str
    symptom: str
    likelihood: str = Field(description="common | uncommon | rare")
    description: str
    recommendation: str = Field(description="monitor | report to MD | consider alternative")


class DrugCheckResult(BaseModel):
    interactions: list[DrugInteraction] = Field(default_factory=list)
    side_effect_correlations: list[SideEffectCorrelation] = Field(default_factory=list)
    therapy_considerations: list[str] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    fall_risk_medications: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DRUG_CHECK_SYSTEM = """\
You are a clinical pharmacology expert assisting rehabilitation therapists.
Given a medication list, analyse:
1. Drug-drug interactions (severity: minor/moderate/major/contraindicated).
2. Side effects relevant to physical/occupational/speech therapy (drowsiness, orthostatic hypotension, bleeding risk, dizziness, fatigue, cognitive effects).
3. Fall-risk medications (sedatives, antihypertensives, opioids, anticholinergics, etc.).
4. Exercise/activity precautions (e.g., beta-blockers limiting HR response).
5. Therapy timing considerations (e.g., schedule therapy at peak Parkinson's medication effectiveness).

Return ONLY valid JSON matching this schema (no markdown fences):
{
  "interactions": [{"drug_a":"…","drug_b":"…","severity":"…","description":"…","clinical_significance":"…","recommendation":"…"}],
  "side_effect_correlations": [{"medication":"…","symptom":"…","likelihood":"…","description":"…","recommendation":"…"}],
  "therapy_considerations": ["…"],
  "alerts": ["…"],
  "fall_risk_medications": ["…"]
}
If there are no items for a field, use an empty list.
"""

_SYMPTOM_CORRELATE_SYSTEM = """\
You are a clinical pharmacology expert. Given a patient's medications, reported symptoms, and diagnoses, determine which symptoms are likely medication side effects vs disease-related.
Return ONLY valid JSON — a list of objects:
[{"medication":"…","symptom":"…","likelihood":"common|uncommon|rare","description":"…","recommendation":"monitor|report to MD|consider alternative"}]
If a symptom is more likely disease-related than medication-related, set medication to "N/A — likely disease-related" and describe the rationale.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def check_drug_interactions(
    medications: list[str],
    llm: Any,  # LLMRouter
) -> DrugCheckResult:
    """Check for drug interactions and rehab-relevant side effects."""
    if not medications:
        return DrugCheckResult()

    med_list = ", ".join(medications)
    messages = [
        Message(role=MessageRole.SYSTEM, content=_DRUG_CHECK_SYSTEM),
        Message(role=MessageRole.USER, content=f"Medications: {med_list}"),
    ]

    try:
        result = await llm.complete_structured(
            messages, DrugCheckResult, temperature=0.2, max_tokens=4096
        )
        return result
    except Exception:
        # Fallback: try unstructured and parse
        logger.warning("Structured drug check failed, trying unstructured parse")
        resp = await llm.complete(messages, temperature=0.2, max_tokens=4096)
        return _parse_drug_check_json(resp.content)


async def correlate_symptoms(
    medications: list[str],
    symptoms: list[str],
    diagnoses: list[str],
    llm: Any,  # LLMRouter
) -> list[SideEffectCorrelation]:
    """Correlate symptoms with medications or disease progression."""
    if not medications or not symptoms:
        return []

    user_msg = (
        f"Medications: {', '.join(medications)}\n"
        f"Symptoms: {', '.join(symptoms)}\n"
        f"Diagnoses: {', '.join(diagnoses)}"
    )
    messages = [
        Message(role=MessageRole.SYSTEM, content=_SYMPTOM_CORRELATE_SYSTEM),
        Message(role=MessageRole.USER, content=user_msg),
    ]

    try:
        resp = await llm.complete(messages, temperature=0.2, max_tokens=4096)
        raw = _extract_json(resp.content)
        items = json.loads(raw)
        return [SideEffectCorrelation(**item) for item in items]
    except Exception as e:
        logger.error("Symptom correlation failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Extract JSON from LLM output that may contain markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        text = "\n".join(lines[start:end])
    return text.strip()


def _parse_drug_check_json(text: str) -> DrugCheckResult:
    """Parse DrugCheckResult from raw LLM JSON text."""
    try:
        raw = _extract_json(text)
        data = json.loads(raw)
        return DrugCheckResult(**data)
    except Exception as e:
        logger.error("Failed to parse drug check JSON: %s", e)
        return DrugCheckResult(alerts=[f"Drug check parsing error: {e}"])
