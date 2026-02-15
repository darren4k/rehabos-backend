"""Evidence-Based Practice Engine.

Uses LLM as a clinical knowledge engine to suggest research-backed treatments,
outcome measures, and goals that providers might miss.
"""

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from rehab_os.llm import LLMRouter, Message, MessageRole

logger = logging.getLogger(__name__)


class EvidenceSuggestion(BaseModel):
    """A single evidence-based suggestion."""
    intervention: str
    evidence_level: str  # "Level I - Systematic Review", "Level II - RCT", etc.
    rationale: str
    source_summary: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    category: str  # intervention, outcome_measure, goal_suggestion, precaution, discharge_criteria


class TreatmentPlanReview(BaseModel):
    """Full evidence-based review of a treatment plan."""
    suggested_interventions: list[EvidenceSuggestion] = Field(default_factory=list)
    suggested_outcome_measures: list[EvidenceSuggestion] = Field(default_factory=list)
    suggested_goals: list[EvidenceSuggestion] = Field(default_factory=list)
    missing_elements: list[str] = Field(default_factory=list)
    defensibility_notes: list[str] = Field(default_factory=list)


EVIDENCE_SYSTEM_PROMPT = """You are an expert clinical research consultant for rehabilitation therapy (PT/OT/SLP).
Given a patient's clinical picture, suggest evidence-based treatments, outcome measures, and goals
the therapist may not have considered.

Your response MUST be valid JSON matching this schema:
{
  "suggested_interventions": [
    {
      "intervention": "Name of intervention or program",
      "evidence_level": "Level I - Systematic Review | Level II - RCT | Level III - Cohort Study | Level IV - Case Series | Expert Consensus",
      "rationale": "Why this applies to this specific patient",
      "source_summary": "Brief description of supporting evidence (guideline name, landmark study, etc.)",
      "relevance_score": 0.0-1.0,
      "category": "intervention"
    }
  ],
  "suggested_outcome_measures": [
    {
      "intervention": "Name of standardized test/measure",
      "evidence_level": "...",
      "rationale": "Why this measure is appropriate",
      "source_summary": "Psychometric properties, MDC/MCID values",
      "relevance_score": 0.0-1.0,
      "category": "outcome_measure"
    }
  ],
  "suggested_goals": [
    {
      "intervention": "SMART goal statement",
      "evidence_level": "...",
      "rationale": "Evidence basis for this functional target",
      "source_summary": "Normative data or recovery benchmarks",
      "relevance_score": 0.0-1.0,
      "category": "goal_suggestion"
    }
  ],
  "missing_elements": ["Element the current plan is missing"],
  "defensibility_notes": ["How to strengthen the plan for payer review"]
}

CLINICAL REASONING GUIDELINES:
1. Reference Clinical Practice Guidelines (CPGs) for the diagnoses
2. Suggest standardized outcome measures with known psychometric properties (MDC, MCID, norms)
3. Include evidence-based interventions the therapist may not have considered
4. Provide SMART goals tied to FUNCTIONAL outcomes (not just impairment-level)
5. Recommend frequency/duration based on evidence and CPGs
6. Suggest discharge criteria based on research benchmarks
7. Identify precautions or contraindications the therapist should document
8. Note any missing elements that could weaken Medicare defensibility

Be specific. Use actual test names, actual CPG references, actual intervention protocols.
Do NOT fabricate citations — reference well-known guidelines and landmark studies by name only if real."""


async def suggest_evidence_based_treatments(
    diagnosis: list[str],
    current_interventions: list[str],
    functional_deficits: list[str],
    patient_context: dict,
    note_type: str,
    llm: LLMRouter,
) -> TreatmentPlanReview:
    """Analyze the clinical picture and suggest evidence-based treatments.

    Args:
        diagnosis: List of diagnoses (e.g., ["CVA - L MCA", "R hemiparesis"])
        current_interventions: What the therapist is already doing
        functional_deficits: Current functional limitations
        patient_context: age, setting, comorbidities, discipline, etc.
        note_type: evaluation, daily_note, progress_note, discharge
        llm: LLM router instance

    Returns:
        TreatmentPlanReview with suggestions across all categories
    """
    user_content = json.dumps({
        "diagnoses": diagnosis,
        "current_interventions": current_interventions,
        "functional_deficits": functional_deficits,
        "patient_context": patient_context,
        "note_type": note_type,
    }, indent=2)

    messages = [
        Message(role=MessageRole.SYSTEM, content=EVIDENCE_SYSTEM_PROMPT),
        Message(
            role=MessageRole.USER,
            content=f"Analyze this clinical picture and provide evidence-based suggestions:\n\n{user_content}",
        ),
    ]

    try:
        result = await llm.complete_structured(
            messages=messages,
            schema=TreatmentPlanReview,
            temperature=0.3,
            max_tokens=4096,
        )
        return result
    except Exception as e:
        logger.warning("Structured evidence suggestion failed, trying raw completion: %s", e)
        # Fallback: raw completion + manual parse
        response = await llm.complete(messages=messages, temperature=0.3, max_tokens=4096)
        try:
            data = json.loads(response.content)
            return TreatmentPlanReview(**data)
        except (json.JSONDecodeError, Exception) as parse_err:
            logger.error("Failed to parse evidence suggestions: %s", parse_err)
            return TreatmentPlanReview(
                missing_elements=["Evidence engine returned unparseable response"],
                defensibility_notes=["Manual review recommended"],
            )
