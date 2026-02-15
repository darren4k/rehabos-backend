"""Medicare & Joint Commission Defensibility Layer.

Checks clinical notes against Medicare documentation requirements and
Joint Commission standards, flagging gaps that could lead to denials.
"""

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from rehab_os.llm import LLMRouter, Message, MessageRole

logger = logging.getLogger(__name__)


class MedicareRequirement(BaseModel):
    """Single Medicare requirement check."""
    requirement: str
    met: bool
    evidence_in_note: Optional[str] = None
    fix_suggestion: Optional[str] = None


class DefensibilityCheck(BaseModel):
    """Full defensibility assessment of a clinical note."""
    overall_score: float = Field(ge=0.0, le=100.0)
    category_scores: dict[str, float] = Field(default_factory=dict)
    passed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    medicare_requirements: list[MedicareRequirement] = Field(default_factory=list)


DEFENSIBILITY_SYSTEM_PROMPT = """You are a Medicare compliance auditor and Joint Commission surveyor
specializing in rehabilitation therapy documentation (PT/OT/SLP).

Analyze the provided clinical note and return a JSON defensibility assessment.

Your response MUST be valid JSON matching this schema:
{
  "overall_score": 0-100,
  "category_scores": {
    "medical_necessity": 0-100,
    "skilled_care_justification": 0-100,
    "functional_outcomes": 0-100,
    "progress_documentation": 0-100,
    "plan_of_care": 0-100,
    "safety_documentation": 0-100,
    "discharge_planning": 0-100
  },
  "passed": ["Requirement that is well-documented"],
  "warnings": ["Thing that could be questioned by a reviewer"],
  "failures": ["Thing that WILL likely result in denial or flag"],
  "recommendations": ["Specific fix to strengthen the note"],
  "medicare_requirements": [
    {
      "requirement": "Skilled care justification",
      "met": true/false,
      "evidence_in_note": "Quote from note that satisfies this (or null)",
      "fix_suggestion": "How to fix if not met (or null)"
    }
  ]
}

MEDICARE REQUIREMENTS — CHECK ALL:
1. Skilled care justification: Why does this require a licensed therapist? Not an aide or caregiver.
2. Medical necessity: Why is therapy needed? Must link to functional deficits.
3. Reasonable and necessary: Treatment is appropriate for the condition.
4. Progress documentation: Measurable improvement or justified plateau with skilled maintenance rationale.
5. Functional goals: Goals tied to functional outcomes, not just impairment-level measures.
6. Prior Level of Function (PLOF): Documented for comparison to current status.
7. Complexity of condition: Why skilled services are needed (not routine/repetitive).
8. Patient response: How the patient responded to treatment this session.
9. Plan of care: Specific, measurable, with frequency/duration.
10. Discharge plan: When and how therapy will end.

JOINT COMMISSION STANDARDS — CHECK:
11. Assessment completeness (all relevant domains assessed)
12. Plan individualized to patient (not generic/template language)
13. Patient/family education documented
14. Coordination of care documented (communication with MD, other disciplines)
15. Safety precautions documented
16. Fall risk assessment (if applicable)
17. Pain assessment
18. Discharge planning from admission

COMMON DENIAL RED FLAGS — CHECK FOR:
- "Maintenance therapy" language without "skilled maintenance" justification
- Lack of measurable progress between sessions/weeks
- Goals stated as impairment-only (e.g., "increase ROM") without functional link
- No skilled rationale for interventions used
- Template/copy-paste language suggesting no individualization
- Missing or vague frequency/duration
- No patient response to treatment documented

Score conservatively. A note that would survive a Medicare audit should score 80+.
A note with denial-risk issues should score below 60."""


async def check_defensibility(
    note_content: dict,
    note_type: str,
    structured_data: dict,
    patient_context: dict,
    llm: LLMRouter,
) -> DefensibilityCheck:
    """Check a clinical note for Medicare and Joint Commission defensibility.

    Args:
        note_content: SOAP sections dict (subjective, objective, assessment, plan)
        note_type: evaluation, daily_note, progress_note, discharge
        structured_data: ROM, MMT, standardized tests, etc.
        patient_context: age, setting, comorbidities, discipline

    Returns:
        DefensibilityCheck with scores, passes, warnings, failures, and recommendations
    """
    user_content = json.dumps({
        "note_content": note_content,
        "note_type": note_type,
        "structured_data": structured_data,
        "patient_context": patient_context,
    }, indent=2)

    messages = [
        Message(role=MessageRole.SYSTEM, content=DEFENSIBILITY_SYSTEM_PROMPT),
        Message(
            role=MessageRole.USER,
            content=f"Audit this clinical note for defensibility:\n\n{user_content}",
        ),
    ]

    try:
        result = await llm.complete_structured(
            messages=messages,
            schema=DefensibilityCheck,
            temperature=0.2,
            max_tokens=4096,
        )
        return result
    except Exception as e:
        logger.warning("Structured defensibility check failed, trying raw: %s", e)
        response = await llm.complete(messages=messages, temperature=0.2, max_tokens=4096)
        try:
            data = json.loads(response.content)
            return DefensibilityCheck(**data)
        except (json.JSONDecodeError, Exception) as parse_err:
            logger.error("Failed to parse defensibility check: %s", parse_err)
            return DefensibilityCheck(
                overall_score=0.0,
                failures=["Defensibility engine returned unparseable response"],
                recommendations=["Manual review required"],
            )
