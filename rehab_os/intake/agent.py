"""Intake agent for structuring referral documents into PatientContext profiles."""

import logging
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from rehab_os.agents.base import AgentContext, BaseAgent, ModelTier
from rehab_os.models.patient import PatientContext

logger = logging.getLogger(__name__)


class IntakeInput(BaseModel):
    """Input for the intake agent."""

    raw_text: str = Field(..., description="Extracted document text")
    source_type: str = Field(
        default="referral",
        description="Document type: referral, prescription, transfer_summary, prior_auth",
    )
    referring_provider: Optional[str] = None
    received_date: Optional[date] = None


class IntakeResult(BaseModel):
    """Structured result from intake processing."""

    patient: PatientContext = Field(..., description="Structured patient profile")
    referral_summary: str = Field(..., description="Plain-language summary of the referral")
    referring_provider: Optional[str] = None
    referring_diagnosis: list[str] = Field(default_factory=list)
    icd_codes_extracted: list[str] = Field(default_factory=list)
    insurance_info: Optional[dict] = Field(
        default=None,
        description="Payer, member_id, group_id, auth_number if found",
    )
    visit_authorization: Optional[dict] = Field(
        default=None,
        description="authorized_visits, frequency, duration, expiry_date",
    )
    extraction_confidence: float = Field(
        ..., ge=0, le=1, description="0-1 confidence in extraction quality"
    )
    missing_fields: list[str] = Field(
        default_factory=list, description="Fields that couldn't be extracted"
    )
    raw_text_snippet: str = Field(
        ..., description="First 500 chars of source text for audit trail"
    )


INTAKE_SYSTEM_PROMPT = """\
You are a clinical intake specialist for a rehabilitation therapy practice. Your job is to \
extract structured patient information from referral documents, prescriptions, and transfer \
summaries.

You MUST return a JSON object matching the IntakeResult schema exactly.

## Extraction Rules

### Demographics
- Extract patient name, DOB, sex/gender, age
- If DOB is given but age is not, calculate age from DOB (assume current date is the \
received_date or today)
- Sex must be one of: "male", "female", "other"
- If sex is not explicitly stated, infer from context clues (e.g., pronouns) or mark as missing

### Diagnoses & ICD Codes
- Extract all diagnoses mentioned in the document
- If ICD-10 codes are explicitly listed, extract them verbatim
- If diagnoses are mentioned without codes, attempt to map to standard ICD-10 codes
- List the primary/referring diagnosis first

### Discipline Identification
- Determine the rehabilitation discipline from referral language:
  - PT (Physical Therapy): ROM, strengthening, gait, balance, mobility, weight bearing
  - OT (Occupational Therapy): ADLs, fine motor, upper extremity function, splinting, cognition
  - SLP (Speech-Language Pathology): swallowing, dysphagia, aphasia, voice, cognition/communication
- Default to PT if unclear

### Clinical Information
- Medications: list all with dosages if provided
- Allergies: list all mentioned
- Surgical history: list procedures with dates if available
- Precautions: weight bearing status, activity restrictions, medical precautions
- Comorbidities: PMH items (HTN, DM, etc.)
- Prior level of function: functional baseline before current episode

### Insurance & Authorization
- Extract payer name, member ID, group number
- Authorization number, authorized visits, frequency, duration
- If any auth info is present, populate visit_authorization dict

### Care Setting
- Map to: inpatient, outpatient, home_health, snf, irf, acute, telehealth
- Default to outpatient if not specified

### Chief Complaint
- Synthesize a concise chief complaint from the referral reason and diagnosis

### Confidence Scoring
- Be CONSERVATIVE with confidence scores
- 0.9-1.0: All key fields clearly stated and extracted
- 0.7-0.89: Most fields present, minor gaps or ambiguities
- 0.5-0.69: Significant missing data or unclear information
- Below 0.5: Major gaps, poor document quality, or conflicting information

### Missing Fields
- Track every expected field that could not be extracted
- Common missing fields: height_cm, weight_kg, vitals, functional_status scores

### Output Format
Return valid JSON matching the IntakeResult schema. For the patient field, use the \
PatientContext schema with all available fields populated.\
"""


class IntakeAgent(BaseAgent[IntakeInput, IntakeResult]):
    """Agent that structures referral documents into PatientContext profiles."""

    def __init__(self, llm):
        super().__init__(
            llm=llm,
            name="intake",
            description="Extracts structured patient profiles from referral documents",
        )

    @property
    def system_prompt(self) -> str:
        return INTAKE_SYSTEM_PROMPT

    @property
    def output_schema(self) -> type[IntakeResult]:
        return IntakeResult

    def format_input(self, inputs: IntakeInput, context: AgentContext) -> str:
        parts = [
            f"## Document Type: {inputs.source_type}",
        ]
        if inputs.referring_provider:
            parts.append(f"## Referring Provider: {inputs.referring_provider}")
        if inputs.received_date:
            parts.append(f"## Received Date: {inputs.received_date.isoformat()}")

        parts.append(f"\n## Document Text:\n{inputs.raw_text}")

        return "\n".join(parts)

    def _generate_input_summary(self, inputs: IntakeInput) -> str:
        return f"{inputs.source_type}: {inputs.raw_text[:80]}..."

    def _generate_output_summary(self, output: IntakeResult) -> str:
        return (
            f"Patient: {output.patient.summary()}, "
            f"confidence: {output.extraction_confidence:.0%}, "
            f"missing: {len(output.missing_fields)}"
        )

    @property
    def temperature(self) -> float:
        return 0.2

    @property
    def model_tier(self) -> ModelTier:
        return ModelTier.STANDARD
