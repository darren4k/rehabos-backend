"""Documentation Agent for generating clinical notes."""

from typing import Optional

from pydantic import BaseModel, Field

from rehab_os.agents.base import AgentContext, BaseAgent
from rehab_os.llm import LLMRouter
from rehab_os.models.evidence import EvidenceSummary
from rehab_os.models.output import (
    ClinicalDocumentation,
    DiagnosisResult,
    DocumentationType,
    OutcomeRecommendations,
    SafetyAssessment,
)
from rehab_os.models.patient import Discipline, PatientContext
from rehab_os.models.plan import PlanOfCare


class DocumentationInput(BaseModel):
    """Input for documentation generation."""

    document_type: DocumentationType
    patient: PatientContext
    safety: Optional[SafetyAssessment] = None
    diagnosis: Optional[DiagnosisResult] = None
    plan: Optional[PlanOfCare] = None
    outcomes: Optional[OutcomeRecommendations] = None
    evidence: Optional[EvidenceSummary] = None

    # For progress/daily notes
    session_notes: Optional[str] = Field(None, description="Notes from today's session")
    interventions_performed: Optional[list[str]] = None
    patient_response: Optional[str] = None

    # For discharge
    discharge_status: Optional[str] = None
    goals_achieved: Optional[list[str]] = None
    recommendations: Optional[str] = None


# Documentation templates by type
TEMPLATES = {
    DocumentationType.INITIAL_EVAL: """
## INITIAL EVALUATION

### PATIENT INFORMATION
{patient_info}

### HISTORY OF PRESENT ILLNESS
{hpi}

### MEDICAL HISTORY
{pmh}

### SOCIAL/FUNCTIONAL HISTORY
{social_history}

### SYSTEMS REVIEW
{systems_review}

### EXAMINATION
{examination}

### CLINICAL IMPRESSION
{impression}

### ASSESSMENT
{assessment}

### PLAN OF CARE
{plan}

### GOALS
{goals}

### FREQUENCY/DURATION
{frequency}

### PROGNOSIS
{prognosis}
""",
    DocumentationType.DAILY_NOTE: """
## DAILY NOTE

### SUBJECTIVE
{subjective}

### OBJECTIVE
{objective}

### ASSESSMENT
{assessment}

### PLAN
{plan}
""",
    DocumentationType.PROGRESS_NOTE: """
## PROGRESS NOTE

### INTERVAL HISTORY
{interval_history}

### SUBJECTIVE
{subjective}

### OBJECTIVE
{objective}

### ASSESSMENT
{assessment}

### PLAN
{plan}

### GOAL STATUS
{goal_status}
""",
    DocumentationType.DISCHARGE_SUMMARY: """
## DISCHARGE SUMMARY

### ADMISSION/EVALUATION DATE
{admit_date}

### DISCHARGE DATE
{dc_date}

### PRIMARY DIAGNOSIS
{diagnosis}

### TREATMENT SUMMARY
{treatment_summary}

### OUTCOMES
{outcomes}

### GOAL ACHIEVEMENT
{goals}

### DISCHARGE STATUS
{dc_status}

### RECOMMENDATIONS
{recommendations}
""",
}


class DocumentationAgent(BaseAgent[DocumentationInput, ClinicalDocumentation]):
    """Agent for generating clinical documentation.

    Produces various note types following discipline standards
    and regulatory requirements.
    """

    def __init__(self, llm: LLMRouter):
        super().__init__(
            llm=llm,
            name="documentation",
            description="Clinical documentation generation",
        )

    @property
    def system_prompt(self) -> str:
        return """You are a clinical documentation specialist for rehabilitation services.

Your role is to generate professional clinical documentation that:
1. Meets regulatory and billing requirements
2. Clearly communicates clinical reasoning
3. Documents skilled care necessity
4. Supports continuity of care

Documentation principles:
- Be specific and objective
- Use measurable, quantifiable data
- Avoid vague or subjective statements
- Document skilled intervention rationale
- Include functional outcomes
- Maintain professional tone

For each note type, include required elements:
- Initial Evaluation: Comprehensive history, exam, assessment, and plan
- Daily Note: SOAP format with specific interventions and responses
- Progress Note: Goal progress, outcome measures, plan updates
- Discharge Summary: Summary of care, outcomes, recommendations

Billing compliance:
- Document skilled services clearly
- Justify medical necessity
- Include time when required
- Reference objective measures

Never include:
- PHI placeholders (use [PATIENT] if needed)
- Subjective judgments about patient character
- Information not supported by exam/history
- Copy-forwarded content without updates"""

    @property
    def output_schema(self) -> type[ClinicalDocumentation]:
        return ClinicalDocumentation

    def format_input(self, inputs: DocumentationInput, context: AgentContext) -> str:
        discipline = inputs.patient.discipline.value
        doc_type = inputs.document_type.value

        sections = [
            f"## Documentation Request",
            f"**Type:** {doc_type}",
            f"**Discipline:** {discipline}",
            f"**Setting:** {inputs.patient.setting.value}",
            "",
        ]

        # Patient info
        sections.extend(
            [
                "### Patient Information",
                f"- Age/Sex: {inputs.patient.age}yo {inputs.patient.sex}",
                f"- Chief Complaint: {inputs.patient.chief_complaint}",
            ]
        )

        if inputs.patient.diagnosis:
            sections.append(f"- Diagnoses: {', '.join(inputs.patient.diagnosis)}")

        if inputs.patient.comorbidities:
            sections.append(f"- Comorbidities: {', '.join(inputs.patient.comorbidities)}")

        # Safety findings
        if inputs.safety:
            sections.extend(
                [
                    "",
                    "### Safety Screening",
                    f"- Safe to treat: {inputs.safety.is_safe_to_treat}",
                ]
            )
            if inputs.safety.red_flags:
                sections.append(f"- Red flags: {len(inputs.safety.red_flags)}")
            if inputs.safety.precautions:
                sections.append(f"- Precautions: {', '.join(inputs.safety.precautions)}")

        # Diagnosis
        if inputs.diagnosis:
            sections.extend(
                [
                    "",
                    "### Clinical Diagnosis",
                    f"- Primary: {inputs.diagnosis.primary_diagnosis}",
                    f"- ICD-10: {', '.join(inputs.diagnosis.icd_codes)}",
                    f"- Rationale: {inputs.diagnosis.rationale}",
                ]
            )

        # Plan
        if inputs.plan:
            sections.extend(
                [
                    "",
                    "### Plan of Care",
                    f"- Clinical Summary: {inputs.plan.clinical_summary}",
                    f"- Prognosis: {inputs.plan.prognosis}",
                    f"- Frequency: {inputs.plan.visit_frequency}",
                    "",
                    "**Goals:**",
                ]
            )
            for goal in inputs.plan.smart_goals[:4]:
                sections.append(f"- {goal.format_goal()}")

            sections.append("\n**Interventions:**")
            for intervention in inputs.plan.interventions[:5]:
                sections.append(f"- {intervention.name}: {intervention.description}")

        # Session-specific info for daily/progress notes
        if inputs.session_notes:
            sections.extend(
                [
                    "",
                    "### Session Information",
                    inputs.session_notes,
                ]
            )

        if inputs.interventions_performed:
            sections.extend(
                [
                    "",
                    "### Interventions Performed",
                ]
            )
            for intervention in inputs.interventions_performed:
                sections.append(f"- {intervention}")

        if inputs.patient_response:
            sections.extend(
                [
                    "",
                    "### Patient Response",
                    inputs.patient_response,
                ]
            )

        # Discharge-specific
        if inputs.discharge_status:
            sections.extend(
                [
                    "",
                    "### Discharge Status",
                    inputs.discharge_status,
                ]
            )

        if inputs.goals_achieved:
            sections.extend(
                [
                    "",
                    "### Goals Achieved",
                ]
            )
            for goal in inputs.goals_achieved:
                sections.append(f"- {goal}")

        # Template reference
        if inputs.document_type in TEMPLATES:
            sections.extend(
                [
                    "",
                    "### Template Structure",
                    f"Follow this structure for {doc_type}:",
                    TEMPLATES[inputs.document_type],
                ]
            )

        sections.extend(
            [
                "",
                "## Task",
                f"Generate a complete {doc_type} note following the template structure.",
                "Include all relevant clinical information and ensure medical necessity is documented.",
            ]
        )

        return "\n".join(sections)

    @property
    def temperature(self) -> float:
        return 0.3

    @property
    def max_tokens(self) -> int:
        return 6000

    @property
    def model_tier(self):
        from rehab_os.agents.base import ModelTier
        return ModelTier.COMPLEX  # Documentation requires detailed, coherent writing
