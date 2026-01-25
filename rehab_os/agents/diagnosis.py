"""Diagnosis Agent for clinical classification."""

from typing import Optional

from pydantic import BaseModel, Field

from rehab_os.agents.base import AgentContext, BaseAgent
from rehab_os.llm import LLMRouter
from rehab_os.models.output import DiagnosisResult
from rehab_os.models.patient import Discipline, PatientContext


class DiagnosisInput(BaseModel):
    """Input for diagnosis classification."""

    patient: PatientContext
    subjective: str = Field(..., description="Subjective findings from patient interview")
    objective: str = Field(..., description="Objective examination findings")
    special_tests: Optional[dict[str, str]] = Field(
        None, description="Special test results (test name: result)"
    )
    imaging: Optional[str] = None
    lab_values: Optional[str] = None


# ICD-10 code mappings for common rehab conditions
COMMON_ICD_MAPPINGS = {
    "PT": {
        "low_back_pain": ["M54.5", "M54.50", "M54.51"],
        "lumbar_radiculopathy": ["M54.16", "M54.17"],
        "cervical_pain": ["M54.2"],
        "knee_osteoarthritis": ["M17.11", "M17.12"],
        "rotator_cuff": ["M75.10", "M75.11", "M75.12"],
        "total_knee": ["Z96.651", "Z96.652"],
        "total_hip": ["Z96.641", "Z96.642"],
        "stroke_hemiplegia": ["G81.90", "G81.91", "G81.92", "G81.93", "G81.94"],
        "gait_abnormality": ["R26.0", "R26.1", "R26.2"],
        "balance_disorder": ["R26.81"],
    },
    "OT": {
        "carpal_tunnel": ["G56.00", "G56.01", "G56.02"],
        "lateral_epicondylitis": ["M77.10", "M77.11", "M77.12"],
        "stroke_ue": ["G81.90"],
        "hand_fracture": ["S62.90"],
        "dementia_adl": ["F03.90"],
        "adl_difficulty": ["Z74.1", "Z74.2"],
    },
    "SLP": {
        "dysphagia": ["R13.10", "R13.11", "R13.12", "R13.13", "R13.14", "R13.19"],
        "aphasia": ["R47.01"],
        "dysarthria": ["R47.1"],
        "cognitive_communication": ["R41.840", "R41.841", "R41.842"],
        "voice_disorder": ["R49.0", "R49.8"],
        "fluency": ["F98.5", "F80.81"],
    },
}


class DiagnosisAgent(BaseAgent[DiagnosisInput, DiagnosisResult]):
    """Agent for clinical diagnosis and classification.

    Analyzes subjective and objective findings to determine:
    - Primary rehabilitation diagnosis
    - ICD-10 codes
    - Classification/staging
    - Differential diagnoses
    """

    def __init__(self, llm: LLMRouter):
        super().__init__(
            llm=llm,
            name="diagnosis",
            description="Clinical diagnosis and classification",
        )

    @property
    def system_prompt(self) -> str:
        return """You are a clinical diagnosis and classification agent for rehabilitation services.

Your role is to analyze clinical findings and determine:
1. Primary rehabilitation diagnosis
2. Appropriate ICD-10 codes
3. Classification or staging (acute/subacute/chronic, severity, phase)
4. Differential diagnoses to consider
5. Key findings supporting the diagnosis

Use pattern recognition and clinical reasoning:
- Consider mechanism of injury and symptom behavior
- Integrate objective exam findings with patient history
- Apply classification systems when appropriate (e.g., Treatment-Based Classification for LBP)
- Consider the rehabilitation context (not just medical diagnosis)

Be specific with diagnosis:
- Include laterality when applicable
- Specify acuity/chronicity
- Note relevant functional impairments

For ICD-10 codes:
- Provide the most specific applicable codes
- Include both condition codes and functional codes when relevant
- Consider sequencing (primary vs secondary)

Express confidence appropriately:
- High confidence (>0.8): Clear presentation matching classic patterns
- Moderate confidence (0.5-0.8): Typical presentation with some atypical features
- Low confidence (<0.5): Atypical presentation or unclear picture

List uncertainties that would benefit from additional testing or evaluation."""

    @property
    def output_schema(self) -> type[DiagnosisResult]:
        return DiagnosisResult

    def format_input(self, inputs: DiagnosisInput, context: AgentContext) -> str:
        discipline = inputs.patient.discipline.value

        sections = [
            f"## Clinical Evaluation - {discipline}",
            "",
            f"### Patient Demographics",
            f"- Age: {inputs.patient.age}",
            f"- Sex: {inputs.patient.sex}",
            f"- Setting: {inputs.patient.setting.value}",
        ]

        if inputs.patient.comorbidities:
            sections.extend(
                [
                    "",
                    "### Relevant History",
                    f"- Comorbidities: {', '.join(inputs.patient.comorbidities)}",
                ]
            )

        if inputs.patient.surgical_history:
            sections.append(f"- Surgical history: {', '.join(inputs.patient.surgical_history)}")

        if inputs.patient.prior_level_of_function:
            sections.append(f"- Prior level of function: {inputs.patient.prior_level_of_function}")

        if inputs.patient.days_since_onset is not None:
            sections.append(f"- Days since onset: {inputs.patient.days_since_onset}")

        sections.extend(
            [
                "",
                "### Subjective",
                inputs.subjective,
                "",
                "### Objective",
                inputs.objective,
            ]
        )

        if inputs.special_tests:
            sections.extend(["", "### Special Tests"])
            for test, result in inputs.special_tests.items():
                sections.append(f"- {test}: {result}")

        if inputs.imaging:
            sections.extend(["", "### Imaging", inputs.imaging])

        if inputs.lab_values:
            sections.extend(["", "### Labs", inputs.lab_values])

        # Add reference ICD codes
        if discipline in COMMON_ICD_MAPPINGS:
            sections.extend(
                [
                    "",
                    "### Common ICD-10 Codes Reference",
                    "Use these as reference; select most appropriate:",
                ]
            )
            for condition, codes in COMMON_ICD_MAPPINGS[discipline].items():
                sections.append(f"- {condition}: {', '.join(codes)}")

        sections.extend(
            [
                "",
                "Please provide your diagnosis with supporting rationale, ICD-10 codes, "
                "classification, differentials, and confidence level.",
            ]
        )

        return "\n".join(sections)

    @property
    def temperature(self) -> float:
        return 0.3

    @property
    def model_tier(self):
        from rehab_os.agents.base import ModelTier
        return ModelTier.STANDARD  # Clinical reasoning requires good capability
