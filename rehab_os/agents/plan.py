"""Plan Agent for treatment planning and goal setting."""

from typing import Optional

from pydantic import BaseModel, Field

from rehab_os.agents.base import AgentContext, BaseAgent
from rehab_os.llm import LLMRouter
from rehab_os.models.evidence import EvidenceSummary
from rehab_os.models.output import DiagnosisResult
from rehab_os.models.patient import Discipline, PatientContext
from rehab_os.models.plan import PlanOfCare


class PlanInput(BaseModel):
    """Input for treatment planning."""

    patient: PatientContext
    diagnosis: DiagnosisResult
    evidence: Optional[EvidenceSummary] = None
    patient_goals: Optional[list[str]] = Field(
        None, description="Patient-stated goals and priorities"
    )
    constraints: Optional[list[str]] = Field(
        None, description="Constraints (e.g., equipment, time, payer)"
    )
    prior_treatment: Optional[str] = Field(
        None, description="Previous treatments and response"
    )


# CPT code references for common interventions
CPT_REFERENCES = {
    "therapeutic_exercise": ["97110"],
    "neuromuscular_reeducation": ["97112"],
    "manual_therapy": ["97140"],
    "therapeutic_activities": ["97530"],
    "gait_training": ["97116"],
    "self_care_training": ["97535"],
    "cognitive_skills": ["97129", "97130"],
    "swallow_function": ["92526"],
    "speech_treatment": ["92507"],
    "evaluation_pt": ["97161", "97162", "97163"],
    "evaluation_ot": ["97165", "97166", "97167"],
    "evaluation_slp": ["92521", "92522", "92523", "92524"],
}


class PlanAgent(BaseAgent[PlanInput, PlanOfCare]):
    """Agent for generating evidence-based treatment plans.

    Creates comprehensive plans including:
    - SMART goals (short and long-term)
    - Interventions with FITT parameters
    - Home exercise programs
    - Discharge criteria
    """

    def __init__(self, llm: LLMRouter):
        super().__init__(
            llm=llm,
            name="plan",
            description="Treatment planning and goal setting",
        )

    @property
    def system_prompt(self) -> str:
        return """You are a clinical treatment planning agent for rehabilitation services.

Your role is to create comprehensive, evidence-based plans of care including:

## SMART Goals
- Specific: Clear, detailed description of what will be achieved
- Measurable: Objective criteria to measure progress
- Achievable: Realistic given patient's condition and resources
- Relevant: Meaningful to patient's function and quality of life
- Time-bound: Clear timeframe for achievement

Short-term goals: 2-4 weeks
Long-term goals: 6-12 weeks or discharge

## Interventions
For each intervention, provide:
- Clear description and rationale
- FITT parameters when applicable:
  - Frequency: How often (e.g., 2-3x/week)
  - Intensity: How hard (e.g., RPE 4-6/10, 60-70% 1RM)
  - Time: Duration (e.g., 20-30 minutes)
  - Type: Mode of exercise/intervention
- Progression criteria
- Relevant precautions

## Home Exercise Program
- Clear, patient-friendly instructions
- Appropriate dosage (sets, reps, frequency)
- Easier and harder modifications
- Clear precautions and stop criteria

## Clinical Reasoning
- Base recommendations on evidence when available
- Consider patient's setting and resources
- Account for comorbidities and precautions
- Plan for contingencies

## Prognosis
- State expected outcome and timeframe
- Identify factors affecting prognosis (positive and negative)
- Define clear discharge criteria

Be specific and actionable. Avoid vague recommendations."""

    @property
    def output_schema(self) -> type[PlanOfCare]:
        return PlanOfCare

    def format_input(self, inputs: PlanInput, context: AgentContext) -> str:
        discipline = inputs.patient.discipline.value

        sections = [
            f"## Treatment Planning Request - {discipline}",
            "",
            "### Patient Summary",
            f"- {inputs.patient.summary()}",
            f"- Setting: {inputs.patient.setting.value}",
        ]

        if inputs.patient.prior_level_of_function:
            sections.append(f"- Prior level of function: {inputs.patient.prior_level_of_function}")

        if inputs.patient.precautions:
            sections.append(f"- Precautions: {', '.join(inputs.patient.precautions)}")

        # Diagnosis
        sections.extend(
            [
                "",
                "### Diagnosis",
                f"**Primary:** {inputs.diagnosis.primary_diagnosis}",
                f"**ICD-10:** {', '.join(inputs.diagnosis.icd_codes)}",
            ]
        )

        if inputs.diagnosis.classification:
            sections.append(f"**Classification:** {inputs.diagnosis.classification}")

        sections.append(f"\n**Clinical Reasoning:** {inputs.diagnosis.rationale}")

        # Functional status
        if inputs.patient.functional_status:
            fs = inputs.patient.functional_status
            sections.extend(["", "### Baseline Functional Status"])
            if fs.berg_balance is not None:
                sections.append(f"- Berg Balance: {fs.berg_balance}/56")
            if fs.timed_up_and_go is not None:
                sections.append(f"- TUG: {fs.timed_up_and_go}s")
            if fs.gait_speed is not None:
                sections.append(f"- Gait Speed: {fs.gait_speed} m/s")
            if fs.fim_score is not None:
                sections.append(f"- FIM: {fs.fim_score}/126")
            if fs.moca is not None:
                sections.append(f"- MoCA: {fs.moca}/30")
            for name, value in fs.custom_scores.items():
                sections.append(f"- {name}: {value}")

        # Evidence
        if inputs.evidence:
            sections.extend(
                [
                    "",
                    "### Available Evidence",
                    inputs.evidence.synthesis or "No synthesis available",
                ]
            )
            if inputs.evidence.guideline_recommendations:
                sections.append("\n**Guideline Recommendations:**")
                for rec in inputs.evidence.guideline_recommendations[:5]:
                    sections.append(
                        f"- {rec.organization}: {rec.recommendation_text} "
                        f"(Strength: {rec.strength.value})"
                    )

        # Patient goals
        if inputs.patient_goals:
            sections.extend(
                [
                    "",
                    "### Patient Goals",
                ]
            )
            for goal in inputs.patient_goals:
                sections.append(f"- {goal}")

        # Constraints
        if inputs.constraints:
            sections.extend(
                [
                    "",
                    "### Constraints",
                ]
            )
            for constraint in inputs.constraints:
                sections.append(f"- {constraint}")

        # Prior treatment
        if inputs.prior_treatment:
            sections.extend(
                [
                    "",
                    "### Prior Treatment",
                    inputs.prior_treatment,
                ]
            )

        # CPT reference
        sections.extend(
            [
                "",
                "### CPT Code Reference",
            ]
        )
        for intervention, codes in CPT_REFERENCES.items():
            sections.append(f"- {intervention}: {', '.join(codes)}")

        sections.extend(
            [
                "",
                "## Task",
                "Create a comprehensive plan of care including:",
                "1. Clinical summary and impression",
                "2. Prognosis and rehab potential",
                "3. SMART goals (2-3 short-term, 1-2 long-term)",
                "4. Interventions with FITT parameters and rationale",
                "5. Visit frequency and expected duration",
                "6. Home exercise program (3-5 exercises)",
                "7. Patient education topics",
                "8. Precautions and contingency plans",
                "9. Discharge criteria",
            ]
        )

        return "\n".join(sections)

    @property
    def temperature(self) -> float:
        return 0.4  # Slightly higher for more creative plan generation

    @property
    def max_tokens(self) -> int:
        return 8000  # Plans can be detailed

    @property
    def model_tier(self):
        from rehab_os.agents.base import ModelTier
        return ModelTier.COMPLEX  # Treatment planning requires sophisticated reasoning
