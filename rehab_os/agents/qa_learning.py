"""QA/Learning Agent for quality assurance and critique."""

from typing import Optional

from pydantic import BaseModel, Field

from rehab_os.agents.base import AgentContext, BaseAgent
from rehab_os.llm import LLMRouter
from rehab_os.models.evidence import EvidenceSummary
from rehab_os.models.output import DiagnosisResult, QAResult, SafetyAssessment
from rehab_os.models.patient import PatientContext
from rehab_os.models.plan import PlanOfCare


class QAInput(BaseModel):
    """Input for QA review."""

    patient: PatientContext
    safety: Optional[SafetyAssessment] = None
    diagnosis: Optional[DiagnosisResult] = None
    plan: Optional[PlanOfCare] = None
    evidence: Optional[EvidenceSummary] = None


class QALearningAgent(BaseAgent[QAInput, QAResult]):
    """Agent for quality assurance and meta-review.

    Reviews the outputs of other agents for:
    - Clinical accuracy
    - Evidence alignment
    - Logical consistency
    - Completeness
    - Areas of uncertainty
    """

    def __init__(self, llm: LLMRouter):
        super().__init__(
            llm=llm,
            name="qa_learning",
            description="Quality assurance and clinical critique",
        )

    @property
    def system_prompt(self) -> str:
        return """You are a clinical quality assurance and peer review agent.

Your role is to critically review clinical reasoning and plans for:

## Quality Dimensions

1. **Clinical Accuracy**
   - Is the diagnosis supported by findings?
   - Are interventions appropriate for the condition?
   - Are there any clinical errors or oversights?

2. **Evidence Alignment**
   - Do recommendations align with available evidence?
   - Are there evidence-based alternatives not considered?
   - Are evidence limitations acknowledged?

3. **Logical Consistency**
   - Do goals match the diagnosis?
   - Do interventions support goals?
   - Is the timeline realistic?

4. **Completeness**
   - Are all key areas addressed?
   - Are there missing assessments or interventions?
   - Is the plan comprehensive for the setting?

5. **Safety**
   - Were red flags appropriately addressed?
   - Are precautions adequate?
   - Are there unaddressed risks?

## Review Output

Provide:
- Overall quality score (0-1)
- Specific strengths (what was done well)
- Suggestions (improvements to consider)
- Concerns (potential problems to address)
- Uncertainty flags (areas needing more info)
- Evidence gaps (where more research is needed)
- Alternative approaches (other valid options)

Be constructive and specific. Don't just identify problems—suggest solutions.
Be balanced—acknowledge strengths while noting areas for improvement.
Be practical—focus on clinically significant issues."""

    @property
    def output_schema(self) -> type[QAResult]:
        return QAResult

    def format_input(self, inputs: QAInput, context: AgentContext) -> str:
        sections = [
            "## Quality Assurance Review Request",
            "",
            f"### Patient Context",
            f"- {inputs.patient.summary()}",
            f"- Setting: {inputs.patient.setting.value}",
            f"- Discipline: {inputs.patient.discipline.value}",
            "",
        ]

        # Safety assessment
        if inputs.safety:
            sections.extend(
                [
                    "### Safety Assessment (to review)",
                    f"- Safe to treat: {inputs.safety.is_safe_to_treat}",
                    f"- Red flags identified: {len(inputs.safety.red_flags)}",
                    f"- Urgency level: {inputs.safety.urgency_level.value}",
                    f"- Summary: {inputs.safety.summary}",
                    "",
                ]
            )
            if inputs.safety.red_flags:
                sections.append("**Red Flags:**")
                for rf in inputs.safety.red_flags:
                    sections.append(f"- {rf.finding}: {rf.description}")
                sections.append("")

        # Diagnosis
        if inputs.diagnosis:
            sections.extend(
                [
                    "### Diagnosis (to review)",
                    f"- Primary: {inputs.diagnosis.primary_diagnosis}",
                    f"- ICD-10: {', '.join(inputs.diagnosis.icd_codes)}",
                    f"- Classification: {inputs.diagnosis.classification or 'N/A'}",
                    f"- Confidence: {inputs.diagnosis.confidence}",
                    f"- Rationale: {inputs.diagnosis.rationale}",
                    "",
                ]
            )
            if inputs.diagnosis.differential_diagnoses:
                sections.append(
                    f"- Differentials: {', '.join(inputs.diagnosis.differential_diagnoses)}"
                )
            if inputs.diagnosis.uncertainties:
                sections.append(f"- Stated uncertainties: {', '.join(inputs.diagnosis.uncertainties)}")
            sections.append("")

        # Plan
        if inputs.plan:
            sections.extend(
                [
                    "### Plan of Care (to review)",
                    f"- Clinical Summary: {inputs.plan.clinical_summary}",
                    f"- Prognosis: {inputs.plan.prognosis}",
                    f"- Rehab Potential: {inputs.plan.rehab_potential}",
                    f"- Visit Frequency: {inputs.plan.visit_frequency}",
                    "",
                    "**Goals:**",
                ]
            )
            for goal in inputs.plan.smart_goals:
                sections.append(f"- [{goal.timeframe.value}] {goal.description}")

            sections.append("\n**Interventions:**")
            for intervention in inputs.plan.interventions:
                sections.append(f"- {intervention.name}: {intervention.rationale}")

            if inputs.plan.precautions:
                sections.append(f"\n**Precautions:** {', '.join(inputs.plan.precautions)}")

            if inputs.plan.home_program:
                sections.append(f"\n**HEP:** {len(inputs.plan.home_program)} exercises")

            sections.append("")

        # Evidence
        if inputs.evidence:
            sections.extend(
                [
                    "### Evidence Summary (used for plan)",
                    f"- Sources reviewed: {inputs.evidence.total_sources}",
                    f"- Confidence: {inputs.evidence.confidence or 'N/A'}",
                ]
            )
            if inputs.evidence.synthesis:
                sections.append(f"- Synthesis: {inputs.evidence.synthesis[:500]}...")
            if inputs.evidence.limitations:
                sections.append(f"- Limitations: {inputs.evidence.limitations}")
            sections.append("")

        sections.extend(
            [
                "## Task",
                "Provide a comprehensive quality review covering:",
                "1. Overall quality score (0-1)",
                "2. Strengths of the clinical reasoning",
                "3. Specific suggestions for improvement",
                "4. Concerns that should be addressed",
                "5. Areas of uncertainty requiring clarification",
                "6. Evidence gaps where more research would help",
                "7. Alternative approaches to consider",
            ]
        )

        return "\n".join(sections)

    @property
    def temperature(self) -> float:
        return 0.4  # Slightly higher for thoughtful critique

    @property
    def max_tokens(self) -> int:
        return 4000

    @property
    def model_tier(self):
        from rehab_os.agents.base import ModelTier
        return ModelTier.COMPLEX  # QA review requires sophisticated meta-reasoning
