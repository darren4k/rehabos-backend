"""Outcome Agent for recommending outcome measures."""

from typing import Optional

from pydantic import BaseModel, Field

from rehab_os.agents.base import AgentContext, BaseAgent
from rehab_os.llm import LLMRouter
from rehab_os.models.output import DiagnosisResult, OutcomeRecommendations
from rehab_os.models.patient import Discipline, PatientContext, CareSetting


class OutcomeInput(BaseModel):
    """Input for outcome measure recommendations."""

    patient: PatientContext
    diagnosis: DiagnosisResult
    goals: Optional[list[str]] = Field(None, description="Treatment goals to measure")


# Discipline-specific outcome measures database
OUTCOME_MEASURES = {
    "PT": {
        "balance": [
            {
                "name": "Berg Balance Scale",
                "abbrev": "BBS",
                "mcid": "4-7 points",
                "mdc": "5 points",
                "time": "15-20 min",
            },
            {
                "name": "Timed Up and Go",
                "abbrev": "TUG",
                "mcid": "2.9 sec",
                "mdc": "2.9 sec",
                "time": "2 min",
            },
            {
                "name": "Dynamic Gait Index",
                "abbrev": "DGI",
                "mcid": "4 points",
                "mdc": "2.9 points",
                "time": "10 min",
            },
        ],
        "mobility": [
            {
                "name": "6-Minute Walk Test",
                "abbrev": "6MWT",
                "mcid": "50m",
                "mdc": "82m",
                "time": "10 min",
            },
            {
                "name": "10-Meter Walk Test",
                "abbrev": "10MWT",
                "mcid": "0.1 m/s",
                "mdc": "0.1 m/s",
                "time": "5 min",
            },
            {
                "name": "Functional Gait Assessment",
                "abbrev": "FGA",
                "mcid": "4 points",
                "mdc": "4.2 points",
                "time": "10 min",
            },
        ],
        "lbp": [
            {
                "name": "Oswestry Disability Index",
                "abbrev": "ODI",
                "mcid": "10%",
                "mdc": "10%",
                "time": "5 min",
            },
            {
                "name": "Roland-Morris Disability Questionnaire",
                "abbrev": "RMDQ",
                "mcid": "5 points",
                "mdc": "5 points",
                "time": "5 min",
            },
        ],
        "neck": [
            {
                "name": "Neck Disability Index",
                "abbrev": "NDI",
                "mcid": "7 points",
                "mdc": "5 points",
                "time": "5 min",
            },
        ],
        "upper_extremity": [
            {
                "name": "DASH",
                "abbrev": "DASH",
                "mcid": "10 points",
                "mdc": "10 points",
                "time": "10 min",
            },
            {
                "name": "QuickDASH",
                "abbrev": "QuickDASH",
                "mcid": "8 points",
                "mdc": "8 points",
                "time": "3 min",
            },
        ],
        "lower_extremity": [
            {
                "name": "Lower Extremity Functional Scale",
                "abbrev": "LEFS",
                "mcid": "9 points",
                "mdc": "9 points",
                "time": "3 min",
            },
        ],
        "general": [
            {
                "name": "Patient-Specific Functional Scale",
                "abbrev": "PSFS",
                "mcid": "2 points",
                "mdc": "2 points",
                "time": "3 min",
            },
            {
                "name": "Global Rating of Change",
                "abbrev": "GROC",
                "mcid": "N/A",
                "mdc": "N/A",
                "time": "1 min",
            },
        ],
    },
    "OT": {
        "hand_function": [
            {
                "name": "Nine-Hole Peg Test",
                "abbrev": "9HPT",
                "mcid": "varies",
                "mdc": "varies",
                "time": "5 min",
            },
            {
                "name": "Grip Strength (Dynamometer)",
                "abbrev": "Grip",
                "mcid": "6 kg",
                "mdc": "5 kg",
                "time": "3 min",
            },
            {
                "name": "Jebsen-Taylor Hand Function Test",
                "abbrev": "JTHFT",
                "mcid": "varies",
                "mdc": "varies",
                "time": "15 min",
            },
        ],
        "adl": [
            {
                "name": "Functional Independence Measure",
                "abbrev": "FIM",
                "mcid": "22 points",
                "mdc": "22 points",
                "time": "30 min",
            },
            {
                "name": "Barthel Index",
                "abbrev": "BI",
                "mcid": "4 points",
                "mdc": "2 points",
                "time": "10 min",
            },
            {
                "name": "Canadian Occupational Performance Measure",
                "abbrev": "COPM",
                "mcid": "2 points",
                "mdc": "2 points",
                "time": "20 min",
            },
        ],
        "cognition": [
            {
                "name": "Montreal Cognitive Assessment",
                "abbrev": "MoCA",
                "mcid": "varies",
                "mdc": "4 points",
                "time": "10 min",
            },
            {
                "name": "Executive Function Performance Test",
                "abbrev": "EFPT",
                "mcid": "varies",
                "mdc": "varies",
                "time": "30-45 min",
            },
        ],
    },
    "SLP": {
        "swallow": [
            {
                "name": "Functional Oral Intake Scale",
                "abbrev": "FOIS",
                "mcid": "1 level",
                "mdc": "1 level",
                "time": "5 min",
            },
            {
                "name": "Eating Assessment Tool-10",
                "abbrev": "EAT-10",
                "mcid": "3 points",
                "mdc": "varies",
                "time": "3 min",
            },
            {
                "name": "Penetration-Aspiration Scale",
                "abbrev": "PAS",
                "mcid": "1 level",
                "mdc": "N/A",
                "time": "instrumental",
            },
        ],
        "cognition_communication": [
            {
                "name": "Cognitive-Communication Checklist for Acquired Brain Injury",
                "abbrev": "CCCABI",
                "mcid": "varies",
                "mdc": "varies",
                "time": "15 min",
            },
            {
                "name": "Functional Assessment of Verbal Reasoning and Executive Strategies",
                "abbrev": "FAVRES",
                "mcid": "varies",
                "mdc": "varies",
                "time": "45 min",
            },
        ],
        "aphasia": [
            {
                "name": "Western Aphasia Battery-Revised",
                "abbrev": "WAB-R",
                "mcid": "varies",
                "mdc": "varies",
                "time": "30-60 min",
            },
            {
                "name": "Boston Naming Test",
                "abbrev": "BNT",
                "mcid": "varies",
                "mdc": "varies",
                "time": "10 min",
            },
        ],
        "voice": [
            {
                "name": "Voice Handicap Index",
                "abbrev": "VHI",
                "mcid": "18 points",
                "mdc": "varies",
                "time": "5 min",
            },
            {
                "name": "CAPE-V",
                "abbrev": "CAPE-V",
                "mcid": "varies",
                "mdc": "varies",
                "time": "5 min",
            },
        ],
    },
}


class OutcomeAgent(BaseAgent[OutcomeInput, OutcomeRecommendations]):
    """Agent for recommending appropriate outcome measures.

    Selects discipline-specific, condition-appropriate outcome measures
    with information on MCID, MDC, and reassessment timing.
    """

    def __init__(self, llm: LLMRouter):
        super().__init__(
            llm=llm,
            name="outcome",
            description="Outcome measure selection and scheduling",
        )

    @property
    def system_prompt(self) -> str:
        return """You are a clinical outcomes measurement specialist for rehabilitation services.

Your role is to recommend appropriate standardized outcome measures based on:
1. The patient's diagnosis and functional impairments
2. The discipline (PT/OT/SLP)
3. The care setting
4. Treatment goals

For each recommended measure, provide:
- Full name and abbreviation
- Why this measure is appropriate
- MCID (Minimal Clinically Important Difference) - smallest change that matters to patients
- MDC (Minimal Detectable Change) - smallest change beyond measurement error
- Administration time
- Recommended frequency of reassessment

Select measures that are:
- Valid and reliable for the patient population
- Feasible in the care setting
- Sensitive to expected change
- Meaningful for documenting progress

Recommend:
- 1-2 PRIMARY measures (must track)
- 1-2 SECONDARY measures (nice to have)
- Appropriate reassessment schedule based on expected recovery trajectory

Consider setting constraints:
- Acute care: Quick, bedside measures
- Outpatient: Can use longer assessments
- Home health: Portable, minimal equipment
- SNF: Balance functional and standardized measures"""

    @property
    def output_schema(self) -> type[OutcomeRecommendations]:
        return OutcomeRecommendations

    def format_input(self, inputs: OutcomeInput, context: AgentContext) -> str:
        discipline = inputs.patient.discipline.value

        sections = [
            f"## Outcome Measure Recommendation Request - {discipline}",
            "",
            "### Patient Context",
            f"- Age: {inputs.patient.age}",
            f"- Setting: {inputs.patient.setting.value}",
            f"- Diagnosis: {inputs.diagnosis.primary_diagnosis}",
        ]

        if inputs.diagnosis.classification:
            sections.append(f"- Classification: {inputs.diagnosis.classification}")

        if inputs.patient.comorbidities:
            sections.append(f"- Comorbidities: {', '.join(inputs.patient.comorbidities[:5])}")

        if inputs.goals:
            sections.extend(
                [
                    "",
                    "### Treatment Goals",
                ]
            )
            for goal in inputs.goals:
                sections.append(f"- {goal}")

        # Baseline scores if available
        if inputs.patient.functional_status:
            fs = inputs.patient.functional_status
            sections.extend(["", "### Current Baseline Scores"])
            if fs.berg_balance is not None:
                sections.append(f"- Berg Balance: {fs.berg_balance}/56")
            if fs.timed_up_and_go is not None:
                sections.append(f"- TUG: {fs.timed_up_and_go}s")
            if fs.gait_speed is not None:
                sections.append(f"- Gait Speed: {fs.gait_speed} m/s")
            if fs.fim_score is not None:
                sections.append(f"- FIM: {fs.fim_score}/126")
            for name, value in fs.custom_scores.items():
                sections.append(f"- {name}: {value}")

        # Available measures reference
        if discipline in OUTCOME_MEASURES:
            sections.extend(
                [
                    "",
                    f"### Available {discipline} Outcome Measures Reference",
                ]
            )
            for category, measures in OUTCOME_MEASURES[discipline].items():
                sections.append(f"\n**{category.replace('_', ' ').title()}:**")
                for m in measures:
                    sections.append(
                        f"- {m['name']} ({m['abbrev']}): "
                        f"MCID={m['mcid']}, MDC={m['mdc']}, Time={m['time']}"
                    )

        sections.extend(
            [
                "",
                "## Task",
                "Recommend primary and secondary outcome measures with:",
                "1. Rationale for each selection",
                "2. MCID/MDC values for interpreting change",
                "3. Recommended reassessment schedule",
            ]
        )

        return "\n".join(sections)

    @property
    def temperature(self) -> float:
        return 0.3

    @property
    def model_tier(self):
        from rehab_os.agents.base import ModelTier
        return ModelTier.FAST  # Outcome selection is relatively straightforward
