"""Red Flag Agent for safety screening and triage."""

from typing import Optional

from pydantic import BaseModel, Field

from rehab_os.agents.base import AgentContext, BaseAgent, RuleBasedMixin
from rehab_os.llm import LLMRouter
from rehab_os.models.output import RedFlag, SafetyAssessment, UrgencyLevel
from rehab_os.models.patient import Discipline, PatientContext


class RedFlagInput(BaseModel):
    """Input for red flag screening."""

    patient: PatientContext
    chief_complaint: str
    subjective_report: Optional[str] = None
    objective_findings: Optional[str] = None


# Discipline-specific red flags (rule-based)
PT_RED_FLAGS = {
    "cauda_equina": {
        "keywords": [
            "saddle anesthesia",
            "bladder dysfunction",
            "bowel incontinence",
            "bilateral leg weakness",
            "urinary retention",
        ],
        "urgency": UrgencyLevel.EMERGENT,
        "action": "Immediate referral to emergency department for urgent MRI and surgical evaluation",
    },
    "cervical_myelopathy": {
        "keywords": [
            "bilateral hand numbness",
            "gait ataxia",
            "hyperreflexia",
            "hoffmann sign",
            "clonus",
            "lhermitte",
        ],
        "urgency": UrgencyLevel.URGENT,
        "action": "Urgent referral to spine specialist for evaluation",
    },
    "fracture": {
        "keywords": [
            "severe trauma",
            "osteoporosis",
            "steroid use",
            "point tenderness",
            "unable to bear weight",
        ],
        "urgency": UrgencyLevel.URGENT,
        "action": "Imaging to rule out fracture before treatment",
    },
    "infection": {
        "keywords": ["fever", "night sweats", "recent infection", "immunocompromised", "iv drug"],
        "urgency": UrgencyLevel.URGENT,
        "action": "Medical evaluation to rule out spinal infection",
    },
    "malignancy": {
        "keywords": [
            "unexplained weight loss",
            "history of cancer",
            "night pain",
            "age > 50",
            "no relief with rest",
        ],
        "urgency": UrgencyLevel.URGENT,
        "action": "Medical evaluation to rule out malignancy",
    },
    "vascular": {
        "keywords": [
            "pulsatile mass",
            "severe tearing pain",
            "hypotension",
            "aortic aneurysm",
            "claudication",
        ],
        "urgency": UrgencyLevel.EMERGENT,
        "action": "Immediate vascular evaluation",
    },
    "dvt": {
        "keywords": [
            "calf swelling",
            "warmth",
            "recent surgery",
            "immobility",
            "homan",
            "unilateral edema",
        ],
        "urgency": UrgencyLevel.URGENT,
        "action": "Medical evaluation for possible DVT",
    },
    "cardiac": {
        "keywords": [
            "chest pain",
            "shortness of breath",
            "left arm pain",
            "jaw pain",
            "diaphoresis",
        ],
        "urgency": UrgencyLevel.EMERGENT,
        "action": "Immediate cardiac evaluation - call 911 if acute",
    },
}

OT_RED_FLAGS = {
    "stroke_acute": {
        "keywords": ["sudden weakness", "facial droop", "slurred speech", "confusion", "fast"],
        "urgency": UrgencyLevel.EMERGENT,
        "action": "Immediate stroke protocol - call 911",
    },
    "compartment_syndrome": {
        "keywords": [
            "severe pain with passive stretch",
            "paresthesia",
            "pallor",
            "pulselessness",
        ],
        "urgency": UrgencyLevel.EMERGENT,
        "action": "Immediate surgical referral",
    },
    "infection_hand": {
        "keywords": ["kanavel signs", "flexor sheath", "spreading erythema", "fever"],
        "urgency": UrgencyLevel.EMERGENT,
        "action": "Immediate hand surgery referral for possible flexor tenosynovitis",
    },
}

SLP_RED_FLAGS = {
    "aspiration_severe": {
        "keywords": [
            "silent aspiration",
            "recurrent pneumonia",
            "wet voice",
            "coughing with eating",
        ],
        "urgency": UrgencyLevel.URGENT,
        "action": "NPO status and instrumental swallow evaluation",
    },
    "airway_obstruction": {
        "keywords": ["stridor", "dyspnea", "choking", "cyanosis"],
        "urgency": UrgencyLevel.EMERGENT,
        "action": "Immediate airway management",
    },
    "stroke_communication": {
        "keywords": ["sudden aphasia", "sudden dysarthria", "acute confusion"],
        "urgency": UrgencyLevel.EMERGENT,
        "action": "Immediate stroke evaluation",
    },
}


class RedFlagAgent(BaseAgent[RedFlagInput, SafetyAssessment], RuleBasedMixin):
    """Agent for screening red flags and safety concerns.

    Combines rule-based screening with LLM reasoning for comprehensive
    safety assessment. Always runs first in the pipeline as a safety gate.
    """

    def __init__(self, llm: LLMRouter):
        super().__init__(
            llm=llm,
            name="red_flag",
            description="Safety screening and red flag identification",
        )

    @property
    def system_prompt(self) -> str:
        return """You are a clinical safety screening agent for rehabilitation services (PT/OT/SLP).

Your role is to identify red flags, contraindications, and safety concerns that require:
- Immediate medical attention (emergent)
- Same-day evaluation (urgent)
- Precautions during treatment
- Referral to other providers

You must be thorough but avoid over-alarming. Consider:
1. Patient history, comorbidities, and medications
2. Vital signs and objective findings
3. Mechanism of injury and symptom patterns
4. Risk factors for serious pathology

For each red flag identified:
- Explain the clinical concern
- Rate the urgency level
- Recommend specific action

If no red flags are present, clearly state the patient is safe to treat with any relevant precautions.

Always err on the side of patient safety, but use clinical reasoning to avoid unnecessary alarm."""

    @property
    def output_schema(self) -> type[SafetyAssessment]:
        return SafetyAssessment

    def format_input(self, inputs: RedFlagInput, context: AgentContext) -> str:
        # Apply rule-based screening first
        rule_findings = self.apply_rules(inputs)

        sections = [
            f"## Patient Information",
            f"- Age: {inputs.patient.age}",
            f"- Sex: {inputs.patient.sex}",
            f"- Discipline: {inputs.patient.discipline.value}",
            f"- Setting: {inputs.patient.setting.value}",
            "",
            f"## Chief Complaint",
            inputs.chief_complaint,
            "",
        ]

        if inputs.patient.diagnosis:
            sections.extend(["## Diagnoses", ", ".join(inputs.patient.diagnosis), ""])

        if inputs.patient.comorbidities:
            sections.extend(["## Comorbidities", ", ".join(inputs.patient.comorbidities), ""])

        if inputs.patient.medications:
            sections.extend(["## Medications", ", ".join(inputs.patient.medications), ""])

        if inputs.patient.precautions:
            sections.extend(["## Current Precautions", ", ".join(inputs.patient.precautions), ""])

        if inputs.patient.vitals:
            vitals = inputs.patient.vitals
            sections.extend(
                [
                    "## Vital Signs",
                    f"- HR: {vitals.heart_rate}" if vitals.heart_rate else "",
                    f"- BP: {vitals.blood_pressure}" if vitals.blood_pressure else "",
                    f"- SpO2: {vitals.oxygen_saturation}%" if vitals.oxygen_saturation else "",
                    f"- Pain: {vitals.pain_level}/10" if vitals.pain_level is not None else "",
                    "",
                ]
            )

        if inputs.subjective_report:
            sections.extend(["## Subjective Report", inputs.subjective_report, ""])

        if inputs.objective_findings:
            sections.extend(["## Objective Findings", inputs.objective_findings, ""])

        if rule_findings:
            sections.extend(
                [
                    "## Rule-Based Screening Findings (pre-identified)",
                    "The following potential red flags were identified by rule-based screening:",
                    "",
                ]
            )
            for finding, details in rule_findings.items():
                sections.append(f"- {finding}: {details}")
            sections.append("")

        sections.append("Please assess for red flags and safety concerns.")

        return "\n".join(sections)

    def apply_rules(self, inputs: RedFlagInput) -> dict[str, str]:
        """Apply discipline-specific rule-based red flag screening."""
        findings = {}

        # Select discipline-specific rules
        if inputs.patient.discipline == Discipline.PT:
            rules = PT_RED_FLAGS
        elif inputs.patient.discipline == Discipline.OT:
            rules = {**PT_RED_FLAGS, **OT_RED_FLAGS}
        else:  # SLP
            rules = SLP_RED_FLAGS

        # Combine text to search
        search_text = " ".join(
            filter(
                None,
                [
                    inputs.chief_complaint.lower(),
                    (inputs.subjective_report or "").lower(),
                    (inputs.objective_findings or "").lower(),
                    " ".join(inputs.patient.diagnosis).lower(),
                    " ".join(inputs.patient.comorbidities).lower(),
                ],
            )
        )

        # Check each rule
        for flag_name, flag_data in rules.items():
            matches = [kw for kw in flag_data["keywords"] if kw in search_text]
            if matches:
                findings[flag_name] = (
                    f"Matched keywords: {', '.join(matches)}. "
                    f"Urgency: {flag_data['urgency'].value}. "
                    f"Action: {flag_data['action']}"
                )

        # Check vital signs
        if inputs.patient.vitals:
            vitals = inputs.patient.vitals
            if vitals.oxygen_saturation and vitals.oxygen_saturation < 90:
                findings["hypoxia"] = f"SpO2 {vitals.oxygen_saturation}% - requires supplemental O2"
            if vitals.heart_rate and (vitals.heart_rate > 120 or vitals.heart_rate < 50):
                findings["abnormal_hr"] = f"HR {vitals.heart_rate} - abnormal cardiac rate"
            if vitals.blood_pressure_systolic:
                if vitals.blood_pressure_systolic > 180:
                    findings["hypertensive_crisis"] = "SBP > 180 - hold exercise, notify MD"
                elif vitals.blood_pressure_systolic < 90:
                    findings["hypotension"] = "SBP < 90 - hold activity, notify MD"

        return findings

    @property
    def temperature(self) -> float:
        return 0.2  # Very low temperature for safety-critical reasoning

    @property
    def model_tier(self):
        from rehab_os.agents.base import ModelTier
        return ModelTier.STANDARD  # Safety requires reliable reasoning
