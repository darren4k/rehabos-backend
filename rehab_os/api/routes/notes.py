"""Skilled Note Documentation Generator.

Generates Medicare-compliant skilled therapy notes with adaptive styling
based on user/company preferences. Supports customizable templates and
AI-learned patterns from sample notes.
"""

import logging
from datetime import datetime, date
from typing import Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request

from rehab_os.models.clinical import (
    ROMEntry,
    MMTEntry,
    StandardizedTest,
    FunctionalDeficit,
    Vitals as ClinicalVitals,
    BalanceAssessment,
    ToneAssessment,
    SensationAssessment,
    GoalWithBaseline,
    BillingCode,
)

router = APIRouter(prefix="/notes", tags=["documentation"])

logger = logging.getLogger(__name__)


# ==================
# TRANSCRIPT-BASED GENERATION
# ==================

class TranscriptNoteRequest(BaseModel):
    """Request to generate a SOAP note from a voice transcript."""
    transcript: str
    note_type: str = "daily_note"
    patient_context: Optional[dict] = None
    preferences: Optional[dict] = None
    patient_id: Optional[str] = None


SOAP_SYSTEM_PROMPT = """You are a clinical documentation specialist for skilled rehabilitation therapy (PT/OT/SLP).
Your task: convert a therapist's voice dictation into a structured, Medicare-compliant SOAP note.

This note is a LEGAL MEDICAL DOCUMENT that may be printed, faxed to referring physicians,
submitted to clearinghouses, and reviewed by Medicare. Include ALL clinical data mentioned.

RULES:
1. Extract information into these sections: Subjective, Objective, Assessment, Plan
2. Use skilled intervention terminology for Medicare compliance:
   - "neuromuscular re-education" not "exercises"
   - "therapeutic exercise" not "stretching"
   - "gait training" not "walking practice"
   - "transfer training" not "help getting up"
   - "functional mobility training", "balance training", "ADL training"
   - "manual therapy", "patient education", "caregiver training"
3. Include measurable/functional data (distances, times, assist levels, reps/sets)
4. Document patient response to interventions
5. Link interventions to functional goals
6. Justify continued skilled care (medical necessity)
7. Reference prior level of function when mentioned

CLINICAL MEASUREMENTS — ALWAYS include when mentioned in transcript:
8. Range of Motion (ROM): List per joint/region with degrees or qualitative (WFL, WNL).
   Format: "ROM: R hip flexion 110°, L hip flexion 108°, bilateral knee ext 0-135°, cervical WFL with mild rigidity"
9. Manual Muscle Testing (MMT): List per muscle group with standard grades (0-5, with +/-).
   Format: "MMT: Hip flexors 4/5 bilat, hip extensors 3+/5 bilat, quads 4/5 bilat, trunk flexion 3+/5"
10. Standardized Tests: Include full scores with normative interpretation.
    Format: "TUG: 18.5 sec (>14 sec = high fall risk); Berg: 42/56 (medium fall risk); Tinetti: 20/28 (high fall risk)"
11. Functional Deficits: Document prior level vs current level per activity.
    Format: "Sit-to-stand: PLOF independent → current min A (1-25%); Gait level surfaces: PLOF independent → current SBA with RW x 150ft"
12. Vitals: BP, HR, SpO2, RR, pain level/location when mentioned
13. Balance: Static/dynamic sitting/standing, single leg stance times, tandem stance
14. Sensation, tone, coordination, posture when mentioned

For EVALUATIONS and PROGRESS NOTES, the Objective section should be comprehensive enough
to stand alone as a clinical record — include ROM, MMT, standardized test tables, and
functional deficit grids even if the therapist stated them quickly during dictation.

STYLE: {style}
- concise: Brief bullet points, abbreviations OK (pt, w/, c/o, s/p, WNL)
- balanced: Standard clinical documentation, moderate detail
- explanatory: Detailed with clinical reasoning and rationale
- specific: Highly detailed, measurement-focused, quantitative

OUTPUT FORMAT (return ONLY valid JSON, no markdown fences):
{{
  "subjective": "...",
  "objective": "...",
  "assessment": "...",
  "plan": "..."
}}

If the transcript is unclear or missing information for a section, write a clinically appropriate placeholder noting what should be documented."""


# ==================
# CONFIGURATION MODELS
# ==================

class NoteStyle(str, Enum):
    """Documentation style preferences."""
    CONCISE = "concise"           # Brief, bullet-point focused
    BALANCED = "balanced"         # Standard clinical documentation
    EXPLANATORY = "explanatory"   # Detailed with rationale
    SPECIFIC = "specific"         # Highly detailed, measurement-focused


class NoteType(str, Enum):
    """Types of skilled notes."""
    EVALUATION = "evaluation"
    DAILY_NOTE = "daily_note"
    PROGRESS_NOTE = "progress_note"
    RECERTIFICATION = "recertification"
    DISCHARGE_SUMMARY = "discharge_summary"


class Discipline(str, Enum):
    PT = "pt"
    OT = "ot"
    SLP = "slp"


class StylePreferences(BaseModel):
    """User or company style preferences for note generation."""
    style: NoteStyle = NoteStyle.BALANCED
    use_abbreviations: bool = True
    include_clinical_reasoning: bool = True
    include_patient_response: bool = True
    include_education_provided: bool = True
    skilled_interventions_focus: bool = True  # Medicare emphasis
    functional_outcome_emphasis: bool = True
    goal_progress_linking: bool = True

    # Medicare-specific
    document_medical_necessity: bool = True
    include_prior_level_function: bool = True
    justify_skilled_care: bool = True

    # Custom formatting
    paragraph_format: bool = False  # False = bullet points
    max_sentences_per_section: Optional[int] = None


class CompanyGuidelines(BaseModel):
    """Company-specific documentation guidelines."""
    company_name: Optional[str] = None
    sample_notes: list[str] = Field(default_factory=list)
    required_phrases: list[str] = Field(default_factory=list)
    prohibited_terms: list[str] = Field(default_factory=list)
    custom_templates: dict[str, str] = Field(default_factory=dict)
    signature_line: Optional[str] = None
    documentation_standards: Optional[str] = None


class UserPreferences(BaseModel):
    """Combined user and company preferences."""
    user_id: Optional[str] = None
    style_preferences: StylePreferences = Field(default_factory=StylePreferences)
    company_guidelines: Optional[CompanyGuidelines] = None
    learned_patterns: dict = Field(default_factory=dict)


# ==================
# NOTE REQUEST/RESPONSE MODELS
# ==================

class FunctionalStatus(BaseModel):
    """Functional status for documentation."""
    activity: str
    level: str
    assistance_type: Optional[str] = None
    equipment: Optional[str] = None
    distance: Optional[str] = None
    time: Optional[str] = None
    prior_level: Optional[str] = None


class InterventionPerformed(BaseModel):
    """Skilled intervention for documentation."""
    intervention: str
    duration_minutes: Optional[int] = None
    parameters: Optional[str] = None
    patient_response: Optional[str] = None
    skilled_rationale: Optional[str] = None


class GoalProgress(BaseModel):
    """Goal progress tracking."""
    goal_area: str
    goal_text: str
    current_status: str
    progress_rating: Literal["met", "progressing", "plateau", "regression", "new"]
    target_date: Optional[str] = None


class NoteRequest(BaseModel):
    """Request to generate a skilled note."""
    note_type: NoteType
    discipline: Discipline

    # Patient context (no PHI)
    age: Optional[int] = None
    diagnosis: list[str] = Field(default_factory=list)
    precautions: list[str] = Field(default_factory=list)
    setting: Optional[str] = None

    # Session details
    session_date: Optional[str] = None
    session_duration_minutes: Optional[int] = None

    # Functional status
    functional_status: list[FunctionalStatus] = Field(default_factory=list)

    # Interventions performed
    interventions: list[InterventionPerformed] = Field(default_factory=list)

    # Goals
    goals: list[GoalProgress] = Field(default_factory=list)

    # Clinical evaluation data
    rom: list[ROMEntry] = Field(default_factory=list)
    mmt: list[MMTEntry] = Field(default_factory=list)
    standardized_tests: list[StandardizedTest] = Field(default_factory=list)
    functional_deficits: list[FunctionalDeficit] = Field(default_factory=list)
    vitals: Optional[ClinicalVitals] = None
    balance: Optional[BalanceAssessment] = None
    tone: list[ToneAssessment] = Field(default_factory=list)
    sensation: list[SensationAssessment] = Field(default_factory=list)

    # Medical history
    social_history: Optional[str] = None
    past_medical_history: list[str] = Field(default_factory=list)
    past_surgical_history: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    durable_medical_equipment: list[str] = Field(default_factory=list)

    # Goals with baselines
    goals_with_baselines: list[GoalWithBaseline] = Field(default_factory=list)

    # Billing
    billing_codes: list[BillingCode] = Field(default_factory=list)

    # Additional context
    patient_response: Optional[str] = None
    education_provided: Optional[str] = None
    plan_for_next_session: Optional[str] = None
    discharge_recommendations: Optional[str] = None
    barriers_to_progress: Optional[str] = None
    caregiver_training: Optional[str] = None

    # Style override for this note
    style_override: Optional[NoteStyle] = None

    # User preferences (optional)
    preferences: Optional[UserPreferences] = None


class GeneratedNote(BaseModel):
    """Generated skilled note."""
    note_type: str
    content: str
    sections: dict[str, str]

    # Medicare compliance indicators
    medicare_compliant: bool = True
    compliance_checklist: dict[str, bool] = Field(default_factory=dict)
    compliance_warnings: list[str] = Field(default_factory=list)

    # Style info
    style_used: str
    word_count: int

    # Suggestions
    improvement_suggestions: list[str] = Field(default_factory=list)


# ==================
# STORAGE (In-memory for preferences)
# ==================

# User preferences store (in production, use database)
user_preferences_store: dict[str, UserPreferences] = {}

# Company guidelines store
company_guidelines_store: dict[str, CompanyGuidelines] = {}


# ==================
# MEDICARE COMPLIANCE HELPERS
# ==================

MEDICARE_REQUIREMENTS = {
    "daily_note": {
        "required": [
            "skilled_intervention",
            "patient_response",
            "functional_progress",
            "plan_continuation"
        ],
        "recommended": [
            "goal_progress",
            "education_provided",
            "time_documentation"
        ]
    },
    "progress_note": {
        "required": [
            "skilled_intervention_summary",
            "goal_progress",
            "functional_outcomes",
            "medical_necessity",
            "plan_continuation"
        ],
        "recommended": [
            "barriers_addressed",
            "caregiver_involvement",
            "discharge_planning"
        ]
    },
    "evaluation": {
        "required": [
            "prior_level_function",
            "current_functional_status",
            "diagnosis_relevance",
            "skilled_need_justification",
            "treatment_plan",
            "goals_measurable"
        ],
        "recommended": [
            "rehab_potential",
            "discharge_plan",
            "patient_education_needs"
        ]
    },
    "recertification": {
        "required": [
            "progress_summary",
            "goal_updates",
            "continued_medical_necessity",
            "expected_outcomes",
            "revised_treatment_plan"
        ],
        "recommended": [
            "barriers_addressed",
            "patient_participation"
        ]
    },
    "discharge_summary": {
        "required": [
            "initial_vs_discharge_status",
            "goals_achieved",
            "recommendations",
            "home_program"
        ],
        "recommended": [
            "follow_up_plan",
            "caregiver_education_completed"
        ]
    }
}

SKILLED_INTERVENTION_KEYWORDS = [
    "neuromuscular re-education",
    "manual therapy",
    "therapeutic exercise",
    "gait training",
    "balance training",
    "transfer training",
    "ADL training",
    "cognitive training",
    "swallow evaluation",
    "speech-language intervention",
    "dysphagia management",
    "patient education",
    "caregiver training",
    "modalities",
    "functional mobility training"
]


def check_medicare_compliance(note_request: NoteRequest, generated_content: dict) -> tuple[bool, dict, list]:
    """Check note for Medicare compliance."""
    note_type = note_request.note_type.value
    requirements = MEDICARE_REQUIREMENTS.get(note_type, {})

    checklist = {}
    warnings = []

    # Check required elements
    required = requirements.get("required", [])
    for req in required:
        # Simple presence check (in production, use NLP)
        present = any(
            req.replace("_", " ") in str(v).lower()
            for v in generated_content.values()
        )
        checklist[req] = present
        if not present:
            warnings.append(f"Missing required element: {req.replace('_', ' ').title()}")

    # Check for skilled interventions
    has_skilled = any(
        keyword.lower() in str(generated_content).lower()
        for keyword in SKILLED_INTERVENTION_KEYWORDS
    )
    checklist["skilled_interventions_documented"] = has_skilled
    if not has_skilled:
        warnings.append("Consider adding skilled intervention terminology")

    # Check for measurable data
    has_measurements = any(
        char.isdigit() for char in str(generated_content)
    )
    checklist["measurable_data_included"] = has_measurements

    # For evaluations, check clinical data completeness
    if note_type == "evaluation" and isinstance(note_request, NoteRequest):
        has_rom = bool(getattr(note_request, 'rom', None))
        has_mmt = bool(getattr(note_request, 'mmt', None))
        has_tests = bool(getattr(note_request, 'standardized_tests', None))
        has_deficits = bool(getattr(note_request, 'functional_deficits', None))

        checklist["rom_documented"] = has_rom
        checklist["mmt_documented"] = has_mmt
        checklist["standardized_tests_documented"] = has_tests
        checklist["functional_deficits_documented"] = has_deficits

        if not has_rom:
            warnings.append("Evaluation missing ROM measurements")
        if not has_mmt:
            warnings.append("Evaluation missing MMT/strength testing")
        if not has_tests:
            warnings.append("Evaluation missing standardized tests (e.g. TUG, Berg)")
        if not has_deficits:
            warnings.append("Evaluation missing functional deficit documentation")

    compliant = len([w for w in warnings if "Missing required" in w]) == 0

    return compliant, checklist, warnings


# ==================
# NOTE GENERATION HELPERS
# ==================

def format_functional_status(status: FunctionalStatus, style: NoteStyle) -> str:
    """Format functional status entry based on style."""
    if style == NoteStyle.CONCISE:
        result = f"{status.activity}: {status.level}"
        if status.equipment:
            result += f" w/ {status.equipment}"
        return result

    elif style == NoteStyle.SPECIFIC:
        result = f"{status.activity}: {status.level}"
        if status.assistance_type:
            result += f" ({status.assistance_type})"
        if status.equipment:
            result += f" using {status.equipment}"
        if status.distance:
            result += f" x {status.distance}"
        if status.time:
            result += f" in {status.time}"
        if status.prior_level:
            result += f" (PLOF: {status.prior_level})"
        return result

    else:  # BALANCED or EXPLANATORY
        parts = [f"{status.activity}: {status.level}"]
        if status.assistance_type:
            parts.append(f"requiring {status.assistance_type}")
        if status.equipment:
            parts.append(f"with {status.equipment}")
        if status.distance:
            parts.append(f"for {status.distance}")
        return " ".join(parts)


def format_intervention(intervention: InterventionPerformed, style: NoteStyle) -> str:
    """Format intervention entry based on style."""
    if style == NoteStyle.CONCISE:
        result = intervention.intervention
        if intervention.duration_minutes:
            result += f" x {intervention.duration_minutes} min"
        return result

    elif style == NoteStyle.EXPLANATORY:
        result = f"{intervention.intervention}"
        if intervention.duration_minutes:
            result += f" for {intervention.duration_minutes} minutes"
        if intervention.parameters:
            result += f" ({intervention.parameters})"
        if intervention.skilled_rationale:
            result += f". Skilled rationale: {intervention.skilled_rationale}"
        if intervention.patient_response:
            result += f" Patient response: {intervention.patient_response}"
        return result

    else:  # BALANCED or SPECIFIC
        parts = [intervention.intervention]
        if intervention.duration_minutes:
            parts.append(f"x {intervention.duration_minutes} min")
        if intervention.parameters:
            parts.append(f"({intervention.parameters})")
        if intervention.patient_response:
            parts.append(f"- Pt response: {intervention.patient_response}")
        return " ".join(parts)


def format_rom(entries: list[ROMEntry], style: NoteStyle) -> str:
    """Format ROM entries for documentation."""
    lines = []
    for e in entries:
        if e.value is not None:
            val = f"{e.value}°"
        elif e.qualitative:
            val = e.qualitative
        else:
            val = "not tested"
        label = e.joint.replace("_", " ").title()
        if e.motion != "general":
            label += f" {e.motion}"
        if style == NoteStyle.CONCISE:
            lines.append(f"{label}: {val}")
        else:
            side = f" ({e.side})" if e.side != "bilateral" else ""
            lines.append(f"{label}{side}: {val}")
    return "\n".join(f"- {l}" for l in lines)


def format_mmt(entries: list[MMTEntry], style: NoteStyle) -> str:
    """Format MMT entries for documentation."""
    lines = []
    for e in entries:
        label = e.muscle_group.replace("_", " ").title()
        side = f" ({e.side})" if e.side != "bilateral" else ""
        lines.append(f"{label}{side}: {e.grade}")
    return "\n".join(f"- {l}" for l in lines)


def format_standardized_tests(tests: list[StandardizedTest], style: NoteStyle) -> str:
    """Format standardized test results."""
    lines = []
    for t in tests:
        result = f"{t.name}: {t.score}"
        if t.max_score is not None:
            result += f"/{t.max_score}"
        if t.unit:
            result += f" {t.unit}"
        if t.interpretation and style != NoteStyle.CONCISE:
            result += f" — {t.interpretation}"
        lines.append(result)
    return "\n".join(f"- {l}" for l in lines)


def format_functional_deficits(deficits: list[FunctionalDeficit], style: NoteStyle) -> str:
    """Format functional deficit grid."""
    lines = []
    current_cat = None
    for d in deficits:
        if d.category != current_cat:
            current_cat = d.category
            lines.append(f"\n  {current_cat.replace('_', ' ').title()}:")
        activity = d.activity.replace("_", " ").title()
        if style == NoteStyle.CONCISE:
            lines.append(f"  {activity}: {d.current_level}")
        else:
            line = f"  {activity}: {d.current_level} (PLOF: {d.prior_level})"
            if d.assistive_device:
                line += f" w/ {d.assistive_device}"
            if d.distance:
                line += f" x {d.distance}"
            if d.quality_notes:
                line += f" — {d.quality_notes}"
            lines.append(line)
    return "\n".join(f"- {l}" if not l.startswith("\n") else l for l in lines)


def format_vitals(vitals: ClinicalVitals) -> str:
    """Format vitals for documentation."""
    parts = []
    if vitals.blood_pressure_sitting:
        parts.append(f"BP: {vitals.blood_pressure_sitting} (sitting)")
    if vitals.blood_pressure_standing:
        parts.append(f"BP: {vitals.blood_pressure_standing} (standing)")
    if vitals.heart_rate:
        parts.append(f"HR: {vitals.heart_rate} bpm")
    if vitals.spo2:
        parts.append(f"SpO2: {vitals.spo2}%")
    if vitals.respiratory_rate:
        parts.append(f"RR: {vitals.respiratory_rate}")
    if vitals.pain_level is not None:
        loc = f" ({vitals.pain_location})" if vitals.pain_location else ""
        parts.append(f"Pain: {vitals.pain_level}/10{loc}")
    return "\n".join(f"- {p}" for p in parts)


def generate_note_sections(request: NoteRequest, style: NoteStyle) -> dict[str, str]:
    """Generate note sections based on request and style."""
    sections = {}

    # Header info
    if request.session_date:
        sections["date"] = request.session_date
    if request.session_duration_minutes:
        sections["duration"] = f"Treatment time: {request.session_duration_minutes} minutes"

    # Diagnosis/Precautions
    if request.diagnosis:
        sections["diagnosis"] = ", ".join(request.diagnosis)
    if request.precautions:
        sections["precautions"] = ", ".join(request.precautions)

    # Vitals
    if request.vitals:
        sections["vitals"] = format_vitals(request.vitals)

    # Social History
    if request.social_history:
        sections["social_history"] = request.social_history

    # Past Medical History
    if request.past_medical_history:
        sections["past_medical_history"] = "\n".join(f"- {dx}" for dx in request.past_medical_history)

    # Medications
    if request.medications:
        sections["medications"] = "\n".join(f"- {m}" for m in request.medications)

    # Functional Deficits (detailed grid)
    if request.functional_deficits:
        sections["functional_deficits"] = format_functional_deficits(request.functional_deficits, style)

    # Functional Status
    if request.functional_status:
        status_lines = [
            format_functional_status(s, style)
            for s in request.functional_status
        ]
        if style == NoteStyle.CONCISE or not request.preferences or not request.preferences.style_preferences.paragraph_format:
            sections["functional_status"] = "\n".join(f"- {line}" for line in status_lines)
        else:
            sections["functional_status"] = ". ".join(status_lines) + "."

    # Skilled Interventions
    if request.interventions:
        intervention_lines = [
            format_intervention(i, style)
            for i in request.interventions
        ]
        sections["skilled_interventions"] = "\n".join(f"- {line}" for line in intervention_lines)

    # ROM (Objective)
    if request.rom:
        sections["rom"] = format_rom(request.rom, style)

    # MMT / Strength (Objective)
    if request.mmt:
        sections["mmt"] = format_mmt(request.mmt, style)

    # Standardized Tests (Assessment)
    if request.standardized_tests:
        sections["standardized_tests"] = format_standardized_tests(request.standardized_tests, style)

    # Balance Assessment
    if request.balance:
        bal = request.balance
        bal_lines = []
        for field_name, label in [
            ("static_sitting", "Static Sitting"), ("dynamic_sitting", "Dynamic Sitting"),
            ("static_standing", "Static Standing"), ("dynamic_standing", "Dynamic Standing"),
            ("single_leg_stance_right", "SLS Right"), ("single_leg_stance_left", "SLS Left"),
            ("tandem_stance", "Tandem Stance"),
        ]:
            val = getattr(bal, field_name, None)
            if val:
                bal_lines.append(f"- {label}: {val}")
        if bal_lines:
            sections["balance_assessment"] = "\n".join(bal_lines)

    # Goals with Baselines
    if request.goals_with_baselines:
        goal_lines = []
        for g in request.goals_with_baselines:
            tf = f" ({g.timeframe})" if g.timeframe else ""
            goal_lines.append(f"- [{g.type.upper()}] {g.area}: {g.goal} (Baseline: {g.baseline}){tf}")
        sections["goals_with_baselines"] = "\n".join(goal_lines)

    # Patient Response
    if request.patient_response:
        sections["patient_response"] = request.patient_response

    # Goals Progress
    if request.goals:
        goal_lines = []
        for g in request.goals:
            progress_emoji = {
                "met": "ACHIEVED",
                "progressing": "PROGRESSING",
                "plateau": "PLATEAU",
                "regression": "REGRESSION",
                "new": "NEW"
            }.get(g.progress_rating, g.progress_rating.upper())

            if style == NoteStyle.CONCISE:
                goal_lines.append(f"{g.goal_area}: {progress_emoji}")
            else:
                goal_lines.append(f"{g.goal_area}: {g.current_status} [{progress_emoji}]")

        sections["goal_progress"] = "\n".join(f"- {line}" for line in goal_lines)

    # Education
    if request.education_provided:
        sections["education_provided"] = request.education_provided

    # Caregiver training
    if request.caregiver_training:
        sections["caregiver_training"] = request.caregiver_training

    # Barriers
    if request.barriers_to_progress:
        sections["barriers"] = request.barriers_to_progress

    # Plan
    if request.plan_for_next_session:
        sections["plan"] = request.plan_for_next_session

    # Discharge (if applicable)
    if request.discharge_recommendations:
        sections["discharge_recommendations"] = request.discharge_recommendations

    return sections


def assemble_note(sections: dict[str, str], note_type: NoteType, style: NoteStyle,
                  preferences: Optional[UserPreferences] = None) -> str:
    """Assemble final note from sections."""
    lines = []

    # Section headers based on note type
    section_order = {
        NoteType.DAILY_NOTE: [
            ("diagnosis", "Dx"),
            ("precautions", "Precautions"),
            ("vitals", "Vitals"),
            ("functional_status", "Functional Status"),
            ("rom", "Range of Motion"),
            ("mmt", "Manual Muscle Testing"),
            ("skilled_interventions", "Skilled Interventions"),
            ("patient_response", "Patient Response"),
            ("standardized_tests", "Standardized Tests"),
            ("goal_progress", "Goal Progress"),
            ("education_provided", "Education Provided"),
            ("plan", "Plan"),
            ("duration", None)
        ],
        NoteType.PROGRESS_NOTE: [
            ("diagnosis", "Dx"),
            ("functional_status", "Current Functional Status"),
            ("goal_progress", "Goal Progress"),
            ("skilled_interventions", "Interventions This Period"),
            ("barriers", "Barriers to Progress"),
            ("plan", "Plan for Continued Treatment"),
            ("duration", "Treatment Time")
        ],
        NoteType.EVALUATION: [
            ("diagnosis", "Diagnosis"),
            ("precautions", "Precautions/Contraindications"),
            ("vitals", "Vitals"),
            ("social_history", "Social History"),
            ("past_medical_history", "Past Medical History"),
            ("medications", "Medications"),
            ("functional_deficits", "Functional Deficits"),
            ("functional_status", "Current Functional Status"),
            ("rom", "Range of Motion"),
            ("mmt", "Manual Muscle Testing"),
            ("balance_assessment", "Balance Assessment"),
            ("standardized_tests", "Standardized Tests"),
            ("skilled_interventions", "Assessment Performed"),
            ("goals_with_baselines", "Goals"),
            ("goal_progress", "Goal Progress"),
            ("plan", "Treatment Plan"),
            ("education_provided", "Patient/Caregiver Education")
        ],
        NoteType.DISCHARGE_SUMMARY: [
            ("diagnosis", "Diagnosis"),
            ("functional_status", "Discharge Functional Status"),
            ("goal_progress", "Goals Achieved"),
            ("discharge_recommendations", "Recommendations"),
            ("education_provided", "Education Provided")
        ]
    }

    order = section_order.get(note_type, list((k, k.replace("_", " ").title()) for k in sections.keys()))

    for section_key, header in order:
        if section_key in sections:
            content = sections[section_key]
            if header:
                lines.append(f"\n{header.upper()}:")
            lines.append(content)

    # Add signature line if company guidelines specify
    if preferences and preferences.company_guidelines and preferences.company_guidelines.signature_line:
        lines.append(f"\n{preferences.company_guidelines.signature_line}")

    return "\n".join(lines).strip()


# ==================
# API ENDPOINTS
# ==================

async def _generate_from_transcript(
    transcript: str,
    note_type: str,
    patient_context: Optional[dict],
    preferences: Optional[dict],
    llm_router,
    session_memory=None,
    patient_id: Optional[str] = None,
) -> GeneratedNote:
    """Core logic for generating a SOAP note from a transcript via LLM."""
    import json as _json
    from rehab_os.llm.base import Message as LLMMessage, MessageRole

    style = "balanced"
    if preferences and "style" in preferences:
        style = preferences["style"]
    elif preferences and "style_preferences" in preferences:
        style = preferences["style_preferences"].get("style", "balanced")

    # Inject cross-namespace longitudinal context if available
    longitudinal_block = ""
    if session_memory and patient_id:
        try:
            from rehab_os.memory.cross_namespace import get_patient_history, format_cross_namespace_context
            records = get_patient_history(session_memory, patient_id)
            longitudinal_block = format_cross_namespace_context(records)
        except Exception as _e:
            logger.warning("Failed to fetch cross-namespace context: %s", _e)

    # Build user message with optional patient context
    user_content = f"Transcript:\n{transcript}"
    if patient_context:
        ctx_parts = []
        for k, v in patient_context.items():
            if isinstance(v, list):
                ctx_parts.append(f"{k}: {', '.join(str(i) for i in v)}")
            else:
                ctx_parts.append(f"{k}: {v}")
        user_content = f"Patient Context:\n" + "\n".join(ctx_parts) + "\n\n" + user_content

    if longitudinal_block:
        user_content = longitudinal_block + "\n\n" + user_content

    messages = [
        LLMMessage(role=MessageRole.SYSTEM, content=SOAP_SYSTEM_PROMPT.format(style=style)),
        LLMMessage(role=MessageRole.USER, content=user_content),
    ]

    response = await llm_router.complete(messages, temperature=0.3, max_tokens=4096)

    # Parse JSON from response
    raw = response.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        sections = _json.loads(raw)
    except _json.JSONDecodeError:
        # If LLM didn't return valid JSON, wrap the whole response
        sections = {
            "subjective": "",
            "objective": "",
            "assessment": "",
            "plan": "",
            "raw_output": raw,
        }

    # Build content string
    content_lines = []
    for section_name in ["subjective", "objective", "assessment", "plan"]:
        val = sections.get(section_name, "")
        if val:
            content_lines.append(f"{section_name.upper()}:\n{val}")
    content = "\n\n".join(content_lines)

    # Medicare compliance check on the sections
    # Reuse the existing keyword-based checks
    checklist = {}
    warnings = []

    has_skilled = any(
        kw.lower() in content.lower() for kw in SKILLED_INTERVENTION_KEYWORDS
    )
    checklist["skilled_interventions_documented"] = has_skilled
    if not has_skilled:
        warnings.append("Consider adding skilled intervention terminology")

    has_measurements = any(c.isdigit() for c in content)
    checklist["measurable_data_included"] = has_measurements
    if not has_measurements:
        warnings.append("Include measurable/quantitative data")

    nt = note_type.replace("-", "_")
    requirements = MEDICARE_REQUIREMENTS.get(nt, {})
    for req in requirements.get("required", []):
        present = req.replace("_", " ") in content.lower()
        checklist[req] = present
        if not present:
            warnings.append(f"Missing required element: {req.replace('_', ' ').title()}")

    compliant = not any("Missing required" in w for w in warnings)

    return GeneratedNote(
        note_type=nt,
        content=content,
        sections=sections,
        medicare_compliant=compliant,
        compliance_checklist=checklist,
        compliance_warnings=warnings,
        style_used=style,
        word_count=len(content.split()),
        improvement_suggestions=[],
    )


@router.post("/generate-from-transcript", response_model=GeneratedNote)
async def generate_note_from_transcript(req: TranscriptNoteRequest, request: Request):
    """Generate a Medicare-compliant SOAP note from a voice transcript using LLM."""
    llm_router = request.app.state.llm_router
    session_memory = getattr(request.app.state, "session_memory", None)
    return await _generate_from_transcript(
        transcript=req.transcript,
        note_type=req.note_type,
        patient_context=req.patient_context,
        preferences=req.preferences,
        llm_router=llm_router,
        session_memory=session_memory,
        patient_id=req.patient_id,
    )


class NoteRequestWithTranscript(NoteRequest):
    """Extended NoteRequest that optionally accepts a transcript field."""
    transcript: Optional[str] = None


@router.post("/generate", response_model=GeneratedNote)
async def generate_skilled_note(request: Request, note_request: NoteRequestWithTranscript):
    """Generate a Medicare-compliant skilled note.

    The note is generated based on the provided clinical data and styled
    according to user/company preferences. Medicare compliance is checked
    and suggestions are provided.

    If a `transcript` field is provided, routes to LLM-based generation.
    """
    # If transcript provided, route to LLM path
    if note_request.transcript:
        llm_router = request.app.state.llm_router
        prefs = None
        if note_request.preferences:
            prefs = {"style": (note_request.style_override or note_request.preferences.style_preferences.style).value}
        elif note_request.style_override:
            prefs = {"style": note_request.style_override.value}
        return await _generate_from_transcript(
            transcript=note_request.transcript,
            note_type=note_request.note_type.value,
            patient_context={"diagnosis": note_request.diagnosis, "precautions": note_request.precautions} if note_request.diagnosis else None,
            preferences=prefs,
            llm_router=llm_router,
        )

    # Determine style to use
    style = note_request.style_override or (
        note_request.preferences.style_preferences.style
        if note_request.preferences else NoteStyle.BALANCED
    )

    # Generate sections
    sections = generate_note_sections(note_request, style)

    # Assemble full note
    content = assemble_note(sections, note_request.note_type, style, note_request.preferences)

    # Check Medicare compliance
    compliant, checklist, warnings = check_medicare_compliance(note_request, sections)

    # Generate improvement suggestions
    suggestions = []
    if not note_request.interventions:
        suggestions.append("Add skilled interventions to justify medical necessity")
    if not note_request.goals:
        suggestions.append("Include measurable goals with progress ratings")
    if not note_request.patient_response:
        suggestions.append("Document patient response to treatment")
    if style != NoteStyle.EXPLANATORY and not any("rationale" in str(i).lower() for i in note_request.interventions):
        suggestions.append("Consider adding skilled rationale for interventions")

    # Apply company-specific requirements
    if note_request.preferences and note_request.preferences.company_guidelines:
        cg = note_request.preferences.company_guidelines

        # Check required phrases
        for phrase in cg.required_phrases:
            if phrase.lower() not in content.lower():
                suggestions.append(f"Company guideline: Include phrase '{phrase}'")

        # Check prohibited terms
        for term in cg.prohibited_terms:
            if term.lower() in content.lower():
                warnings.append(f"Company guideline: Remove prohibited term '{term}'")

    return GeneratedNote(
        note_type=note_request.note_type.value,
        content=content,
        sections=sections,
        medicare_compliant=compliant,
        compliance_checklist=checklist,
        compliance_warnings=warnings,
        style_used=style.value,
        word_count=len(content.split()),
        improvement_suggestions=suggestions
    )


@router.post("/preferences/save")
async def save_user_preferences(user_id: str, preferences: UserPreferences):
    """Save user documentation preferences."""
    preferences.user_id = user_id
    user_preferences_store[user_id] = preferences

    return {"status": "saved", "user_id": user_id}


@router.get("/preferences/{user_id}")
async def get_user_preferences(user_id: str):
    """Get user documentation preferences."""
    if user_id not in user_preferences_store:
        return UserPreferences(user_id=user_id)
    return user_preferences_store[user_id]


@router.post("/company-guidelines/save")
async def save_company_guidelines(company_id: str, guidelines: CompanyGuidelines):
    """Save company documentation guidelines."""
    company_guidelines_store[company_id] = guidelines

    return {"status": "saved", "company_id": company_id}


@router.get("/company-guidelines/{company_id}")
async def get_company_guidelines(company_id: str):
    """Get company documentation guidelines."""
    if company_id not in company_guidelines_store:
        raise HTTPException(status_code=404, detail="Company guidelines not found")
    return company_guidelines_store[company_id]


@router.post("/learn-from-sample")
async def learn_from_sample_note(user_id: str, sample_note: str, note_type: NoteType):
    """Analyze a sample note to learn user's documentation patterns.

    The AI will extract patterns like:
    - Preferred sentence structure
    - Common phrases used
    - Section organization
    - Level of detail
    """
    # Get or create user preferences
    preferences = user_preferences_store.get(user_id, UserPreferences(user_id=user_id))

    # Analyze sample (simplified - in production use NLP/LLM)
    patterns = {
        "note_type": note_type.value,
        "word_count": len(sample_note.split()),
        "uses_bullets": "-" in sample_note or "•" in sample_note,
        "uses_abbreviations": any(abbr in sample_note for abbr in ["pt", "w/", "c/o", "s/p", "WNL"]),
        "sentence_avg_length": len(sample_note) / max(1, sample_note.count(".")),
        "extracted_phrases": [],  # Would extract common phrases with NLP
    }

    # Store learned patterns
    if "samples" not in preferences.learned_patterns:
        preferences.learned_patterns["samples"] = []
    preferences.learned_patterns["samples"].append(patterns)

    # Update style inference
    if patterns["uses_bullets"]:
        preferences.style_preferences.paragraph_format = False
    if patterns["word_count"] < 150:
        preferences.style_preferences.style = NoteStyle.CONCISE
    elif patterns["word_count"] > 400:
        preferences.style_preferences.style = NoteStyle.EXPLANATORY

    preferences.style_preferences.use_abbreviations = patterns["uses_abbreviations"]

    user_preferences_store[user_id] = preferences

    return {
        "status": "learned",
        "patterns_extracted": patterns,
        "updated_preferences": preferences.style_preferences.model_dump()
    }


@router.get("/templates")
async def get_note_templates():
    """Get available note templates."""
    return {
        "templates": [
            {
                "type": "daily_note",
                "name": "Daily Treatment Note",
                "description": "Standard daily skilled therapy note"
            },
            {
                "type": "progress_note",
                "name": "Progress Note",
                "description": "Weekly/bi-weekly progress summary"
            },
            {
                "type": "evaluation",
                "name": "Initial Evaluation",
                "description": "Comprehensive initial evaluation"
            },
            {
                "type": "recertification",
                "name": "Recertification",
                "description": "Recertification for continued care"
            },
            {
                "type": "discharge_summary",
                "name": "Discharge Summary",
                "description": "Discharge documentation"
            }
        ],
        "styles": [
            {"value": "concise", "label": "Concise", "description": "Brief, bullet-focused"},
            {"value": "balanced", "label": "Balanced", "description": "Standard clinical format"},
            {"value": "explanatory", "label": "Explanatory", "description": "Detailed with rationale"},
            {"value": "specific", "label": "Specific", "description": "Measurement-focused"}
        ]
    }


@router.get("/medicare-requirements/{note_type}")
async def get_medicare_requirements(note_type: str):
    """Get Medicare documentation requirements for a note type."""
    if note_type not in MEDICARE_REQUIREMENTS:
        raise HTTPException(status_code=404, detail="Unknown note type")

    return {
        "note_type": note_type,
        "requirements": MEDICARE_REQUIREMENTS[note_type],
        "skilled_keywords": SKILLED_INTERVENTION_KEYWORDS[:10],
        "tips": [
            "Document medical necessity for skilled services",
            "Link interventions to functional goals",
            "Include measurable progress data",
            "Document patient response to treatment",
            "Justify continued need for skilled care"
        ]
    }
