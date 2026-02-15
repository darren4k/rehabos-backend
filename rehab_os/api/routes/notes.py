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

router = APIRouter(prefix="/notes", tags=["documentation"])

logger = logging.getLogger(__name__)


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
            ("functional_status", "Functional Status"),
            ("skilled_interventions", "Skilled Interventions"),
            ("patient_response", "Patient Response"),
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
            ("functional_status", "Current Functional Status"),
            ("skilled_interventions", "Assessment Performed"),
            ("goal_progress", "Established Goals"),
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

@router.post("/generate", response_model=GeneratedNote)
async def generate_skilled_note(request: NoteRequest):
    """Generate a Medicare-compliant skilled note.

    The note is generated based on the provided clinical data and styled
    according to user/company preferences. Medicare compliance is checked
    and suggestions are provided.
    """
    # Determine style to use
    style = request.style_override or (
        request.preferences.style_preferences.style
        if request.preferences else NoteStyle.BALANCED
    )

    # Generate sections
    sections = generate_note_sections(request, style)

    # Assemble full note
    content = assemble_note(sections, request.note_type, style, request.preferences)

    # Check Medicare compliance
    compliant, checklist, warnings = check_medicare_compliance(request, sections)

    # Generate improvement suggestions
    suggestions = []
    if not request.interventions:
        suggestions.append("Add skilled interventions to justify medical necessity")
    if not request.goals:
        suggestions.append("Include measurable goals with progress ratings")
    if not request.patient_response:
        suggestions.append("Document patient response to treatment")
    if style != NoteStyle.EXPLANATORY and not any("rationale" in str(i).lower() for i in request.interventions):
        suggestions.append("Consider adding skilled rationale for interventions")

    # Apply company-specific requirements
    if request.preferences and request.preferences.company_guidelines:
        cg = request.preferences.company_guidelines

        # Check required phrases
        for phrase in cg.required_phrases:
            if phrase.lower() not in content.lower():
                suggestions.append(f"Company guideline: Include phrase '{phrase}'")

        # Check prohibited terms
        for term in cg.prohibited_terms:
            if term.lower() in content.lower():
                warnings.append(f"Company guideline: Remove prohibited term '{term}'")

    return GeneratedNote(
        note_type=request.note_type.value,
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
