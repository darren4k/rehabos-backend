"""Mobile-optimized API endpoints for RehabOS.

Designed for quick consultations during rounds and
patient-facing features like home exercise programs.
"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/mobile", tags=["mobile"])


class QuickConsultRequest(BaseModel):
    """Minimal request for quick mobile consultations."""

    chief_complaint: str = Field(..., description="Brief description of patient issue")
    age: int = Field(50, ge=0, le=120)
    discipline: str = Field("PT", description="PT, OT, or SLP")

    # Optional quick fields
    setting: str = "outpatient"
    red_flag_symptoms: list[str] = Field(default_factory=list)
    prior_treatment: Optional[str] = None


class QuickConsultResponse(BaseModel):
    """Condensed response for mobile display."""

    consult_id: str

    # Safety (always shown first)
    is_safe: bool
    urgency: str  # routine, urgent, emergent
    red_flag_alert: Optional[str] = None

    # Diagnosis summary
    diagnosis: str
    icd_code: str
    confidence: float

    # Key recommendations (top 3)
    key_interventions: list[str]
    visit_frequency: str

    # For patient handoff
    patient_instructions: Optional[str] = None


class HEPRequest(BaseModel):
    """Request for home exercise program."""

    condition: str
    discipline: str = "PT"
    difficulty_level: str = "moderate"  # easy, moderate, advanced
    equipment_available: list[str] = Field(default_factory=list)


class Exercise(BaseModel):
    """A single exercise in the HEP."""

    name: str
    description: str
    sets: int
    reps: str  # "10-15" or "30 seconds"
    frequency: str  # "2x daily"
    precautions: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None


class HEPResponse(BaseModel):
    """Home exercise program response."""

    condition: str
    difficulty_level: str
    exercises: list[Exercise]
    general_instructions: str
    precautions: list[str]
    progression_criteria: str


class SafetyCheckRequest(BaseModel):
    """Quick safety check request."""

    symptoms: list[str]
    discipline: str = "PT"
    age: Optional[int] = None
    vital_signs: Optional[dict[str, Any]] = None


class SafetyCheckResponse(BaseModel):
    """Safety check response."""

    is_safe: bool
    urgency_level: str
    red_flags_found: list[str]
    recommendation: str
    refer_to: Optional[str] = None


@router.post("/quick-consult", response_model=QuickConsultResponse)
async def quick_consult(request: QuickConsultRequest):
    """Quick consultation for mobile use during rounds.

    Returns condensed results optimized for small screens
    and quick decision-making.
    """
    import uuid
    from rehab_os.models.output import ClinicalRequest
    from rehab_os.models.patient import PatientContext, Discipline, CareSetting
    from rehab_os.llm import create_router_from_settings
    from rehab_os.agents import Orchestrator

    consult_id = str(uuid.uuid4())[:8]

    # Build minimal patient context
    patient = PatientContext(
        age=request.age,
        sex="other",
        chief_complaint=request.chief_complaint,
        discipline=Discipline(request.discipline),
        setting=CareSetting(request.setting),
    )

    clinical_request = ClinicalRequest(
        query=request.chief_complaint,
        patient=patient,
        discipline=Discipline(request.discipline),
        setting=CareSetting(request.setting),
        task_type="plan_only",  # Skip documentation for speed
    )

    try:
        llm = create_router_from_settings()
        orchestrator = Orchestrator(llm=llm)

        # Run with skip_qa for faster response
        result = await orchestrator.process(clinical_request, skip_qa=True)

        # Build condensed response
        red_flag_alert = None
        if result.safety and result.safety.red_flags:
            red_flag_alert = f"{len(result.safety.red_flags)} red flag(s): " + \
                ", ".join(rf.finding for rf in result.safety.red_flags[:2])

        key_interventions = []
        if result.plan and result.plan.interventions:
            key_interventions = [i.name for i in result.plan.interventions[:3]]

        return QuickConsultResponse(
            consult_id=consult_id,
            is_safe=result.safety.is_safe_to_treat if result.safety else True,
            urgency=result.safety.urgency_level.value if result.safety else "routine",
            red_flag_alert=red_flag_alert,
            diagnosis=result.diagnosis.primary_diagnosis if result.diagnosis else "Assessment needed",
            icd_code=result.diagnosis.icd_codes[0] if result.diagnosis and result.diagnosis.icd_codes else "Z99.9",
            confidence=result.diagnosis.confidence if result.diagnosis else 0.0,
            key_interventions=key_interventions,
            visit_frequency=result.plan.visit_frequency if result.plan else "TBD",
            patient_instructions=None,  # Could add from plan.home_program
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consultation failed: {str(e)}")


@router.post("/safety-check", response_model=SafetyCheckResponse)
async def safety_check(request: SafetyCheckRequest):
    """Quick safety screening for symptom triage.

    Use this before starting treatment to check for red flags.
    """
    from rehab_os.models.patient import PatientContext, Discipline, CareSetting
    from rehab_os.agents.red_flag import RedFlagAgent, RedFlagInput
    from rehab_os.llm import create_router_from_settings

    try:
        llm = create_router_from_settings()
        agent = RedFlagAgent(llm)

        patient = PatientContext(
            age=request.age or 50,
            sex="other",
            chief_complaint=", ".join(request.symptoms),
            discipline=Discipline(request.discipline),
            setting=CareSetting.OUTPATIENT,
        )

        red_flag_input = RedFlagInput(
            patient=patient,
            chief_complaint=", ".join(request.symptoms),
            vitals=request.vital_signs,
        )

        result = await agent.run(red_flag_input)

        red_flags_found = [rf.finding for rf in result.red_flags] if result.red_flags else []

        return SafetyCheckResponse(
            is_safe=result.is_safe_to_treat,
            urgency_level=result.urgency_level.value,
            red_flags_found=red_flags_found,
            recommendation=result.summary,
            refer_to=result.referral_to,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Safety check failed: {str(e)}")


@router.post("/hep", response_model=HEPResponse)
async def get_home_exercise_program(request: HEPRequest):
    """Get a home exercise program for a condition.

    Returns exercises formatted for patient education
    and mobile viewing.
    """
    # This would ideally use the PlanAgent or a dedicated HEP database
    # For now, return structured placeholder

    # Common exercises by condition (simplified)
    hep_templates = {
        "low back pain": {
            "exercises": [
                Exercise(
                    name="Cat-Cow Stretch",
                    description="On hands and knees, alternate arching and rounding your back",
                    sets=2,
                    reps="10 each direction",
                    frequency="2x daily",
                    precautions="Stop if sharp pain occurs",
                ),
                Exercise(
                    name="Bird Dog",
                    description="On hands and knees, extend opposite arm and leg",
                    sets=2,
                    reps="10 each side",
                    frequency="2x daily",
                ),
                Exercise(
                    name="Pelvic Tilts",
                    description="Lying on back with knees bent, flatten low back against floor",
                    sets=2,
                    reps="15",
                    frequency="2x daily",
                ),
            ],
            "precautions": ["Avoid bending and twisting simultaneously", "Stop if symptoms worsen"],
            "progression": "Add resistance band when able to complete 3 sets of 15 without fatigue",
        },
        "knee pain": {
            "exercises": [
                Exercise(
                    name="Quad Sets",
                    description="Sitting with leg straight, tighten thigh muscle and hold",
                    sets=3,
                    reps="10 (hold 5 seconds)",
                    frequency="3x daily",
                ),
                Exercise(
                    name="Straight Leg Raises",
                    description="Lying on back, lift straight leg to height of bent knee",
                    sets=2,
                    reps="10-15",
                    frequency="2x daily",
                ),
                Exercise(
                    name="Heel Slides",
                    description="Lying on back, slide heel toward buttock and return",
                    sets=2,
                    reps="15",
                    frequency="2x daily",
                ),
            ],
            "precautions": ["Avoid deep squats", "Use ice after exercise if swelling occurs"],
            "progression": "Add ankle weight when exercises become easy",
        },
    }

    # Find matching template or use default
    condition_key = request.condition.lower()
    template = None
    for key, value in hep_templates.items():
        if key in condition_key:
            template = value
            break

    if not template:
        # Generic template
        template = {
            "exercises": [
                Exercise(
                    name="Range of Motion",
                    description="Gentle movement through available range",
                    sets=2,
                    reps="10",
                    frequency="2x daily",
                ),
            ],
            "precautions": ["Start slowly", "Stop if pain increases"],
            "progression": "Increase reps when comfortable",
        }

    return HEPResponse(
        condition=request.condition,
        difficulty_level=request.difficulty_level,
        exercises=template["exercises"],
        general_instructions="Perform exercises on a firm surface. Breathe normally throughout. Do not hold your breath.",
        precautions=template["precautions"],
        progression_criteria=template["progression"],
    )


@router.get("/disciplines")
async def get_disciplines():
    """Get available disciplines and their descriptions."""
    return {
        "disciplines": [
            {
                "code": "PT",
                "name": "Physical Therapy",
                "description": "Movement, strength, and functional rehabilitation",
                "common_conditions": ["Back pain", "Joint pain", "Post-surgical", "Sports injuries"],
            },
            {
                "code": "OT",
                "name": "Occupational Therapy",
                "description": "Daily activities, fine motor, and adaptive strategies",
                "common_conditions": ["Stroke", "Hand injuries", "Cognitive impairment", "ADL training"],
            },
            {
                "code": "SLP",
                "name": "Speech-Language Pathology",
                "description": "Communication, swallowing, and cognitive-communication",
                "common_conditions": ["Dysphagia", "Aphasia", "Voice disorders", "Cognitive deficits"],
            },
        ]
    }


@router.get("/settings")
async def get_care_settings():
    """Get available care settings."""
    return {
        "settings": [
            {"code": "outpatient", "name": "Outpatient", "typical_frequency": "2-3x/week"},
            {"code": "inpatient", "name": "Inpatient Rehab", "typical_frequency": "Daily"},
            {"code": "acute", "name": "Acute Care", "typical_frequency": "1-2x/day"},
            {"code": "home_health", "name": "Home Health", "typical_frequency": "2-3x/week"},
            {"code": "snf", "name": "Skilled Nursing", "typical_frequency": "5x/week"},
        ]
    }
