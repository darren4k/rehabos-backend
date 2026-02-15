"""Intelligent Rehabilitation Program Generator.

Creates personalized rehab programs that scale from standard templates
to highly individualized plans based on available patient data.
Includes learning system for continuous improvement.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/programs", tags=["programs"])

# Learning data storage (in production, use database)
LEARNING_DATA_PATH = Path("data/learning")
LEARNING_DATA_PATH.mkdir(parents=True, exist_ok=True)


# ==================
# DATA MODELS
# ==================

class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None


class LabValue(BaseModel):
    name: str
    value: str
    unit: Optional[str] = None
    flag: Optional[str] = None  # high, low, critical


class FunctionalStatus(BaseModel):
    mobility: Optional[str] = None  # independent, supervision, min assist, etc.
    transfers: Optional[str] = None
    balance: Optional[str] = None
    endurance: Optional[str] = None
    cognition: Optional[str] = None
    pain_level: Optional[int] = None  # 0-10


class ProgramRequest(BaseModel):
    """Request for rehab program generation."""
    # Minimal info (required)
    condition: str = Field(..., description="Primary condition or diagnosis")
    discipline: str = Field(default="PT", description="PT, OT, or SLP")

    # Basic info (improves personalization)
    age: Optional[int] = None
    sex: Optional[str] = None
    care_setting: Optional[str] = None  # outpatient, inpatient, home health, SNF

    # Clinical info (further personalization)
    diagnoses: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)
    surgical_history: Optional[str] = None
    onset_date: Optional[str] = None  # When condition started

    # Medical data (high personalization)
    medications: list[Medication] = Field(default_factory=list)
    labs: list[LabValue] = Field(default_factory=list)
    vitals: Optional[dict] = None  # BP, HR, SpO2, etc.

    # Functional data (highest personalization)
    functional_status: Optional[FunctionalStatus] = None
    prior_level_of_function: Optional[str] = None
    patient_goals: list[str] = Field(default_factory=list)
    barriers: list[str] = Field(default_factory=list)  # Transportation, cognition, etc.
    equipment_available: list[str] = Field(default_factory=list)

    # Preferences
    preferred_duration_weeks: Optional[int] = None
    sessions_per_week: Optional[int] = None


class Exercise(BaseModel):
    """Single exercise in the program."""
    name: str
    description: str
    purpose: str
    sets: int
    reps: str  # Can be "10-15" or "30 seconds"
    frequency: str
    progression: str
    precautions: list[str] = Field(default_factory=list)
    modifications: list[str] = Field(default_factory=list)
    evidence_level: Optional[str] = None


class ProgramPhase(BaseModel):
    """A phase of the rehab program."""
    phase_name: str
    duration: str
    goals: list[str]
    exercises: list[Exercise]
    criteria_to_progress: list[str]


class RehabProgram(BaseModel):
    """Complete rehabilitation program."""
    program_id: str
    generated_at: str
    personalization_level: str  # standard, moderate, high, individualized
    personalization_score: int  # 0-100

    # Program overview
    condition: str
    discipline: str
    program_duration: str
    frequency: str

    # Clinical summary
    clinical_rationale: str
    key_impairments: list[str]

    # Goals
    short_term_goals: list[str]
    long_term_goals: list[str]

    # Program phases
    phases: list[ProgramPhase]

    # Home program
    home_exercises: list[Exercise]

    # Safety
    precautions: list[str]
    contraindications: list[str]
    red_flags: list[str]

    # Outcomes
    outcome_measures: list[str]
    reassessment_schedule: str

    # Evidence
    evidence_summary: str
    key_references: list[str]

    # Learning
    feedback_id: str  # For submitting feedback


class ProgramFeedback(BaseModel):
    """Feedback for learning system."""
    feedback_id: str
    program_id: str
    rating: int = Field(..., ge=1, le=5)
    effectiveness: Optional[str] = None  # very_effective, effective, neutral, ineffective
    patient_adherence: Optional[str] = None  # excellent, good, fair, poor
    modifications_made: list[str] = Field(default_factory=list)
    outcomes_achieved: list[str] = Field(default_factory=list)
    comments: Optional[str] = None
    clinician_specialty: Optional[str] = None


class ProgramRefineRequest(BaseModel):
    """Request for refining a program based on follow-up feedback."""
    program_id: str
    section: str = Field(..., description="Section to refine: exercises, goals, home, safety")
    feedback: str = Field(..., description="User feedback or clarification")
    original_request: Optional[ProgramRequest] = None
    current_program: Optional[dict] = None


class ProgramRefineResponse(BaseModel):
    """Response from program refinement."""
    response: str
    updated_program: Optional[RehabProgram] = None
    modifications_made: list[str] = Field(default_factory=list)


# ==================
# KNOWLEDGE BASE
# ==================

CONDITION_PROTOCOLS = {
    "parkinson": {
        "key_impairments": ["Bradykinesia", "Rigidity", "Postural instability", "Gait dysfunction", "Freezing of gait"],
        "evidence_interventions": [
            {"name": "LSVT BIG", "evidence": "Level I", "description": "Amplitude-focused movement training"},
            {"name": "Treadmill training", "evidence": "Level I", "description": "Gait training with/without BWS"},
            {"name": "Tai Chi", "evidence": "Level I", "description": "Balance and falls prevention"},
            {"name": "Dual-task training", "evidence": "Level II", "description": "Cognitive-motor integration"},
            {"name": "Cueing strategies", "evidence": "Level I", "description": "Auditory/visual cues for movement"},
        ],
        "outcome_measures": ["TUG", "Mini-BESTest", "6MWT", "PDQ-39", "Freezing of Gait Questionnaire"],
        "precautions": ["Monitor for orthostatic hypotension", "Time sessions with medication cycle", "Watch for freezing triggers"],
    },
    "stroke": {
        "key_impairments": ["Hemiparesis", "Spasticity", "Balance deficits", "Coordination impairment", "Sensory deficits"],
        "evidence_interventions": [
            {"name": "Task-specific training", "evidence": "Level I", "description": "Repetitive practice of functional tasks"},
            {"name": "Constraint-induced movement therapy", "evidence": "Level I", "description": "For upper extremity recovery"},
            {"name": "Body weight supported treadmill", "evidence": "Level I", "description": "Gait retraining"},
            {"name": "Mirror therapy", "evidence": "Level I", "description": "Visual feedback for motor recovery"},
            {"name": "Electrical stimulation", "evidence": "Level I", "description": "NMES for motor recruitment"},
        ],
        "outcome_measures": ["Fugl-Meyer", "Berg Balance Scale", "10MWT", "FIM", "Stroke Impact Scale"],
        "precautions": ["Monitor blood pressure", "Watch for autonomic dysreflexia", "Assess fall risk"],
    },
    "total_knee": {
        "key_impairments": ["Pain", "Limited ROM", "Quadriceps weakness", "Gait dysfunction", "Swelling"],
        "evidence_interventions": [
            {"name": "Progressive ROM exercises", "evidence": "Level I", "description": "Restore knee flexion/extension"},
            {"name": "Quadriceps strengthening", "evidence": "Level I", "description": "Open and closed chain exercises"},
            {"name": "Gait training", "evidence": "Level I", "description": "Normalize gait pattern"},
            {"name": "Balance training", "evidence": "Level II", "description": "Proprioceptive retraining"},
            {"name": "Functional training", "evidence": "Level I", "description": "Stairs, transfers, ADLs"},
        ],
        "outcome_measures": ["Knee ROM", "Quad strength", "TUG", "KOOS", "6MWT"],
        "precautions": ["Follow weight bearing precautions", "Monitor for DVT signs", "Ice and elevation for swelling"],
    },
    "low_back_pain": {
        "key_impairments": ["Pain", "Limited mobility", "Core weakness", "Deconditioning", "Fear avoidance"],
        "evidence_interventions": [
            {"name": "Motor control exercises", "evidence": "Level I", "description": "Core stabilization training"},
            {"name": "Graded activity", "evidence": "Level I", "description": "Progressive return to function"},
            {"name": "McKenzie method", "evidence": "Level I", "description": "Directional preference exercises"},
            {"name": "Manual therapy", "evidence": "Level I", "description": "Joint mobilization, soft tissue work"},
            {"name": "Aerobic exercise", "evidence": "Level I", "description": "Walking, swimming, cycling"},
        ],
        "outcome_measures": ["ODI", "NPRS", "FABQ", "Lumbar ROM", "Repeated movements"],
        "precautions": ["Screen for red flags", "Avoid aggravating positions initially", "Progress gradually"],
    },
    "falls": {
        "key_impairments": ["Balance deficits", "Lower extremity weakness", "Gait abnormalities", "Fear of falling", "Sensory deficits"],
        "evidence_interventions": [
            {"name": "Otago Exercise Program", "evidence": "Level I", "description": "Strength and balance for fall prevention"},
            {"name": "Tai Chi", "evidence": "Level I", "description": "Balance and coordination"},
            {"name": "Perturbation training", "evidence": "Level I", "description": "Reactive balance training"},
            {"name": "Dual-task training", "evidence": "Level II", "description": "Cognitive-motor challenges"},
            {"name": "Gait training", "evidence": "Level I", "description": "Speed, endurance, obstacles"},
        ],
        "outcome_measures": ["Berg Balance Scale", "TUG", "30s Sit-to-Stand", "ABC Scale", "4-Stage Balance Test"],
        "precautions": ["Ensure safe environment", "Use gait belt", "Progress balance challenges carefully"],
    },
    "rotator_cuff": {
        "key_impairments": ["Pain", "Weakness", "Limited ROM", "Scapular dyskinesis", "Functional limitations"],
        "evidence_interventions": [
            {"name": "Rotator cuff strengthening", "evidence": "Level I", "description": "Progressive resistance exercises"},
            {"name": "Scapular stabilization", "evidence": "Level I", "description": "Periscapular muscle training"},
            {"name": "ROM exercises", "evidence": "Level I", "description": "PROM progressing to AROM"},
            {"name": "Manual therapy", "evidence": "Level II", "description": "Joint and soft tissue mobilization"},
            {"name": "Neuromuscular control", "evidence": "Level II", "description": "Proprioceptive training"},
        ],
        "outcome_measures": ["Shoulder ROM", "Strength testing", "DASH", "Penn Shoulder Score", "SPADI"],
        "precautions": ["Respect tissue healing timeline", "Avoid painful positions", "Progress gradually"],
    },
    "dysphagia": {
        "key_impairments": ["Swallowing dysfunction", "Aspiration risk", "Reduced oral control", "Delayed swallow trigger", "Residue"],
        "evidence_interventions": [
            {"name": "Mendelsohn maneuver", "evidence": "Level II", "description": "Prolonged laryngeal elevation"},
            {"name": "Effortful swallow", "evidence": "Level II", "description": "Increased posterior tongue base movement"},
            {"name": "Supraglottic swallow", "evidence": "Level II", "description": "Airway protection technique"},
            {"name": "Shaker exercise", "evidence": "Level I", "description": "UES opening exercise"},
            {"name": "NMES", "evidence": "Level II", "description": "Electrical stimulation for swallow muscles"},
        ],
        "outcome_measures": ["FOIS", "EAT-10", "MASA", "PAS", "Diet level"],
        "precautions": ["NPO if severe aspiration", "Appropriate diet texture", "Supervision during meals"],
    },
}

EXERCISE_LIBRARY = {
    "PT": {
        "strength": [
            {"name": "Squats", "description": "Stand with feet shoulder-width apart, lower hips back and down", "muscle_groups": ["Quadriceps", "Glutes", "Core"]},
            {"name": "Bridges", "description": "Lie on back, knees bent, lift hips toward ceiling", "muscle_groups": ["Glutes", "Hamstrings", "Core"]},
            {"name": "Clamshells", "description": "Side-lying, knees bent, open top knee while keeping feet together", "muscle_groups": ["Hip abductors", "Glutes"]},
            {"name": "Step-ups", "description": "Step up onto platform leading with involved leg", "muscle_groups": ["Quadriceps", "Glutes"]},
            {"name": "Heel raises", "description": "Rise up onto toes, hold, lower slowly", "muscle_groups": ["Gastrocnemius", "Soleus"]},
            {"name": "Straight leg raises", "description": "Lie on back, lift straight leg to 45 degrees", "muscle_groups": ["Quadriceps", "Hip flexors"]},
            {"name": "Side-lying hip abduction", "description": "Lie on side, lift top leg toward ceiling", "muscle_groups": ["Hip abductors", "Glutes"]},
            {"name": "Prone hip extension", "description": "Lie on stomach, lift leg toward ceiling", "muscle_groups": ["Glutes", "Hamstrings"]},
        ],
        "balance": [
            {"name": "Single leg stance", "description": "Stand on one leg, maintain balance", "progression": "Eyes closed, unstable surface"},
            {"name": "Tandem stance", "description": "Stand heel-to-toe, maintain balance", "progression": "Eyes closed, head turns"},
            {"name": "Weight shifts", "description": "Shift weight side to side and forward/back", "progression": "On foam, with perturbations"},
            {"name": "Heel-toe walking", "description": "Walk in straight line, heel touching toe", "progression": "Backward, with head turns"},
            {"name": "Standing on foam", "description": "Maintain balance on unstable surface", "progression": "Single leg, eyes closed"},
        ],
        "flexibility": [
            {"name": "Hamstring stretch", "description": "Seated or supine, extend knee, lean forward", "hold": "30 seconds"},
            {"name": "Quad stretch", "description": "Standing or prone, bend knee, hold ankle", "hold": "30 seconds"},
            {"name": "Hip flexor stretch", "description": "Half-kneeling, shift weight forward", "hold": "30 seconds"},
            {"name": "Calf stretch", "description": "Wall stretch, keep heel down", "hold": "30 seconds"},
            {"name": "Piriformis stretch", "description": "Supine figure-4 position, pull knee to chest", "hold": "30 seconds"},
        ],
        "aerobic": [
            {"name": "Walking program", "description": "Progressive walking at moderate intensity", "progression": "Increase duration and speed"},
            {"name": "Stationary cycling", "description": "Low-impact cardiovascular exercise", "progression": "Increase resistance and duration"},
            {"name": "Aquatic exercise", "description": "Pool-based cardiovascular training", "progression": "Increase speed and depth"},
        ],
    },
    "OT": {
        "fine_motor": [
            {"name": "Pegboard activities", "description": "Place and remove pegs of various sizes", "purpose": "Pinch strength and coordination"},
            {"name": "Theraputty exercises", "description": "Pinch, roll, and squeeze putty", "purpose": "Hand strength"},
            {"name": "Coin manipulation", "description": "Pick up and manipulate coins", "purpose": "In-hand manipulation"},
        ],
        "ADL": [
            {"name": "Dressing practice", "description": "Practice donning/doffing clothing", "purpose": "Independence in self-care"},
            {"name": "Meal preparation", "description": "Graded kitchen tasks", "purpose": "IADL independence"},
            {"name": "Grooming tasks", "description": "Practice hygiene activities", "purpose": "Self-care independence"},
        ],
    },
    "SLP": {
        "swallowing": [
            {"name": "Mendelsohn maneuver", "description": "Hold larynx elevated during swallow", "purpose": "Improve UES opening"},
            {"name": "Effortful swallow", "description": "Swallow with maximum effort", "purpose": "Improve bolus clearance"},
            {"name": "Shaker exercise", "description": "Head lifts in supine", "purpose": "Strengthen suprahyoid muscles"},
            {"name": "Masako maneuver", "description": "Swallow with tongue protruded", "purpose": "Strengthen pharyngeal wall"},
        ],
        "voice": [
            {"name": "LSVT LOUD exercises", "description": "High-effort phonation tasks", "purpose": "Increase vocal loudness"},
            {"name": "Pitch glides", "description": "Glide from low to high pitch", "purpose": "Improve vocal flexibility"},
        ],
    },
}

MEDICATION_CONSIDERATIONS = {
    "anticoagulants": {"exercise_consideration": "Avoid high fall risk activities, no contact sports", "monitoring": "Watch for bruising, bleeding"},
    "beta_blockers": {"exercise_consideration": "HR will not reflect true exertion, use RPE", "monitoring": "Monitor for hypotension"},
    "antihypertensives": {"exercise_consideration": "Risk of orthostatic hypotension", "monitoring": "Slow position changes"},
    "opioids": {"exercise_consideration": "May affect balance and cognition", "monitoring": "Fall risk, drowsiness"},
    "muscle_relaxants": {"exercise_consideration": "May cause drowsiness and weakness", "monitoring": "Balance, alertness"},
    "diuretics": {"exercise_consideration": "Risk of dehydration and electrolyte imbalance", "monitoring": "Hydration, cramping"},
    "insulin": {"exercise_consideration": "Risk of hypoglycemia with exercise", "monitoring": "Blood glucose before/after"},
    "steroids": {"exercise_consideration": "Risk of tendon weakness, osteoporosis", "monitoring": "Avoid high-impact if chronic use"},
}

# Comprehensive comorbidity considerations for rehab
COMORBIDITY_CONSIDERATIONS = {
    "diabetes": {
        "keywords": ["diabetes", "diabetic", "dm", "dm2", "dm1", "type 2", "type 1", "a1c", "blood sugar"],
        "exercise_modifications": [
            "Check blood glucose before and after exercise",
            "Have fast-acting glucose available (juice, glucose tabs)",
            "Avoid exercise if BG >250 mg/dL or <100 mg/dL",
            "Inspect feet before and after weight-bearing exercise",
            "Prefer non-weight bearing cardio if peripheral neuropathy present",
        ],
        "precautions": [
            "Risk of hypoglycemia with prolonged exercise",
            "Delayed wound healing - protect residual limb/incisions",
            "Peripheral neuropathy may mask pain/injury",
            "Autonomic neuropathy may affect HR response",
        ],
        "monitoring": [
            "Blood glucose before/after exercise",
            "Daily foot inspection",
            "Signs of hypoglycemia: shakiness, confusion, sweating",
            "Wound healing progress",
        ],
        "red_flags": [
            "BG <70 mg/dL - treat hypoglycemia before continuing",
            "BG >300 mg/dL with ketones - no exercise",
            "New foot wound or infection",
            "Signs of diabetic ketoacidosis",
        ],
    },
    "chf": {
        "keywords": ["chf", "heart failure", "hf", "ef", "ejection fraction", "cardiomyopathy", "lvef"],
        "exercise_modifications": [
            "Use RPE scale (12-14/20) rather than HR targets",
            "Interval training with rest breaks",
            "Avoid Valsalva maneuver - exhale on exertion",
            "Upper extremity exercise may be better tolerated",
            "Seated or recumbent positions as needed",
        ],
        "precautions": [
            "Fluid restriction may apply - confirm hydration guidelines",
            "Weight gain >2-3 lbs/day indicates fluid retention",
            "Avoid exercise during acute exacerbations",
            "May have activity restrictions from cardiologist",
        ],
        "monitoring": [
            "Daily weight - report gain >2-3 lbs",
            "Dyspnea scale before/during/after",
            "Peripheral edema",
            "Oxygen saturation if available",
            "RPE - stop if >15/20",
        ],
        "red_flags": [
            "Increasing shortness of breath at rest",
            "New or worsening edema",
            "Chest pain or pressure",
            "Weight gain >3 lbs in 1 day or 5 lbs in 1 week",
            "Oxygen saturation <90%",
        ],
    },
    "urinary_incontinence": {
        "keywords": ["incontinence", "urine", "bladder", "urinary", "leakage", "catheter", "foley"],
        "exercise_modifications": [
            "Include pelvic floor exercises (Kegels) in program",
            "Schedule bathroom breaks before exercise sessions",
            "Avoid high-impact exercises that increase intra-abdominal pressure",
            "Modify exercises to reduce bearing down",
            "Consider timing of fluid intake around exercise",
        ],
        "precautions": [
            "May need protective undergarments during exercise",
            "Ensure bathroom accessibility during sessions",
            "Avoid exercises that worsen symptoms",
            "If catheterized - secure catheter during mobility",
        ],
        "monitoring": [
            "Frequency and severity of leakage episodes",
            "Bladder diary if appropriate",
            "Pelvic floor strength progression",
        ],
        "red_flags": [
            "New onset incontinence - needs medical evaluation",
            "Blood in urine",
            "Pain with urination",
            "Signs of UTI (fever, confusion, foul odor)",
        ],
    },
    "copd": {
        "keywords": ["copd", "emphysema", "chronic bronchitis", "pulmonary", "lung disease", "oxygen"],
        "exercise_modifications": [
            "Pursed-lip breathing during exertion",
            "Rest breaks as needed - don't push through dyspnea",
            "Supplemental O2 as prescribed during exercise",
            "Coordinate breathing with movement",
            "Upper extremity exercises may increase dyspnea",
        ],
        "precautions": [
            "Avoid exercise during acute exacerbations",
            "Temperature extremes may worsen symptoms",
            "Air pollution/allergens may trigger symptoms",
        ],
        "monitoring": [
            "Dyspnea scale (Borg) - keep at 3-4/10",
            "Oxygen saturation - maintain >88-90%",
            "Recovery time to baseline breathing",
            "Sputum changes",
        ],
        "red_flags": [
            "SpO2 <88% that doesn't recover with rest",
            "Severe dyspnea at rest",
            "Change in sputum color/amount",
            "Fever or increased cough",
        ],
    },
    "hypertension": {
        "keywords": ["hypertension", "htn", "high blood pressure", "bp", "blood pressure"],
        "exercise_modifications": [
            "Avoid heavy resistance with Valsalva",
            "Gradual warm-up and cool-down",
            "Avoid rapid position changes",
            "Moderate intensity aerobic preferred",
        ],
        "precautions": [
            "Check BP before exercise if symptomatic",
            "Avoid exercise if SBP >180 or DBP >110",
            "Stay hydrated",
        ],
        "monitoring": [
            "Pre/post BP if available",
            "Symptoms: headache, dizziness, visual changes",
        ],
        "red_flags": [
            "BP >180/110 mmHg",
            "Severe headache during exercise",
            "Chest pain or shortness of breath",
            "Visual disturbances",
        ],
    },
    "substance_abuse": {
        "keywords": ["alcohol", "drug abuse", "substance", "addiction", "etoh", "ivdu", "opioid use"],
        "exercise_modifications": [
            "Schedule sessions for optimal alertness",
            "Verify sobriety before sessions requiring balance/coordination",
            "May need modified intensity during withdrawal/recovery",
            "Include stress-reduction techniques",
        ],
        "precautions": [
            "Increased fall risk if under influence",
            "May affect medication compliance",
            "Nutritional deficiencies may impact healing",
            "Liver function may affect medication metabolism",
        ],
        "monitoring": [
            "Alertness and cognition during sessions",
            "Compliance with exercise program",
            "Signs of withdrawal or relapse",
        ],
        "red_flags": [
            "Signs of acute intoxication - defer exercise",
            "Severe withdrawal symptoms",
            "Non-compliance with safety precautions",
        ],
    },
    "obesity": {
        "keywords": ["obesity", "obese", "bmi", "overweight", "morbid obesity"],
        "exercise_modifications": [
            "Start with low-impact activities",
            "Seated or aquatic exercises may be preferred",
            "Shorter, more frequent sessions",
            "Gradual progression of duration before intensity",
        ],
        "precautions": [
            "Joint protection strategies",
            "Heat intolerance - cool environment",
            "May need bariatric equipment",
        ],
        "monitoring": [
            "Comfort and tolerance",
            "Joint pain",
            "Skin integrity in skin folds",
        ],
        "red_flags": [
            "Chest pain or severe dyspnea",
            "Joint pain that worsens with exercise",
        ],
    },
    "osteoporosis": {
        "keywords": ["osteoporosis", "osteopenia", "bone density", "dexa", "fracture risk"],
        "exercise_modifications": [
            "Avoid flexion-based exercises (sit-ups, toe touches)",
            "No high-impact activities",
            "Focus on weight-bearing as tolerated and balance",
            "Resistance training for bone health",
        ],
        "precautions": [
            "Fall prevention is critical",
            "Avoid twisting or bending spine",
            "Gentle progression of resistance",
        ],
        "monitoring": [
            "Pain with activity",
            "Balance and fall risk",
        ],
        "red_flags": [
            "New onset back pain - possible compression fracture",
            "Pain after minor trauma",
        ],
    },
    "dementia": {
        "keywords": ["dementia", "alzheimer", "cognitive impairment", "memory", "confusion"],
        "exercise_modifications": [
            "Simple, repetitive exercises",
            "Visual demonstrations over verbal instructions",
            "Consistent routine and environment",
            "One-step commands",
            "Caregiver involvement essential",
        ],
        "precautions": [
            "Supervision required",
            "May not report pain accurately",
            "Wandering risk - secure environment",
        ],
        "monitoring": [
            "Understanding of instructions",
            "Safety awareness",
            "Non-verbal signs of distress",
        ],
        "red_flags": [
            "Acute change in cognition (delirium)",
            "Agitation or combativeness",
            "Signs of pain (grimacing, guarding)",
        ],
    },
}


# ==================
# HELPER FUNCTIONS
# ==================

def calculate_personalization_score(request: ProgramRequest) -> tuple[int, str]:
    """Calculate how personalized the program can be based on available data."""
    score = 0

    # Minimal info (always present)
    score += 10  # condition
    score += 5   # discipline

    # Basic info
    if request.age: score += 10
    if request.sex: score += 5
    if request.care_setting: score += 5

    # Clinical info
    if request.diagnoses: score += min(len(request.diagnoses) * 5, 15)
    if request.comorbidities: score += min(len(request.comorbidities) * 5, 15)
    if request.surgical_history: score += 5
    if request.onset_date: score += 5

    # Medical data
    if request.medications: score += min(len(request.medications) * 3, 15)
    if request.labs: score += min(len(request.labs) * 3, 10)
    if request.vitals: score += 5

    # Functional data (highest value)
    if request.functional_status:
        if request.functional_status.mobility: score += 5
        if request.functional_status.transfers: score += 5
        if request.functional_status.balance: score += 5
        if request.functional_status.pain_level is not None: score += 5
    if request.prior_level_of_function: score += 10
    if request.patient_goals: score += min(len(request.patient_goals) * 5, 15)
    if request.barriers: score += min(len(request.barriers) * 3, 10)
    if request.equipment_available: score += 5

    # Cap at 100
    score = min(score, 100)

    # Determine level
    if score < 25:
        level = "standard"
    elif score < 50:
        level = "moderate"
    elif score < 75:
        level = "high"
    else:
        level = "individualized"

    return score, level


def find_matching_protocol(condition: str) -> dict:
    """Find the best matching protocol for a condition."""
    condition_lower = condition.lower()

    for key, protocol in CONDITION_PROTOCOLS.items():
        if key in condition_lower:
            return protocol

    # Check for common variations
    if any(term in condition_lower for term in ["tka", "tkr", "knee replacement", "knee arthroplasty"]):
        return CONDITION_PROTOCOLS["total_knee"]
    if any(term in condition_lower for term in ["tha", "thr", "hip replacement", "hip arthroplasty"]):
        return CONDITION_PROTOCOLS.get("total_hip", CONDITION_PROTOCOLS["total_knee"])
    if any(term in condition_lower for term in ["cva", "cerebrovascular", "hemiplegia", "hemiparesis"]):
        return CONDITION_PROTOCOLS["stroke"]
    if any(term in condition_lower for term in ["pd", "parkinson"]):
        return CONDITION_PROTOCOLS["parkinson"]
    if any(term in condition_lower for term in ["lbp", "lumbar", "back pain", "sciatica"]):
        return CONDITION_PROTOCOLS["low_back_pain"]
    if any(term in condition_lower for term in ["fall", "balance", "unsteady"]):
        return CONDITION_PROTOCOLS["falls"]
    if any(term in condition_lower for term in ["shoulder", "rotator", "rct"]):
        return CONDITION_PROTOCOLS["rotator_cuff"]
    if any(term in condition_lower for term in ["swallow", "dysphagia", "aspiration"]):
        return CONDITION_PROTOCOLS["dysphagia"]

    # Default protocol
    return {
        "key_impairments": ["Functional limitations", "Strength deficits", "Mobility impairment"],
        "evidence_interventions": [
            {"name": "Therapeutic exercise", "evidence": "Level I", "description": "Progressive strengthening"},
            {"name": "Functional training", "evidence": "Level I", "description": "Task-specific practice"},
            {"name": "Manual therapy", "evidence": "Level II", "description": "Hands-on techniques as indicated"},
        ],
        "outcome_measures": ["Patient-specific functional scale", "NPRS", "Functional assessments"],
        "precautions": ["Progress gradually", "Monitor symptoms", "Respect pain limits"],
    }


def get_medication_precautions(medications: list[Medication]) -> list[str]:
    """Get exercise precautions based on medications."""
    precautions = []

    med_categories = {
        "anticoagulants": ["warfarin", "coumadin", "eliquis", "xarelto", "pradaxa", "heparin", "lovenox"],
        "beta_blockers": ["metoprolol", "atenolol", "carvedilol", "propranolol", "bisoprolol"],
        "antihypertensives": ["lisinopril", "amlodipine", "losartan", "hydrochlorothiazide", "hctz"],
        "opioids": ["oxycodone", "hydrocodone", "morphine", "tramadol", "fentanyl", "percocet", "vicodin"],
        "muscle_relaxants": ["flexeril", "cyclobenzaprine", "baclofen", "tizanidine", "robaxin"],
        "diuretics": ["lasix", "furosemide", "hydrochlorothiazide", "spironolactone"],
        "insulin": ["insulin", "lantus", "humalog", "novolog"],
        "steroids": ["prednisone", "methylprednisolone", "dexamethasone"],
    }

    for med in medications:
        med_name = med.name.lower()
        for category, drugs in med_categories.items():
            if any(drug in med_name for drug in drugs):
                considerations = MEDICATION_CONSIDERATIONS.get(category, {})
                if considerations.get("exercise_consideration"):
                    precautions.append(f"{med.name}: {considerations['exercise_consideration']}")
                break

    return precautions


def generate_program_id() -> str:
    """Generate unique program ID."""
    timestamp = datetime.now(timezone.utc).isoformat()
    return hashlib.sha256(timestamp.encode()).hexdigest()[:12]


def build_exercises_for_phase(
    phase: str,
    discipline: str,
    protocol: dict,
    personalization_level: str,
    functional_status: Optional[FunctionalStatus],
    age: Optional[int],
) -> list[Exercise]:
    """Build exercise list for a program phase."""
    exercises = []
    library = EXERCISE_LIBRARY.get(discipline, EXERCISE_LIBRARY["PT"])

    # Determine intensity based on phase and personalization
    if phase == "initial":
        intensity = "low"
        sets = 2
        reps = "8-10"
    elif phase == "intermediate":
        intensity = "moderate"
        sets = 3
        reps = "10-15"
    else:  # advanced
        intensity = "high"
        sets = 3
        reps = "12-15"

    # Adjust for age
    if age and age > 75:
        sets = max(1, sets - 1)
        reps = "8-10"

    # Adjust for functional status
    if functional_status and functional_status.pain_level and functional_status.pain_level > 5:
        intensity = "low"
        sets = 2

    # Select exercises based on protocol
    if discipline == "PT":
        # Strength exercises
        for ex in library.get("strength", [])[:4]:
            exercises.append(Exercise(
                name=ex["name"],
                description=ex["description"],
                purpose=f"Strengthen {', '.join(ex.get('muscle_groups', ['targeted muscles']))}",
                sets=sets,
                reps=reps,
                frequency="3x/week",
                progression=f"Increase resistance when {reps} reps achieved easily",
                precautions=protocol.get("precautions", [])[:2],
                evidence_level="Level I-II",
            ))

        # Balance exercises
        for ex in library.get("balance", [])[:2]:
            exercises.append(Exercise(
                name=ex["name"],
                description=ex["description"],
                purpose="Improve balance and reduce fall risk",
                sets=3,
                reps="30 seconds",
                frequency="Daily",
                progression=ex.get("progression", "Progress to more challenging surfaces"),
                precautions=["Ensure safety support nearby"],
                evidence_level="Level I",
            ))

        # Flexibility
        for ex in library.get("flexibility", [])[:3]:
            exercises.append(Exercise(
                name=ex["name"],
                description=ex["description"],
                purpose="Improve flexibility and reduce muscle tension",
                sets=2,
                reps=ex.get("hold", "30 seconds"),
                frequency="Daily",
                progression="Increase hold time to 60 seconds",
                precautions=["No bouncing", "Stretch to mild tension only"],
                evidence_level="Level I",
            ))

    elif discipline == "SLP":
        for ex in library.get("swallowing", [])[:3]:
            exercises.append(Exercise(
                name=ex["name"],
                description=ex["description"],
                purpose=ex.get("purpose", "Improve swallowing function"),
                sets=3,
                reps="10 repetitions",
                frequency="3x/day",
                progression="Increase repetitions as tolerated",
                precautions=["Stop if coughing or discomfort"],
                evidence_level="Level II",
            ))

    return exercises


def apply_learning_adjustments(program: dict, condition: str) -> dict:
    """Apply adjustments based on learning data from previous programs."""
    learning_file = LEARNING_DATA_PATH / "feedback_summary.json"

    if not learning_file.exists():
        return program

    try:
        with open(learning_file) as f:
            learning_data = json.load(f)

        # Get insights for this condition
        condition_key = condition.lower().replace(" ", "_")
        insights = learning_data.get("condition_insights", {}).get(condition_key, {})

        if insights:
            # Apply successful modifications
            if insights.get("top_modifications"):
                program["learning_applied"] = insights["top_modifications"][:3]

            # Adjust based on effectiveness data
            if insights.get("avg_effectiveness", 0) < 3:
                program["clinical_rationale"] += " Note: Consider supplementary interventions based on prior outcomes."

    except (json.JSONDecodeError, KeyError):
        pass

    return program


# ==================
# API ENDPOINTS
# ==================

@router.post("/generate", response_model=RehabProgram)
async def generate_program(request: ProgramRequest):
    """Generate a personalized rehabilitation program.

    The program scales from standard to highly individualized based on
    the amount and quality of patient data provided.

    Minimal input (condition only) → Standard evidence-based protocol
    Full input (all fields) → Highly individualized program
    """
    # Calculate personalization
    score, level = calculate_personalization_score(request)

    # Find matching protocol
    protocol = find_matching_protocol(request.condition)

    # Get medication precautions
    med_precautions = get_medication_precautions(request.medications) if request.medications else []

    # Determine program parameters
    if request.sessions_per_week:
        frequency = f"{request.sessions_per_week}x/week"
    elif request.care_setting == "inpatient":
        frequency = "Daily"
    elif request.care_setting == "acute":
        frequency = "1-2x/day"
    elif request.care_setting == "snf":
        frequency = "5x/week"
    else:
        frequency = "2-3x/week"

    duration_weeks = request.preferred_duration_weeks or (
        12 if request.care_setting == "outpatient" else
        4 if request.care_setting == "inpatient" else
        8
    )

    # Build clinical rationale
    rationale_parts = [f"Evidence-based rehabilitation program for {request.condition}."]
    if request.age:
        rationale_parts.append(f"Patient is {request.age} years old.")
    if request.comorbidities:
        rationale_parts.append(f"Program modified for comorbidities: {', '.join(request.comorbidities[:3])}.")
    if level == "individualized" and request.patient_goals:
        rationale_parts.append(f"Tailored to patient goals: {', '.join(request.patient_goals[:2])}.")

    # Generate goals
    short_term_goals = [
        f"Reduce pain/symptoms by 50% within 2-4 weeks",
        f"Improve {protocol['key_impairments'][0].lower()} within 2 weeks",
        "Establish home exercise program independence",
    ]

    long_term_goals = [
        f"Return to prior level of function",
        f"Achieve independence in functional activities",
        "Prevent recurrence/progression",
    ]

    if request.patient_goals:
        long_term_goals = request.patient_goals[:3] + long_term_goals[len(request.patient_goals):3]

    # Build program phases
    phases = []

    # Initial phase
    phases.append(ProgramPhase(
        phase_name="Initial/Acute Phase",
        duration="Weeks 1-2" if duration_weeks >= 4 else "Week 1",
        goals=[
            "Reduce pain and inflammation",
            "Establish baseline function",
            "Begin gentle mobility exercises",
        ],
        exercises=build_exercises_for_phase(
            "initial", request.discipline, protocol,
            level, request.functional_status, request.age
        ),
        criteria_to_progress=[
            "Pain controlled (≤4/10)",
            "Tolerating initial exercises",
            "No adverse reactions",
        ],
    ))

    # Intermediate phase
    if duration_weeks >= 4:
        phases.append(ProgramPhase(
            phase_name="Intermediate/Strengthening Phase",
            duration=f"Weeks 3-{min(duration_weeks-2, 8)}",
            goals=[
                "Progressive strengthening",
                "Improve functional mobility",
                "Advance balance training",
            ],
            exercises=build_exercises_for_phase(
                "intermediate", request.discipline, protocol,
                level, request.functional_status, request.age
            ),
            criteria_to_progress=[
                "Strength improved by 20%",
                "Functional goals 50% achieved",
                "Ready for advanced activities",
            ],
        ))

    # Advanced phase
    if duration_weeks >= 8:
        phases.append(ProgramPhase(
            phase_name="Advanced/Return to Function Phase",
            duration=f"Weeks {min(duration_weeks-2, 9)}-{duration_weeks}",
            goals=[
                "Sport/activity-specific training" if request.prior_level_of_function else "Advanced functional training",
                "Maximize strength and endurance",
                "Establish maintenance program",
            ],
            exercises=build_exercises_for_phase(
                "advanced", request.discipline, protocol,
                level, request.functional_status, request.age
            ),
            criteria_to_progress=[
                "Functional goals achieved",
                "Independent with HEP",
                "Ready for discharge",
            ],
        ))

    # Build home exercises (subset of phase 1-2)
    home_exercises = phases[0].exercises[:4] if phases else []

    # Compile precautions
    all_precautions = protocol.get("precautions", []) + med_precautions
    if request.age and request.age > 70:
        all_precautions.append("Monitor for fatigue in older adult")
    if request.barriers:
        if "cognition" in str(request.barriers).lower():
            all_precautions.append("Provide written instructions, involve caregiver")

    # Generate IDs
    program_id = generate_program_id()
    feedback_id = generate_program_id()

    # Build program
    program = RehabProgram(
        program_id=program_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        personalization_level=level,
        personalization_score=score,
        condition=request.condition,
        discipline=request.discipline,
        program_duration=f"{duration_weeks} weeks",
        frequency=frequency,
        clinical_rationale=" ".join(rationale_parts),
        key_impairments=protocol.get("key_impairments", []),
        short_term_goals=short_term_goals,
        long_term_goals=long_term_goals,
        phases=phases,
        home_exercises=home_exercises,
        precautions=all_precautions[:8],
        contraindications=protocol.get("contraindications", []),
        red_flags=[
            "Sudden increase in pain or swelling",
            "New neurological symptoms",
            "Signs of infection (fever, redness, warmth)",
            "Chest pain or shortness of breath with activity",
        ],
        outcome_measures=protocol.get("outcome_measures", ["Patient-reported outcomes"]),
        reassessment_schedule=f"Every {duration_weeks // 3 or 2} weeks, or with significant change",
        evidence_summary=f"Program based on current clinical practice guidelines and Level I-II evidence for {request.condition}.",
        key_references=[
            f"Clinical Practice Guidelines for {request.condition}",
            "American Physical Therapy Association recommendations",
            "Cochrane systematic reviews",
        ],
        feedback_id=feedback_id,
    )

    # Apply learning adjustments
    program_dict = program.model_dump()
    program_dict = apply_learning_adjustments(program_dict, request.condition)

    return RehabProgram(**program_dict)


def detect_comorbidities(text: str) -> list[dict]:
    """Detect comorbidities from text and return their considerations."""
    text_lower = text.lower()
    detected = []

    for condition, data in COMORBIDITY_CONSIDERATIONS.items():
        for keyword in data["keywords"]:
            if keyword in text_lower:
                detected.append({
                    "condition": condition,
                    "data": data,
                })
                break

    return detected


def build_comorbidity_response(detected_comorbidities: list[dict], section: str) -> str:
    """Build a comprehensive response based on detected comorbidities."""
    if not detected_comorbidities:
        return ""

    response_parts = []

    for item in detected_comorbidities:
        condition = item["condition"].replace("_", " ").title()
        data = item["data"]

        if section == "exercises":
            response_parts.append(f"\n**{condition} - Exercise Modifications:**")
            for mod in data["exercise_modifications"][:3]:
                response_parts.append(f"• {mod}")

        elif section == "safety":
            response_parts.append(f"\n**{condition} - Precautions:**")
            for prec in data["precautions"][:3]:
                response_parts.append(f"• {prec}")
            response_parts.append(f"\n**{condition} - Red Flags:**")
            for rf in data["red_flags"][:3]:
                response_parts.append(f"⚠️ {rf}")

        elif section == "goals":
            response_parts.append(f"\n**{condition} - Monitoring Required:**")
            for mon in data["monitoring"][:3]:
                response_parts.append(f"• {mon}")

        else:  # home or other
            response_parts.append(f"\n**{condition} Considerations:**")
            for mod in data["exercise_modifications"][:2]:
                response_parts.append(f"• {mod}")
            for prec in data["precautions"][:2]:
                response_parts.append(f"• {prec}")

    return "\n".join(response_parts)


@router.post("/refine", response_model=ProgramRefineResponse)
async def refine_program(request: Request, refine_request: ProgramRefineRequest):
    """Refine a program based on follow-up feedback.

    This endpoint processes user feedback about specific sections and
    returns updated recommendations with explanations.
    """
    section = refine_request.section
    feedback = refine_request.feedback
    current_program = refine_request.current_program
    original_request = refine_request.original_request

    modifications_made = []
    response_text = ""
    updated_program = None

    # Analyze the feedback for condition-specific modifications
    feedback_lower = feedback.lower()

    # Build comprehensive context from all sources
    condition_text = ""
    comorbidities_text = ""
    if current_program:
        condition_text = current_program.get("condition", "").lower()
    if original_request:
        if isinstance(original_request, dict):
            comorbidities_text = " ".join(original_request.get("comorbidities", []))
        elif hasattr(original_request, "comorbidities"):
            comorbidities_text = " ".join(original_request.comorbidities or [])

    combined_context = f"{feedback_lower} {condition_text} {comorbidities_text}"

    # Detect comorbidities from all context
    detected_comorbidities = detect_comorbidities(combined_context)

    # Detect amputation details
    is_bka = any(term in combined_context for term in ["bka", "below knee", "transtibial"])
    is_aka = any(term in combined_context for term in ["aka", "above knee", "transfemoral"])
    is_left = any(term in combined_context for term in ["left", "l bka", "l aka", " l "])
    is_right = any(term in combined_context for term in ["right", "r bka", "r aka", " r "])
    has_prosthesis = "with prosthesis" in combined_context or "prosthetic" in combined_context
    no_prosthesis = any(term in combined_context for term in [
        "without prosthesis", "no prosthesis", "not fitted", "no prosthetic"
    ])
    is_amputation = is_bka or is_aka or "amput" in combined_context

    # Detect specific exercise in feedback
    exercise_mentioned = None
    exercise_lower_names = ["squats", "lunges", "step-ups", "heel raises", "single leg stance",
                           "tandem stance", "heel-toe walking", "walking program"]
    for ex_name in exercise_lower_names:
        if ex_name in feedback_lower:
            exercise_mentioned = ex_name.title()
            break

    # Build side description
    side_desc = "left" if is_left else ("right" if is_right else "")
    amp_type = "BKA" if is_bka else ("AKA" if is_aka else "")

    # Pre-prosthetic exercise alternatives
    AMPUTATION_ALTERNATIVES = {
        "Squats": {
            "replacement": "Seated Knee Extensions (Intact LE)",
            "description": "Seated in wheelchair or sturdy chair, extend knee against gravity or resistance band",
            "rationale": "Maintains quad strength on intact limb without requiring bilateral stance",
            "sets": 3, "reps": "10-15",
        },
        "Lunges": {
            "replacement": "Single Leg Press (Intact LE)",
            "description": "Using resistance band or leg press machine, perform single leg press with intact limb",
            "rationale": "Strengthens intact limb without requiring bilateral weight bearing",
            "sets": 3, "reps": "10-12",
        },
        "Step-ups": {
            "replacement": "Seated Hip Flexion",
            "description": "Seated, lift knee toward chest against resistance",
            "rationale": "Hip strengthening without standing balance requirement",
            "sets": 3, "reps": "10-15",
        },
        "Heel raises": {
            "replacement": "Seated Ankle Pumps",
            "description": "Seated with foot on ground, perform ankle dorsiflexion/plantarflexion",
            "rationale": "Maintains ankle mobility and calf activation on intact side",
            "sets": 2, "reps": "20",
        },
    }

    RESIDUAL_LIMB_EXERCISES = [
        {"name": "Residual Limb ROM", "description": "Gentle knee flexion/extension within pain-free range"},
        {"name": "Hip Strengthening (Residual Side)", "description": "Sidelying hip abduction, prone hip extension"},
        {"name": "Desensitization", "description": "Gentle massage and tapping with various textures"},
    ]

    # Generate response based on section
    if section == "exercises":
        if is_amputation and exercise_mentioned and (no_prosthesis or not has_prosthesis):
            alt = AMPUTATION_ALTERNATIVES.get(exercise_mentioned)
            if alt:
                response_text = (
                    f"**Exercise Modified for {side_desc.upper()} {amp_type} (Pre-Prosthetic Phase)**\n\n"
                    f"❌ **Removed:** {exercise_mentioned}\n"
                    f"*Not appropriate - requires bilateral stance which patient cannot perform*\n\n"
                    f"✅ **Replacement:** {alt['replacement']}\n"
                    f"*{alt['description']}*\n\n"
                    f"**Rationale:** {alt['rationale']}\n"
                    f"**Dosage:** {alt['sets']} sets × {alt['reps']}\n\n"
                    f"**Additional Exercises for Pre-Prosthetic Phase:**\n"
                )
                for ex in RESIDUAL_LIMB_EXERCISES[:2]:
                    response_text += f"• **{ex['name']}** - {ex['description']}\n"
                modifications_made.append(f"Replaced {exercise_mentioned} with {alt['replacement']}")
            else:
                response_text = (
                    f"**{exercise_mentioned}** is not appropriate for {side_desc} {amp_type} without prosthesis.\n\n"
                    "**Recommended alternatives:**\n"
                    "• Seated resistance exercises for intact limb\n"
                    "• Residual limb conditioning exercises\n"
                    "• Upper body and core strengthening\n"
                    "• Wheelchair mobility training"
                )
                modifications_made.append(f"Flagged {exercise_mentioned}")
        elif is_amputation and (no_prosthesis or not has_prosthesis):
            response_text = (
                f"**Program for {side_desc.upper()} {amp_type} Pre-Prosthetic Phase**\n\n"
                "✅ **Appropriate exercises:**\n"
                "• Seated/supported single leg exercises\n"
                "• Residual limb ROM and conditioning\n"
                "• Core stability (seated/supine)\n"
                "• Upper body strengthening\n"
                "• Transfer training\n\n"
                "❌ **Avoid until prosthetic fitting:**\n"
                "• Bilateral standing exercises (squats, lunges)\n"
                "• Standing balance exercises\n"
                "• Gait training"
            )
            modifications_made.append("Pre-prosthetic phase modifications applied")
        else:
            response_text = "Please specify which exercise needs modification and the patient's specific limitation."

    elif section == "goals":
        if is_amputation and (no_prosthesis or not has_prosthesis):
            response_text = (
                f"**Goals for {side_desc.upper()} {amp_type} Pre-Prosthetic Phase**\n\n"
                "**Short-term (2-4 weeks):**\n"
                "• Independence with bed mobility\n"
                "• Transfers bed↔wheelchair with supervision\n"
                "• Residual limb ROM maintained\n\n"
                "**Long-term (6-8 weeks):**\n"
                "• Independent transfers all surfaces\n"
                "• Wheelchair mobility community distances\n"
                "• Ready for prosthetic fitting"
            )
            modifications_made.append("Goals adapted for pre-prosthetic phase")
        else:
            response_text = "Goals will be adjusted based on your feedback."

    elif section == "home":
        if is_amputation and (no_prosthesis or not has_prosthesis):
            response_text = (
                f"**Home Program for {side_desc.upper()} {amp_type} (No Prosthesis)**\n\n"
                "**Daily:**\n"
                "• Residual limb ROM - 10 reps, 2x/day\n"
                "• Prone lying - 20-30 min/day\n"
                "• Seated knee extension - 3×15\n"
                "• Upper body theraband - 3×15\n\n"
                "**Precautions:**\n"
                "• Do NOT hang residual limb over bed edge\n"
                "• Lock wheelchair brakes for all transfers"
            )
            modifications_made.append("Home program for pre-prosthetic phase")
        else:
            response_text = "Home exercises will be adjusted for patient's environment."

    elif section == "safety":
        if is_amputation:
            response_text = (
                f"**Safety for {side_desc.upper()} {amp_type}**\n\n"
                "**Fall Prevention:**\n"
                "• Lock brakes before ALL transfers\n"
                "• Use transfer board\n"
                "• Clear pathways\n\n"
                "**Residual Limb:**\n"
                "• Daily inspection\n"
                "• Report: redness, drainage, fever"
            )
            modifications_made.append("Safety precautions for amputation")
        else:
            response_text = "Safety precautions reviewed."

    else:
        response_text = f"Thank you for the details about {section}."

    # Add comorbidity-specific information
    if detected_comorbidities:
        comorbidity_response = build_comorbidity_response(detected_comorbidities, section)
        if comorbidity_response:
            response_text += f"\n\n---\n**Comorbidity Considerations:**{comorbidity_response}"
            for c in detected_comorbidities:
                modifications_made.append(f"Added {c['condition'].replace('_', ' ').title()} considerations")

    return ProgramRefineResponse(
        response=response_text,
        updated_program=updated_program,
        modifications_made=modifications_made,
    )


def update_program_exercise(current_program: dict, old_exercise: str, new_exercise: dict, additional_exercises: list) -> RehabProgram:
    """Update program by replacing an exercise and adding residual limb exercises."""
    import copy
    updated = copy.deepcopy(current_program)

    # Replace the exercise in phases
    for phase in updated.get("phases", []):
        new_exercises = []
        for ex in phase.get("exercises", []):
            if ex.get("name", "").lower() == old_exercise.lower():
                # Replace with new exercise
                new_exercises.append({
                    "name": new_exercise["replacement"],
                    "description": new_exercise["description"],
                    "purpose": new_exercise["rationale"],
                    "sets": new_exercise["sets"],
                    "reps": new_exercise["reps"],
                    "frequency": "3x/week",
                    "progression": "Increase resistance as tolerated",
                    "precautions": ["Maintain proper form", "Stop if pain"],
                    "modifications": [],
                    "evidence_level": "Level II",
                })
            else:
                new_exercises.append(ex)
        phase["exercises"] = new_exercises

    # Add residual limb exercises to first phase
    if updated.get("phases") and additional_exercises:
        for add_ex in additional_exercises[:2]:
            updated["phases"][0]["exercises"].append({
                "name": add_ex["name"],
                "description": add_ex["description"],
                "purpose": add_ex["purpose"],
                "sets": add_ex["sets"],
                "reps": add_ex["reps"],
                "frequency": "Daily",
                "progression": "Progress as tolerated",
                "precautions": add_ex.get("precautions", []),
                "modifications": [],
                "evidence_level": "Level II",
            })

    try:
        return RehabProgram(**updated)
    except Exception:
        return None


@router.post("/feedback")
async def submit_feedback(feedback: ProgramFeedback):
    """Submit feedback on a generated program for continuous learning.

    This feedback is used to improve future program generation.
    """
    # Store feedback
    feedback_file = LEARNING_DATA_PATH / "feedback.jsonl"

    with open(feedback_file, "a") as f:
        feedback_data = feedback.model_dump()
        feedback_data["submitted_at"] = datetime.now(timezone.utc).isoformat()
        f.write(json.dumps(feedback_data) + "\n")

    # Update summary statistics
    await update_learning_summary()

    return {
        "status": "received",
        "message": "Thank you! Your feedback helps improve RehabOS.",
        "feedback_id": feedback.feedback_id,
    }


async def update_learning_summary():
    """Update aggregated learning data from feedback."""
    feedback_file = LEARNING_DATA_PATH / "feedback.jsonl"
    summary_file = LEARNING_DATA_PATH / "feedback_summary.json"

    if not feedback_file.exists():
        return

    # Aggregate feedback
    condition_data = {}
    total_ratings = []

    with open(feedback_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                total_ratings.append(entry.get("rating", 3))

                # Track by condition (would need condition in feedback)
                if entry.get("modifications_made"):
                    for mod in entry["modifications_made"]:
                        condition_data.setdefault("general", {}).setdefault("modifications", []).append(mod)

            except json.JSONDecodeError:
                continue

    # Build summary
    summary = {
        "total_feedback_count": len(total_ratings),
        "average_rating": sum(total_ratings) / len(total_ratings) if total_ratings else 0,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "condition_insights": {},
    }

    # Save summary
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


@router.get("/learning-stats")
async def get_learning_stats():
    """Get statistics about the learning system."""
    summary_file = LEARNING_DATA_PATH / "feedback_summary.json"

    if not summary_file.exists():
        return {
            "status": "initializing",
            "message": "Learning system collecting data",
            "feedback_count": 0,
            "insights_available": False,
        }

    with open(summary_file) as f:
        summary = json.load(f)

    return {
        "status": "active",
        "feedback_count": summary.get("total_feedback_count", 0),
        "average_rating": round(summary.get("average_rating", 0), 2),
        "last_updated": summary.get("last_updated"),
        "insights_available": summary.get("total_feedback_count", 0) >= 10,
    }


@router.get("/templates")
async def get_program_templates():
    """Get available program templates/conditions."""
    return {
        "conditions": list(CONDITION_PROTOCOLS.keys()),
        "disciplines": ["PT", "OT", "SLP"],
        "care_settings": ["outpatient", "inpatient", "acute", "home_health", "snf"],
    }


class VoiceProgramRequest(BaseModel):
    """Request for voice-to-program parsing."""
    transcript: str


class ParsedProgramData(BaseModel):
    """Parsed program data from voice transcript."""
    condition: Optional[str] = None
    discipline: str = "PT"
    age: Optional[int] = None
    sex: Optional[str] = None
    care_setting: Optional[str] = None
    diagnoses: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)
    patient_goals: list[str] = Field(default_factory=list)
    functional_status: Optional[FunctionalStatus] = None
    prior_level_of_function: Optional[str] = None
    medications: list[Medication] = Field(default_factory=list)
    confidence: float = 0.0
    parsed_fields: list[str] = Field(default_factory=list)


@router.post("/parse-voice", response_model=ParsedProgramData)
async def parse_voice_to_program(request: VoiceProgramRequest):
    """Parse a voice transcript into program request fields.

    Examples of supported input:
    - "76 year old male with Parkinson's, outpatient PT, goals are to improve walking and reduce falls"
    - "Create a program for total knee replacement, 65 year old female, SNF setting"
    - "Stroke patient, 80 years old, max assist for mobility, history of diabetes and hypertension"
    """
    import re

    transcript = request.transcript.lower()
    parsed_fields = []

    # Parse age
    age = None
    age_patterns = [
        r'(\d{1,3})\s*(?:year|yr|y/o|yo|y\.o\.)\s*old',
        r'age[\s:]*(\d{1,3})',
        r'(\d{1,3})\s*(?:male|female|m|f)\b',
    ]
    for pattern in age_patterns:
        match = re.search(pattern, transcript)
        if match:
            try:
                age = int(match.group(1))
                if 0 < age < 120:
                    parsed_fields.append("age")
                    break
            except ValueError:
                pass

    # Parse sex
    sex = None
    if re.search(r'\b(male|man|gentleman)\b', transcript):
        sex = "male"
        parsed_fields.append("sex")
    elif re.search(r'\b(female|woman|lady)\b', transcript):
        sex = "female"
        parsed_fields.append("sex")

    # Parse discipline
    discipline = "PT"
    if any(term in transcript for term in ["occupational", " ot ", "ot,", "ot."]):
        discipline = "OT"
        parsed_fields.append("discipline")
    elif any(term in transcript for term in ["speech", "slp", "swallow", "dysphagia"]):
        discipline = "SLP"
        parsed_fields.append("discipline")
    elif any(term in transcript for term in ["physical", " pt ", "pt,", "pt."]):
        discipline = "PT"
        parsed_fields.append("discipline")

    # Parse care setting
    care_setting = None
    settings_map = {
        "outpatient": ["outpatient", "op", "clinic", "office"],
        "inpatient": ["inpatient", "ip", "hospital", "acute care"],
        "home_health": ["home health", "home care", "hh", "at home"],
        "snf": ["snf", "skilled nursing", "nursing home", "nursing facility", "rehab facility"],
    }
    for setting, keywords in settings_map.items():
        if any(kw in transcript for kw in keywords):
            care_setting = setting
            parsed_fields.append("care_setting")
            break

    # Parse condition (primary diagnosis)
    condition = None
    condition_keywords = {
        "parkinson": ["parkinson", "pd", "parkinson's disease"],
        "stroke": ["stroke", "cva", "cerebrovascular", "hemiplegia", "hemiparesis"],
        "total knee replacement": ["tka", "tkr", "total knee", "knee replacement", "knee arthroplasty"],
        "total hip replacement": ["tha", "thr", "total hip", "hip replacement", "hip arthroplasty"],
        "low back pain": ["low back", "lbp", "lumbar", "back pain", "sciatica"],
        "rotator cuff": ["rotator cuff", "shoulder", "rct"],
        "falls/balance": ["falls", "fall risk", "balance", "unsteady"],
        "dysphagia": ["dysphagia", "swallowing", "aspiration"],
        "fracture": ["fracture", "broken"],
    }
    for cond, keywords in condition_keywords.items():
        if any(kw in transcript for kw in keywords):
            condition = cond.title()
            parsed_fields.append("condition")
            break

    # If no specific condition found, try to extract from transcript
    if not condition:
        # Look for "for X" or "with X" patterns
        patterns = [
            r'(?:for|with|has|patient with)\s+([a-z\s]+?)(?:,|\.|patient|who|goal)',
            r'(?:program for|treatment for)\s+([a-z\s]+?)(?:,|\.|patient)',
        ]
        for pattern in patterns:
            match = re.search(pattern, transcript)
            if match:
                potential = match.group(1).strip()
                if len(potential) > 3 and len(potential) < 50:
                    condition = potential.title()
                    parsed_fields.append("condition")
                    break

    # Parse comorbidities
    comorbidities = []
    comorbidity_terms = [
        "diabetes", "hypertension", "htn", "copd", "chf", "heart failure",
        "atrial fibrillation", "afib", "osteoporosis", "arthritis", "obesity",
        "ckd", "kidney disease", "dementia", "depression", "anxiety",
        "cancer", "neuropathy", "dvt", "pe", "asthma"
    ]
    for term in comorbidity_terms:
        if term in transcript:
            comorbidities.append(term.title() if term != "htn" else "Hypertension")
    if comorbidities:
        parsed_fields.append("comorbidities")

    # Parse diagnoses (secondary)
    diagnoses = []
    # Already captured in comorbidities for now

    # Parse patient goals
    goals = []
    goal_patterns = [
        r'goal[s]?\s+(?:is|are|to|:)\s*(.+?)(?:\.|,\s*(?:and|history|patient|with)|$)',
        r'wants?\s+to\s+(.+?)(?:\.|,|$)',
        r'improve\s+(.+?)(?:\.|,|and|$)',
        r'reduce\s+(.+?)(?:\.|,|and|$)',
        r'return\s+to\s+(.+?)(?:\.|,|and|$)',
    ]
    for pattern in goal_patterns:
        matches = re.findall(pattern, transcript)
        for match in matches:
            goal = match.strip()
            if len(goal) > 3 and len(goal) < 100:
                goals.append(goal.capitalize())
    if goals:
        parsed_fields.append("patient_goals")

    # Parse functional status
    functional_status = None
    mobility = None
    balance = None

    mobility_patterns = {
        "independent": ["independent", "walks independently", "ambulates independently"],
        "supervision": ["supervision", "standby", "sba"],
        "min_assist": ["minimal assist", "min assist", "minimum assist", "contact guard", "cga"],
        "mod_assist": ["moderate assist", "mod assist"],
        "max_assist": ["maximum assist", "max assist", "total assist", "dependent"],
    }
    for level, keywords in mobility_patterns.items():
        if any(kw in transcript for kw in keywords):
            mobility = level
            break

    balance_patterns = {
        "good": ["good balance", "balance good"],
        "fair": ["fair balance", "balance fair", "some balance"],
        "poor": ["poor balance", "balance poor", "impaired balance"],
    }
    for level, keywords in balance_patterns.items():
        if any(kw in transcript for kw in keywords):
            balance = level
            break

    if mobility or balance:
        functional_status = FunctionalStatus(mobility=mobility, balance=balance)
        parsed_fields.append("functional_status")

    # Parse prior level of function
    prior_level = None
    prior_patterns = [
        r'(?:prior level|plof|before|previously)\s*[:\-]?\s*(.+?)(?:\.|,|now|currently|$)',
        r'was\s+(?:able to|previously)\s+(.+?)(?:\.|,|before|$)',
    ]
    for pattern in prior_patterns:
        match = re.search(pattern, transcript)
        if match:
            prior_level = match.group(1).strip().capitalize()
            if len(prior_level) > 3:
                parsed_fields.append("prior_level_of_function")
                break

    # Parse medications
    medications = []
    common_meds = [
        "lisinopril", "metoprolol", "amlodipine", "losartan", "atorvastatin",
        "metformin", "omeprazole", "levothyroxine", "gabapentin",
        "furosemide", "warfarin", "eliquis", "xarelto", "aspirin",
        "sinemet", "levodopa", "plavix", "insulin", "prednisone",
    ]
    for med in common_meds:
        if med in transcript:
            medications.append(Medication(name=med.capitalize()))
    if medications:
        parsed_fields.append("medications")

    # Calculate confidence
    confidence = len(parsed_fields) / 10 * 100  # Max 10 key fields
    confidence = min(confidence, 100)

    return ParsedProgramData(
        condition=condition,
        discipline=discipline,
        age=age,
        sex=sex,
        care_setting=care_setting,
        diagnoses=diagnoses,
        comorbidities=comorbidities,
        patient_goals=goals,
        functional_status=functional_status,
        prior_level_of_function=prior_level,
        medications=medications,
        confidence=confidence,
        parsed_fields=parsed_fields,
    )
