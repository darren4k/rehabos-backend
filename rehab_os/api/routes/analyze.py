"""Patient analysis API for clinical decision support.

Analyzes patient data (from EMR or direct input) and provides
evidence-based rehabilitation recommendations.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/analyze", tags=["analyze"])


class Medication(BaseModel):
    """Patient medication."""
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None


class LabValue(BaseModel):
    """Lab result."""
    name: str
    value: str
    unit: Optional[str] = None
    flag: Optional[str] = None  # "high", "low", "critical"


class VitalSigns(BaseModel):
    """Current vital signs."""
    blood_pressure: Optional[str] = None  # "120/80"
    heart_rate: Optional[int] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[float] = None
    temperature: Optional[float] = None


class PatientData(BaseModel):
    """Structured patient data - can come from EMR or manual entry."""

    # Demographics
    age: int
    sex: str = "unknown"

    # Clinical
    diagnoses: list[str] = Field(default_factory=list, description="Active diagnoses/ICD codes")
    chief_complaint: Optional[str] = None
    history_of_present_illness: Optional[str] = None

    # Medical context
    comorbidities: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    surgical_history: list[str] = Field(default_factory=list)

    # Current status
    vitals: Optional[VitalSigns] = None
    labs: list[LabValue] = Field(default_factory=list)

    # Functional info
    prior_level_of_function: Optional[str] = None
    current_functional_status: Optional[str] = None
    living_situation: Optional[str] = None
    assistive_devices: list[str] = Field(default_factory=list)

    # Rehab-specific
    discipline: str = "PT"
    care_setting: str = "outpatient"
    referral_reason: Optional[str] = None


class RehabRecommendation(BaseModel):
    """A specific treatment recommendation."""
    intervention: str
    rationale: str
    evidence_level: str
    frequency: Optional[str] = None
    precautions: list[str] = Field(default_factory=list)
    modifications: list[str] = Field(default_factory=list)


class MedicationConsideration(BaseModel):
    """Medication consideration for therapy."""
    medication: str
    consideration: str
    action: str  # "monitor", "time_around", "caution", "contraindication"


class AnalysisResponse(BaseModel):
    """Complete patient analysis with evidence-based recommendations."""

    # Summary
    clinical_summary: str
    rehab_appropriateness: str  # "appropriate", "appropriate_with_precautions", "defer", "contraindicated"

    # Key findings
    relevant_diagnoses: list[str]
    key_considerations: list[str]
    precautions: list[str]
    contraindications: list[str]

    # Recommendations
    recommended_interventions: list[RehabRecommendation]
    suggested_outcome_measures: list[str]
    suggested_frequency: str
    estimated_duration: str

    # Medication considerations
    medication_considerations: list[MedicationConsideration]

    # Lab considerations
    lab_considerations: list[str]

    # Citations
    relevant_guidelines: list[str]

    # Follow-up
    red_flags_to_monitor: list[str]
    reassessment_triggers: list[str]


# Clinical rules for analysis
MEDICATION_RULES = {
    "anticoagulant": {
        "keywords": ["warfarin", "coumadin", "eliquis", "apixaban", "xarelto", "rivaroxaban", "heparin", "lovenox", "pradaxa"],
        "consideration": "Increased bleeding risk with falls or manual therapy",
        "action": "caution",
    },
    "antihypertensive": {
        "keywords": ["lisinopril", "metoprolol", "amlodipine", "losartan", "hydrochlorothiazide", "hctz"],
        "consideration": "Risk of orthostatic hypotension - monitor BP with position changes",
        "action": "monitor",
    },
    "diabetic": {
        "keywords": ["metformin", "insulin", "glipizide", "jardiance", "ozempic", "trulicity"],
        "consideration": "Monitor for hypoglycemia during exercise; check glucose if symptomatic",
        "action": "monitor",
    },
    "opioid": {
        "keywords": ["oxycodone", "hydrocodone", "morphine", "fentanyl", "tramadol", "norco", "percocet"],
        "consideration": "May affect balance, cognition, and fall risk; sedation effects",
        "action": "caution",
    },
    "parkinsons": {
        "keywords": ["carbidopa", "levodopa", "sinemet", "ropinirole", "pramipexole", "mirapex", "requip"],
        "consideration": "Time therapy during 'on' periods; watch for wearing-off and dyskinesias",
        "action": "time_around",
    },
    "muscle_relaxant": {
        "keywords": ["flexeril", "cyclobenzaprine", "baclofen", "tizanidine", "zanaflex", "robaxin"],
        "consideration": "May cause sedation and affect balance; increased fall risk",
        "action": "caution",
    },
    "benzodiazepine": {
        "keywords": ["lorazepam", "ativan", "diazepam", "valium", "alprazolam", "xanax", "clonazepam", "klonopin"],
        "consideration": "Significant fall risk factor; affects balance and cognition",
        "action": "caution",
    },
    "steroid": {
        "keywords": ["prednisone", "dexamethasone", "methylprednisolone", "solu-medrol"],
        "consideration": "Long-term use: bone health, skin fragility, muscle weakness",
        "action": "monitor",
    },
}

DIAGNOSIS_PROTOCOLS = {
    "parkinson": {
        "keywords": ["parkinson", "g20"],
        "interventions": [
            RehabRecommendation(
                intervention="LSVT BIG or amplitude-based training",
                rationale="Level I evidence for improving bradykinesia and functional mobility",
                evidence_level="Level I",
                frequency="4x/week for 4 weeks",
                precautions=["Time with medication 'on' periods"],
            ),
            RehabRecommendation(
                intervention="Balance and falls prevention training",
                rationale="High fall risk in PD; perturbation training reduces falls",
                evidence_level="Level I",
                frequency="2-3x/week",
                precautions=["Screen for orthostatic hypotension", "Assess freezing triggers"],
            ),
            RehabRecommendation(
                intervention="Gait training with cueing strategies",
                rationale="External cues (auditory, visual) improve gait initiation and reduce freezing",
                evidence_level="Level I",
                precautions=["Identify freezing triggers"],
            ),
            RehabRecommendation(
                intervention="High-intensity aerobic exercise",
                rationale="May have neuroprotective effects; improves cardiovascular fitness",
                evidence_level="Level I",
                frequency="3x/week, 30-40 min at 70-85% max HR",
            ),
        ],
        "outcome_measures": ["TUG", "Mini-BESTest", "6MWT", "5x STS", "PDQ-39"],
        "guidelines": ["AAN Parkinson Disease PT Guideline (2022)"],
    },
    "stroke": {
        "keywords": ["stroke", "cva", "i63", "i64", "hemiplegia", "hemiparesis"],
        "interventions": [
            RehabRecommendation(
                intervention="Task-specific training",
                rationale="Neuroplasticity requires high-repetition, meaningful practice",
                evidence_level="Level I",
                frequency="Daily during acute rehab; 3x/week outpatient",
            ),
            RehabRecommendation(
                intervention="Gait training (overground and treadmill)",
                rationale="Task-specific approach improves walking speed and endurance",
                evidence_level="Level I",
            ),
            RehabRecommendation(
                intervention="Balance training",
                rationale="Reduces fall risk; improves functional mobility",
                evidence_level="Level I",
            ),
        ],
        "outcome_measures": ["10MWT", "6MWT", "Berg Balance Scale", "FIM", "ARAT"],
        "guidelines": ["AHA/ASA Stroke Rehabilitation Guidelines (2016)"],
    },
    "low back pain": {
        "keywords": ["low back", "lumbar", "m54", "lbp"],
        "interventions": [
            RehabRecommendation(
                intervention="Therapeutic exercise (core stabilization, motor control)",
                rationale="First-line treatment with Level I evidence",
                evidence_level="Level I",
                frequency="2-3x/week",
            ),
            RehabRecommendation(
                intervention="Patient education and self-management",
                rationale="Reduces fear-avoidance; improves long-term outcomes",
                evidence_level="Level I",
            ),
            RehabRecommendation(
                intervention="Manual therapy (as adjunct)",
                rationale="Short-term pain relief when combined with exercise",
                evidence_level="Level II",
                precautions=["Not as standalone treatment"],
            ),
        ],
        "outcome_measures": ["ODI", "NPRS", "FABQ"],
        "guidelines": ["APTA Low Back Pain CPG (2021)"],
    },
    "fall": {
        "keywords": ["fall", "falls", "r29.6", "w19"],
        "interventions": [
            RehabRecommendation(
                intervention="Multimodal exercise program",
                rationale="Balance + strength training reduces falls 23-40%",
                evidence_level="Level I",
                frequency="3+ hours/week",
            ),
            RehabRecommendation(
                intervention="Home safety evaluation",
                rationale="Environmental modifications reduce fall risk",
                evidence_level="Level I",
            ),
            RehabRecommendation(
                intervention="Floor transfer training",
                rationale="Ability to get up after fall reduces injury severity",
                evidence_level="Level II",
            ),
        ],
        "outcome_measures": ["TUG", "Berg", "30s STS", "ABC Scale"],
        "guidelines": ["CDC STEADI Falls Prevention (2023)"],
    },
    "total knee": {
        "keywords": ["tka", "total knee", "knee replacement", "z96.65"],
        "interventions": [
            RehabRecommendation(
                intervention="ROM exercises (flexion/extension)",
                rationale="Early ROM prevents arthrofibrosis",
                evidence_level="Level I",
                frequency="Multiple times daily",
                precautions=["Follow surgeon protocol for ROM limits"],
            ),
            RehabRecommendation(
                intervention="Quadriceps strengthening",
                rationale="Quad strength correlates with functional outcomes",
                evidence_level="Level I",
            ),
            RehabRecommendation(
                intervention="Gait training with progression to least restrictive device",
                rationale="Early ambulation improves outcomes",
                evidence_level="Level I",
                precautions=["Weight bearing per surgeon protocol"],
            ),
        ],
        "outcome_measures": ["TUG", "KOOS", "Knee ROM", "Quad strength"],
        "guidelines": ["APTA Total Knee Arthroplasty CPG"],
    },
    "hip fracture": {
        "keywords": ["hip fracture", "s72", "orif hip", "hemiarthroplasty"],
        "interventions": [
            RehabRecommendation(
                intervention="Early mobilization",
                rationale="Reduces complications; improves functional outcomes",
                evidence_level="Level I",
                frequency="Begin POD 1 if medically stable",
                precautions=["Weight bearing per surgeon", "Hip precautions if applicable"],
            ),
            RehabRecommendation(
                intervention="Progressive strengthening",
                rationale="Addresses muscle weakness from injury and immobility",
                evidence_level="Level I",
            ),
            RehabRecommendation(
                intervention="Balance and falls prevention",
                rationale="High risk for subsequent falls",
                evidence_level="Level I",
            ),
        ],
        "outcome_measures": ["TUG", "FIM", "30s STS"],
        "guidelines": ["AAOS Hip Fracture Guidelines"],
    },
}


def analyze_medications(medications: list[Medication]) -> list[MedicationConsideration]:
    """Analyze medications for therapy considerations."""
    considerations = []

    for med in medications:
        med_lower = med.name.lower()
        for category, rules in MEDICATION_RULES.items():
            for keyword in rules["keywords"]:
                if keyword in med_lower:
                    considerations.append(MedicationConsideration(
                        medication=med.name,
                        consideration=rules["consideration"],
                        action=rules["action"],
                    ))
                    break

    return considerations


def analyze_labs(labs: list[LabValue]) -> list[str]:
    """Analyze labs for therapy considerations."""
    considerations = []

    for lab in labs:
        lab_name = lab.name.lower()

        if "inr" in lab_name and lab.flag in ["high", "critical"]:
            considerations.append(f"INR {lab.value} - elevated bleeding risk; avoid aggressive manual therapy")

        if "hemoglobin" in lab_name or "hgb" in lab_name:
            try:
                val = float(lab.value.replace(" ", "").split()[0])
                if val < 8:
                    considerations.append(f"Hemoglobin {lab.value} - significant anemia; monitor for fatigue, limit intensity")
                elif val < 10:
                    considerations.append(f"Hemoglobin {lab.value} - mild anemia; may affect exercise tolerance")
            except:
                pass

        if "platelet" in lab_name or "plt" in lab_name:
            try:
                val = float(lab.value.replace(",", "").replace(" ", "").split()[0])
                if val < 50000:
                    considerations.append(f"Platelets {lab.value} - bleeding precautions; avoid falls, no aggressive soft tissue work")
                elif val < 100000:
                    considerations.append(f"Platelets {lab.value} - mild thrombocytopenia; use caution with manual therapy")
            except:
                pass

        if "glucose" in lab_name or "blood sugar" in lab_name:
            if lab.flag == "low":
                considerations.append(f"Low glucose noted - have fast-acting carbs available during exercise")
            elif lab.flag == "high":
                considerations.append(f"Elevated glucose - monitor for symptoms; may affect wound healing")

        if "creatinine" in lab_name:
            try:
                val = float(lab.value.replace(" ", "").split()[0])
                if val > 2.0:
                    considerations.append(f"Creatinine {lab.value} - renal impairment; monitor fluid status and medication effects")
            except:
                pass

    return considerations


def get_matching_protocols(diagnoses: list[str], chief_complaint: Optional[str]) -> list[dict]:
    """Find matching treatment protocols based on diagnoses."""
    matches = []
    search_text = " ".join(diagnoses).lower()
    if chief_complaint:
        search_text += " " + chief_complaint.lower()

    for protocol_name, protocol in DIAGNOSIS_PROTOCOLS.items():
        for keyword in protocol["keywords"]:
            if keyword in search_text:
                matches.append({"name": protocol_name, "protocol": protocol})
                break

    return matches


@router.post("/patient", response_model=AnalysisResponse)
async def analyze_patient(patient: PatientData):
    """Analyze patient data and provide evidence-based rehab recommendations.

    Accepts structured patient data (can come from EMR integration or manual entry)
    and returns comprehensive treatment recommendations with citations.
    """

    # Analyze medications
    med_considerations = analyze_medications(patient.medications)

    # Analyze labs
    lab_considerations = analyze_labs(patient.labs)

    # Find matching protocols
    protocols = get_matching_protocols(patient.diagnoses, patient.chief_complaint)

    # Build recommendations
    all_interventions = []
    all_outcome_measures = []
    all_guidelines = []

    for match in protocols:
        protocol = match["protocol"]
        all_interventions.extend(protocol["interventions"])
        all_outcome_measures.extend(protocol.get("outcome_measures", []))
        all_guidelines.extend(protocol.get("guidelines", []))

    # Deduplicate
    all_outcome_measures = list(set(all_outcome_measures))
    all_guidelines = list(set(all_guidelines))

    # Build precautions based on age, comorbidities, medications
    precautions = []
    contraindications = []
    red_flags = []

    # Age-based considerations
    if patient.age >= 75:
        precautions.append("Advanced age - monitor for fatigue, fall risk, orthostatic changes")
    if patient.age >= 85:
        precautions.append("Very elderly - conservative progression; prioritize safety and function")

    # Comorbidity-based
    for comorbidity in patient.comorbidities:
        c_lower = comorbidity.lower()
        if "cardiac" in c_lower or "heart" in c_lower or "chf" in c_lower:
            precautions.append("Cardiac history - monitor vitals; RPE-based exercise intensity")
            red_flags.append("New/worsening chest pain, SOB, or edema")
        if "copd" in c_lower or "pulmonary" in c_lower:
            precautions.append("Pulmonary - monitor O2 sat; allow rest breaks")
            red_flags.append("Oxygen desaturation <88% or significant SOB")
        if "diabetes" in c_lower:
            precautions.append("Diabetes - monitor for hypoglycemia; foot inspection")
            red_flags.append("Hypoglycemia symptoms during exercise")
        if "osteoporosis" in c_lower:
            precautions.append("Osteoporosis - avoid high-impact; fall prevention priority")
        if "dementia" in c_lower or "cognitive" in c_lower:
            precautions.append("Cognitive impairment - simplified instructions; caregiver involvement")

    # Medication-based precautions
    high_risk_meds = [m for m in med_considerations if m.action in ["caution", "contraindication"]]
    if high_risk_meds:
        precautions.append(f"High-risk medications: {', '.join(m.medication for m in high_risk_meds)}")

    # Build clinical summary
    diagnosis_summary = ", ".join(patient.diagnoses[:3]) if patient.diagnoses else "See referral"

    clinical_summary = f"{patient.age}yo {patient.sex}"
    if diagnosis_summary:
        clinical_summary += f" with {diagnosis_summary}"
    if patient.chief_complaint:
        clinical_summary += f". Presenting with {patient.chief_complaint}"
    if patient.current_functional_status:
        clinical_summary += f". Current function: {patient.current_functional_status}"

    # Determine appropriateness
    if contraindications:
        appropriateness = "defer"
    elif len(precautions) > 3 or high_risk_meds:
        appropriateness = "appropriate_with_precautions"
    else:
        appropriateness = "appropriate"

    # Default frequency/duration if no specific protocol
    if not protocols:
        suggested_frequency = "2-3x/week"
        estimated_duration = "4-8 weeks"
    else:
        suggested_frequency = "Per protocol - typically 2-3x/week"
        estimated_duration = "4-12 weeks depending on condition and goals"

    return AnalysisResponse(
        clinical_summary=clinical_summary,
        rehab_appropriateness=appropriateness,
        relevant_diagnoses=patient.diagnoses,
        key_considerations=[
            f"Age: {patient.age}",
            f"Setting: {patient.care_setting}",
            f"Medications: {len(patient.medications)} active",
            f"Comorbidities: {len(patient.comorbidities)}",
        ],
        precautions=precautions,
        contraindications=contraindications,
        recommended_interventions=all_interventions,
        suggested_outcome_measures=all_outcome_measures,
        suggested_frequency=suggested_frequency,
        estimated_duration=estimated_duration,
        medication_considerations=med_considerations,
        lab_considerations=lab_considerations,
        relevant_guidelines=all_guidelines,
        red_flags_to_monitor=red_flags,
        reassessment_triggers=[
            "Significant change in symptoms or function",
            "New falls or near-falls",
            "Failure to progress as expected",
            "New medical issues or hospitalizations",
        ],
    )


class QuickAnalysisRequest(BaseModel):
    diagnoses: list[str]
    age: int = 65
    discipline: str = "PT"


@router.post("/quick-analysis")
async def quick_analysis(body: QuickAnalysisRequest):
    """Quick analysis based on diagnoses only.

    Use this for rapid lookup without full patient context.
    """
    diagnoses = body.diagnoses
    protocols = get_matching_protocols(diagnoses, None)

    if not protocols:
        return {
            "diagnoses": diagnoses,
            "message": "No specific protocols found. Consider general therapeutic exercise and functional training.",
            "suggested_outcome_measures": ["TUG", "6MWT", "Patient-reported outcome"],
        }

    all_interventions = []
    all_guidelines = []

    for match in protocols:
        all_interventions.extend([
            {"intervention": i.intervention, "rationale": i.rationale, "evidence_level": i.evidence_level}
            for i in match["protocol"]["interventions"]
        ])
        all_guidelines.extend(match["protocol"].get("guidelines", []))

    return {
        "diagnoses": diagnoses,
        "matched_protocols": [m["name"] for m in protocols],
        "recommended_interventions": all_interventions,
        "relevant_guidelines": list(set(all_guidelines)),
    }
