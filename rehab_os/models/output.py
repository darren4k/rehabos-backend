"""Output schemas for agent responses."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from rehab_os.models.evidence import Citation, EvidenceSummary
from rehab_os.models.patient import Discipline, PatientContext, CareSetting
from rehab_os.models.plan import PlanOfCare


class UrgencyLevel(str, Enum):
    """Urgency level for clinical findings."""

    EMERGENT = "emergent"  # Immediate action required
    URGENT = "urgent"  # Same-day evaluation needed
    ROUTINE = "routine"  # Standard care pathway
    LOW = "low"  # Monitor only


class RedFlag(BaseModel):
    """Individual red flag finding."""

    finding: str
    description: str
    rationale: str
    recommended_action: str
    urgency: UrgencyLevel


class SafetyAssessment(BaseModel):
    """Safety screening results from RedFlagAgent."""

    is_safe_to_treat: bool
    red_flags: list[RedFlag] = Field(default_factory=list)
    precautions: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    urgency_level: UrgencyLevel = UrgencyLevel.ROUTINE
    referral_recommended: bool = False
    referral_to: Optional[str] = None
    referral_rationale: Optional[str] = None
    summary: str

    @property
    def has_critical_findings(self) -> bool:
        """Check if any critical red flags present."""
        return any(rf.urgency == UrgencyLevel.EMERGENT for rf in self.red_flags)


class DiagnosisResult(BaseModel):
    """Diagnosis classification from DiagnosisAgent."""

    primary_diagnosis: str
    icd_codes: list[str] = Field(default_factory=list)
    classification: Optional[str] = None  # e.g., "acute", "chronic", severity
    differential_diagnoses: list[str] = Field(default_factory=list)
    rationale: str
    key_findings: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)
    uncertainties: list[str] = Field(default_factory=list)


class OutcomeMeasure(BaseModel):
    """Recommended outcome measure."""

    name: str
    abbreviation: Optional[str] = None
    description: str
    rationale: str = Field(..., description="Why this measure is appropriate")
    frequency: str = Field(..., description="How often to assess")
    mcid: Optional[str] = Field(None, description="Minimal clinically important difference")
    mdc: Optional[str] = Field(None, description="Minimal detectable change")
    normative_data: Optional[str] = None
    administration_time: Optional[str] = None


class OutcomeRecommendations(BaseModel):
    """Outcome measures recommendations from OutcomeAgent."""

    primary_measures: list[OutcomeMeasure] = Field(default_factory=list)
    secondary_measures: list[OutcomeMeasure] = Field(default_factory=list)
    reassessment_schedule: str
    rationale: str


class DocumentationType(str, Enum):
    """Type of clinical documentation."""

    INITIAL_EVAL = "initial_evaluation"
    DAILY_NOTE = "daily_note"
    PROGRESS_NOTE = "progress_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    REEVALUATION = "reevaluation"


class ClinicalDocumentation(BaseModel):
    """Generated clinical documentation."""

    document_type: DocumentationType
    content: str
    sections: dict[str, str] = Field(default_factory=dict)
    word_count: int
    citations: list[Citation] = Field(default_factory=list)


class QAResult(BaseModel):
    """Quality assurance review from QALearningAgent."""

    overall_quality: float = Field(..., ge=0, le=1)
    strengths: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    alternative_approaches: list[str] = Field(default_factory=list)


class ClinicalRequest(BaseModel):
    """Input request for clinical consultation."""

    query: str = Field(..., description="Clinical question or consultation request")
    patient: Optional[PatientContext] = None
    discipline: Discipline = Discipline.PT
    setting: CareSetting = CareSetting.OUTPATIENT
    task_type: str = Field(
        default="full_consult",
        description="Type: full_consult, diagnosis_only, plan_only, evidence_search",
    )
    include_documentation: bool = False
    documentation_type: Optional[DocumentationType] = None


class ConsultationResponse(BaseModel):
    """Complete response from orchestrator."""

    # Core outputs
    safety: SafetyAssessment
    diagnosis: Optional[DiagnosisResult] = None
    evidence: Optional[EvidenceSummary] = None
    plan: Optional[PlanOfCare] = None
    outcomes: Optional[OutcomeRecommendations] = None
    documentation: Optional[ClinicalDocumentation] = None

    # Quality
    qa_review: Optional[QAResult] = None

    # Metadata
    citations: list[Citation] = Field(default_factory=list)
    processing_notes: list[str] = Field(default_factory=list)

    # Disclaimer
    disclaimer: str = Field(
        default="This is clinical decision support information only. "
        "It is not a substitute for professional clinical judgment. "
        "All recommendations should be verified by a licensed clinician."
    )
