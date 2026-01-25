"""Clinical Practice Guideline data models and repository."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from rehab_os.models.evidence import EvidenceLevel, RecommendationStrength


class GuidelineSource(BaseModel):
    """Source organization for a clinical practice guideline."""

    name: str
    abbreviation: str
    url: Optional[str] = None


# Standard guideline sources
APTA = GuidelineSource(name="American Physical Therapy Association", abbreviation="APTA")
AOTA = GuidelineSource(
    name="American Occupational Therapy Association", abbreviation="AOTA"
)
ASHA = GuidelineSource(
    name="American Speech-Language-Hearing Association", abbreviation="ASHA"
)
JOSPT = GuidelineSource(
    name="Journal of Orthopaedic & Sports Physical Therapy", abbreviation="JOSPT"
)


class GuidelineRecommendation(BaseModel):
    """A single recommendation from a clinical practice guideline."""

    recommendation_id: str
    text: str
    strength: RecommendationStrength
    evidence_level: EvidenceLevel
    population: Optional[str] = None
    intervention: Optional[str] = None
    comparison: Optional[str] = None
    outcome: Optional[str] = None
    notes: Optional[str] = None


class ClinicalPracticeGuideline(BaseModel):
    """Clinical Practice Guideline document."""

    title: str
    source: GuidelineSource
    publication_date: date
    conditions: list[str] = Field(default_factory=list)
    disciplines: list[str] = Field(default_factory=list)  # PT, OT, SLP
    recommendations: list[GuidelineRecommendation] = Field(default_factory=list)
    summary: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None

    def get_recommendations_by_strength(
        self,
        min_strength: RecommendationStrength,
    ) -> list[GuidelineRecommendation]:
        """Get recommendations at or above a strength level."""
        strength_order = [
            RecommendationStrength.STRONG_FOR,
            RecommendationStrength.MODERATE_FOR,
            RecommendationStrength.WEAK_FOR,
            RecommendationStrength.NEUTRAL,
        ]

        min_index = strength_order.index(min_strength)
        valid_strengths = strength_order[: min_index + 1]

        return [r for r in self.recommendations if r.strength in valid_strengths]


# Sample guidelines for common conditions
SAMPLE_GUIDELINES = [
    ClinicalPracticeGuideline(
        title="Low Back Pain Clinical Practice Guidelines",
        source=JOSPT,
        publication_date=date(2021, 4, 1),
        conditions=["low back pain", "lumbar radiculopathy"],
        disciplines=["PT"],
        recommendations=[
            GuidelineRecommendation(
                recommendation_id="LBP-1",
                text="Clinicians should use validated self-report questionnaires to assess pain intensity, disability, and function",
                strength=RecommendationStrength.STRONG_FOR,
                evidence_level=EvidenceLevel.LEVEL_1A,
                outcome="Assessment",
            ),
            GuidelineRecommendation(
                recommendation_id="LBP-2",
                text="For acute low back pain, clinicians should provide education on favorable prognosis and self-management",
                strength=RecommendationStrength.STRONG_FOR,
                evidence_level=EvidenceLevel.LEVEL_1B,
                intervention="Patient education",
            ),
            GuidelineRecommendation(
                recommendation_id="LBP-3",
                text="For chronic low back pain, clinicians should provide structured exercise programs",
                strength=RecommendationStrength.STRONG_FOR,
                evidence_level=EvidenceLevel.LEVEL_1A,
                intervention="Therapeutic exercise",
            ),
        ],
    ),
    ClinicalPracticeGuideline(
        title="Neck Pain Clinical Practice Guidelines",
        source=JOSPT,
        publication_date=date(2017, 1, 1),
        conditions=["neck pain", "cervical radiculopathy"],
        disciplines=["PT"],
        recommendations=[
            GuidelineRecommendation(
                recommendation_id="NECK-1",
                text="Clinicians should use manual therapy including manipulation and mobilization combined with exercise",
                strength=RecommendationStrength.STRONG_FOR,
                evidence_level=EvidenceLevel.LEVEL_1A,
                intervention="Manual therapy + exercise",
            ),
        ],
    ),
    ClinicalPracticeGuideline(
        title="Stroke Rehabilitation Guidelines",
        source=APTA,
        publication_date=date(2022, 6, 1),
        conditions=["stroke", "CVA", "hemiplegia"],
        disciplines=["PT", "OT", "SLP"],
        recommendations=[
            GuidelineRecommendation(
                recommendation_id="STROKE-1",
                text="High-intensity, repetitive task-specific practice should be incorporated into rehabilitation",
                strength=RecommendationStrength.STRONG_FOR,
                evidence_level=EvidenceLevel.LEVEL_1A,
                intervention="Task-specific training",
            ),
            GuidelineRecommendation(
                recommendation_id="STROKE-2",
                text="Aerobic exercise training should be included in stroke rehabilitation",
                strength=RecommendationStrength.MODERATE_FOR,
                evidence_level=EvidenceLevel.LEVEL_1B,
                intervention="Aerobic exercise",
            ),
        ],
    ),
    ClinicalPracticeGuideline(
        title="Dysphagia Management Guidelines",
        source=ASHA,
        publication_date=date(2021, 3, 1),
        conditions=["dysphagia", "swallowing disorder"],
        disciplines=["SLP"],
        recommendations=[
            GuidelineRecommendation(
                recommendation_id="DYSPH-1",
                text="Instrumental assessment (VFSS or FEES) should be used to characterize swallowing physiology",
                strength=RecommendationStrength.STRONG_FOR,
                evidence_level=EvidenceLevel.LEVEL_2A,
                intervention="Instrumental assessment",
            ),
            GuidelineRecommendation(
                recommendation_id="DYSPH-2",
                text="Diet texture modifications should be based on instrumental findings",
                strength=RecommendationStrength.MODERATE_FOR,
                evidence_level=EvidenceLevel.LEVEL_2B,
                intervention="Diet modification",
            ),
        ],
    ),
]


class GuidelineRepository:
    """Repository for accessing clinical practice guidelines."""

    def __init__(self, guidelines: Optional[list[ClinicalPracticeGuideline]] = None):
        """Initialize with optional list of guidelines."""
        self._guidelines = guidelines or SAMPLE_GUIDELINES

    def get_by_condition(self, condition: str) -> list[ClinicalPracticeGuideline]:
        """Get guidelines matching a condition."""
        condition_lower = condition.lower()
        return [
            g
            for g in self._guidelines
            if any(condition_lower in c.lower() for c in g.conditions)
        ]

    def get_by_discipline(self, discipline: str) -> list[ClinicalPracticeGuideline]:
        """Get guidelines for a discipline."""
        return [g for g in self._guidelines if discipline in g.disciplines]

    def get_recommendations(
        self,
        condition: str,
        discipline: Optional[str] = None,
        min_strength: Optional[RecommendationStrength] = None,
    ) -> list[GuidelineRecommendation]:
        """Get all recommendations for a condition.

        Args:
            condition: Clinical condition
            discipline: Optional discipline filter
            min_strength: Optional minimum strength filter

        Returns:
            List of recommendations
        """
        guidelines = self.get_by_condition(condition)

        if discipline:
            guidelines = [g for g in guidelines if discipline in g.disciplines]

        recommendations = []
        for g in guidelines:
            if min_strength:
                recommendations.extend(g.get_recommendations_by_strength(min_strength))
            else:
                recommendations.extend(g.recommendations)

        return recommendations

    def add_guideline(self, guideline: ClinicalPracticeGuideline) -> None:
        """Add a guideline to the repository."""
        self._guidelines.append(guideline)

    @property
    def all_guidelines(self) -> list[ClinicalPracticeGuideline]:
        """Get all guidelines."""
        return self._guidelines
