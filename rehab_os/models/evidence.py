"""Evidence and citation models."""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class EvidenceLevel(str, Enum):
    """Level of evidence classification."""

    LEVEL_1A = "1a"  # Systematic review of RCTs
    LEVEL_1B = "1b"  # Individual RCT
    LEVEL_2A = "2a"  # Systematic review of cohort studies
    LEVEL_2B = "2b"  # Individual cohort study
    LEVEL_3A = "3a"  # Systematic review of case-control studies
    LEVEL_3B = "3b"  # Individual case-control study
    LEVEL_4 = "4"  # Case series
    LEVEL_5 = "5"  # Expert opinion
    CPG = "cpg"  # Clinical Practice Guideline
    UNKNOWN = "unknown"


class RecommendationStrength(str, Enum):
    """Strength of recommendation."""

    STRONG_FOR = "strong_for"
    MODERATE_FOR = "moderate_for"
    WEAK_FOR = "weak_for"
    NEUTRAL = "neutral"
    WEAK_AGAINST = "weak_against"
    MODERATE_AGAINST = "moderate_against"
    STRONG_AGAINST = "strong_against"


class Citation(BaseModel):
    """Bibliographic citation."""

    title: str
    authors: list[str] = Field(default_factory=list)
    journal: Optional[str] = None
    year: Optional[int] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[HttpUrl] = None

    def format_apa(self) -> str:
        """Format citation in APA style."""
        if not self.authors:
            author_str = "Unknown"
        elif len(self.authors) == 1:
            author_str = self.authors[0]
        elif len(self.authors) <= 3:
            author_str = ", ".join(self.authors[:-1]) + f", & {self.authors[-1]}"
        else:
            author_str = f"{self.authors[0]} et al."

        year_str = f"({self.year})" if self.year else "(n.d.)"
        journal_str = f" {self.journal}" if self.journal else ""
        vol_str = f", {self.volume}" if self.volume else ""
        pages_str = f", {self.pages}" if self.pages else ""

        return f"{author_str} {year_str}. {self.title}.{journal_str}{vol_str}{pages_str}."


class Evidence(BaseModel):
    """Evidence item from knowledge base or literature."""

    content: str = Field(..., description="Summary or excerpt of evidence")
    source: str = Field(..., description="Source name (guideline, article, etc.)")
    citation: Optional[Citation] = None
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN
    recommendation_strength: Optional[RecommendationStrength] = None

    # Metadata
    condition: Optional[str] = None
    intervention: Optional[str] = None
    population: Optional[str] = None
    outcome: Optional[str] = None

    # Vector store metadata
    relevance_score: Optional[float] = Field(None, ge=0, le=1)
    chunk_id: Optional[str] = None

    # Timestamps
    publication_date: Optional[date] = None
    retrieved_date: Optional[date] = None


class GuidelineRecommendation(BaseModel):
    """Specific recommendation from a clinical practice guideline."""

    guideline_name: str
    organization: str  # APTA, AOTA, ASHA, etc.
    recommendation_text: str
    strength: RecommendationStrength
    evidence_level: EvidenceLevel
    conditions: list[str] = Field(default_factory=list)
    population: Optional[str] = None
    notes: Optional[str] = None
    citation: Optional[Citation] = None


class EvidenceSummary(BaseModel):
    """Aggregated evidence for a clinical question."""

    query: str
    total_sources: int
    evidence_items: list[Evidence] = Field(default_factory=list)
    guideline_recommendations: list[GuidelineRecommendation] = Field(default_factory=list)
    synthesis: Optional[str] = None
    limitations: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
