"""Clinical data models for comprehensive therapy evaluations.

Rich data structures for ROM, MMT, standardized tests, functional deficits,
and related clinical measures — aligned with the DocPilot evaluation schema.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ROMEntry(BaseModel):
    """Range of Motion measurement for a specific joint/region."""
    joint: str  # e.g. "right_hip", "cervical", "bilateral_knees"
    motion: str = "general"  # e.g. "flexion", "extension", "rotation", "general"
    value: Optional[float] = None  # degrees, or None if qualitative only
    qualitative: Optional[str] = None  # "WFL", "WFL with mild rigidity"
    side: str = "bilateral"  # left, right, bilateral


class MMTEntry(BaseModel):
    """Manual Muscle Testing entry for a specific muscle group."""
    muscle_group: str  # e.g. "hip_flexion", "knee_extension", "trunk_flexion"
    grade: str  # "5/5", "4/5", "3+/5", "2/5", etc.
    side: str = "bilateral"  # left, right, bilateral


class StandardizedTest(BaseModel):
    """Standardized outcome measure / clinical test result."""
    name: str  # "TUG", "Berg Balance Scale", "Tinetti", "5xSTS", "Functional Reach"
    score: float
    max_score: Optional[float] = None
    unit: Optional[str] = None  # "seconds", "inches", etc.
    interpretation: Optional[str] = None
    sub_scores: Optional[dict[str, float]] = None  # e.g. Tinetti balance/gait breakdown


class FunctionalDeficit(BaseModel):
    """Functional deficit with prior and current levels."""
    category: str  # "bed_mobility", "transfers", "gait", "balance"
    activity: str  # "rolling", "sit_to_stand", "level_surfaces"
    prior_level: str
    current_level: str
    assistive_device: Optional[str] = None
    distance: Optional[str] = None
    quality_notes: Optional[str] = None


class Vitals(BaseModel):
    """Patient vitals at time of evaluation/session."""
    blood_pressure_sitting: Optional[str] = None
    blood_pressure_standing: Optional[str] = None
    heart_rate: Optional[int] = None
    spo2: Optional[int] = None
    respiratory_rate: Optional[int] = None
    pain_level: Optional[float] = None
    pain_location: Optional[str] = None
    temperature: Optional[float] = None


class BalanceAssessment(BaseModel):
    """Detailed balance assessment (static/dynamic)."""
    static_sitting: Optional[str] = None
    dynamic_sitting: Optional[str] = None
    static_standing: Optional[str] = None
    dynamic_standing: Optional[str] = None
    single_leg_stance_right: Optional[str] = None
    single_leg_stance_left: Optional[str] = None
    tandem_stance: Optional[str] = None


class ToneAssessment(BaseModel):
    """Muscle tone assessment."""
    region: str
    finding: str  # e.g. "Mild cogwheel rigidity bilaterally"


class SensationAssessment(BaseModel):
    """Sensation assessment."""
    modality: str  # "light_touch", "proprioception", "sharp/dull"
    finding: str  # "Intact bilateral LE", "Mildly impaired bilateral LE"


class PostureAssessment(BaseModel):
    """Posture/alignment findings."""
    findings: str


class GoalWithBaseline(BaseModel):
    """Treatment goal linked to a measured baseline."""
    area: str  # "Strength/MMT", "Balance", "Gait", "TUG"
    goal: str
    baseline: str
    timeframe: Optional[str] = None  # "2 weeks", "4 weeks"
    type: str = "short_term"  # short_term, long_term


class BillingCode(BaseModel):
    """CPT billing code entry."""
    code: str  # "97162", "97110"
    description: Optional[str] = None
    minutes: Optional[int] = None
    units: int = 1
    modifier: Optional[str] = None
    comments: Optional[str] = None
