"""Plan of care and treatment models."""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class GoalTimeframe(str, Enum):
    """Timeframe for goal achievement."""

    SHORT_TERM = "short_term"  # 2-4 weeks
    LONG_TERM = "long_term"  # 6-12 weeks
    DISCHARGE = "discharge"


class GoalStatus(str, Enum):
    """Current status of a goal."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    MODIFIED = "modified"
    DISCONTINUED = "discontinued"


class SMARTGoal(BaseModel):
    """SMART goal for treatment planning."""

    description: str = Field(..., description="Full goal statement")
    timeframe: GoalTimeframe = GoalTimeframe.SHORT_TERM
    target_date: Optional[date] = None
    status: GoalStatus = GoalStatus.NOT_STARTED

    # SMART components
    specific: str = Field(..., description="What exactly will be accomplished")
    measurable: str = Field(..., description="How progress will be measured")
    achievable: str = Field(..., description="Why this is realistic")
    relevant: str = Field(..., description="Why this matters to the patient")
    time_bound: str = Field(..., description="When this will be achieved")

    # Metrics
    baseline_value: Optional[str] = None
    target_value: Optional[str] = None
    current_value: Optional[str] = None

    def format_goal(self) -> str:
        """Format goal as standard clinical statement."""
        return f"Patient will {self.specific} as measured by {self.measurable} within {self.time_bound}."


class FITTParameters(BaseModel):
    """FITT principle for exercise prescription."""

    frequency: str = Field(..., description="How often (e.g., '3x/week')")
    intensity: str = Field(..., description="How hard (e.g., 'RPE 4-6/10')")
    time: str = Field(..., description="Duration (e.g., '20-30 minutes')")
    type: str = Field(..., description="Mode of exercise")


class Intervention(BaseModel):
    """Therapeutic intervention."""

    name: str
    category: str = Field(
        ...,
        description="Category: therapeutic_exercise, manual_therapy, "
        "neuromuscular_reeducation, modalities, patient_education, etc.",
    )
    description: str
    rationale: str = Field(..., description="Clinical reasoning for this intervention")
    fitt: Optional[FITTParameters] = None

    # Progression
    progression_criteria: Optional[str] = None
    precautions: list[str] = Field(default_factory=list)

    # Evidence
    evidence_support: Optional[str] = None
    evidence_level: Optional[str] = None

    # Billing
    cpt_codes: list[str] = Field(default_factory=list)


class Exercise(BaseModel):
    """Home exercise program item."""

    name: str
    instructions: str
    sets: Optional[int] = None
    reps: Optional[str] = None  # Can be "10-15" or "30 seconds"
    frequency: str = Field(default="daily")
    hold_time: Optional[str] = None
    rest_time: Optional[str] = None

    # Modifications
    easier_version: Optional[str] = None
    harder_version: Optional[str] = None

    # Precautions
    precautions: list[str] = Field(default_factory=list)
    stop_if: list[str] = Field(default_factory=list)

    # Optional media
    image_url: Optional[str] = None
    video_url: Optional[str] = None

    def format_prescription(self) -> str:
        """Format exercise as prescription string."""
        parts = [self.name]
        if self.sets and self.reps:
            parts.append(f"{self.sets} sets x {self.reps}")
        if self.hold_time:
            parts.append(f"hold {self.hold_time}")
        parts.append(self.frequency)
        return " - ".join(parts)


class ContingencyPlan(BaseModel):
    """Contingency plan for potential complications."""

    scenario: str = Field(..., description="What might happen")
    indicators: list[str] = Field(..., description="Signs to watch for")
    action: str = Field(..., description="What to do")
    escalation: Optional[str] = Field(None, description="When to seek additional care")


class PlanOfCare(BaseModel):
    """Complete plan of care."""

    # Summary
    clinical_summary: str
    clinical_impression: str
    prognosis: str = Field(..., description="Expected outcome and timeframe")
    rehab_potential: str = Field(..., description="Good, fair, poor with rationale")

    # Goals
    smart_goals: list[SMARTGoal] = Field(default_factory=list)

    # Treatment plan
    interventions: list[Intervention] = Field(default_factory=list)
    visit_frequency: str = Field(..., description="e.g., '2x/week for 6 weeks'")
    expected_duration: str = Field(..., description="Total episode duration")

    # Home program
    home_program: list[Exercise] = Field(default_factory=list)
    patient_education: list[str] = Field(default_factory=list)

    # Safety
    precautions: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    contingency_plans: list[ContingencyPlan] = Field(default_factory=list)

    # Coordination
    referrals: list[str] = Field(default_factory=list)
    communication_plan: Optional[str] = None

    # Discharge planning
    discharge_criteria: list[str] = Field(default_factory=list)
    discharge_recommendations: Optional[str] = None

    def get_short_term_goals(self) -> list[SMARTGoal]:
        """Get short-term goals only."""
        return [g for g in self.smart_goals if g.timeframe == GoalTimeframe.SHORT_TERM]

    def get_long_term_goals(self) -> list[SMARTGoal]:
        """Get long-term goals only."""
        return [g for g in self.smart_goals if g.timeframe == GoalTimeframe.LONG_TERM]
