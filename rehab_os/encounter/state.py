"""Encounter State — structured data model for a clinical documentation session.

Tracks everything collected during an encounter, what's missing, and
provides a summary for the Brain's dynamic prompt.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EncounterPhase(str, Enum):
    SETUP = "setup"
    SUBJECTIVE = "subjective"
    OBJECTIVE = "objective"
    ASSESSMENT = "assessment"
    PLAN = "plan"
    REVIEW = "review"
    COMPLETE = "complete"


# ── Subjective ────────────────────────────────────────────────────────────────


class SubjectiveData(BaseModel):
    chief_complaint: Optional[str] = None
    pain_level: Optional[int] = None
    pain_location: Optional[str] = None
    pain_quality: Optional[str] = None
    hep_compliance: Optional[str] = None
    new_complaints: Optional[str] = None
    patient_report: Optional[str] = None


# ── Objective ─────────────────────────────────────────────────────────────────


class VitalsData(BaseModel):
    bp: Optional[str] = None
    pulse: Optional[int] = None
    spo2: Optional[int] = None
    respiratory_rate: Optional[int] = None
    temperature: Optional[float] = None


class ROMEntry(BaseModel):
    joint: str
    motion: str = "flexion"
    value: Optional[int] = None
    side: str = "bilateral"
    qualitative: Optional[str] = None


class MMTEntry(BaseModel):
    muscle_group: str
    grade: str
    side: str = "bilateral"


class StandardizedTestEntry(BaseModel):
    name: str
    score: str
    max_score: Optional[str] = None
    interpretation: Optional[str] = None


class InterventionEntry(BaseModel):
    name: str
    duration_minutes: Optional[int] = None
    parameters: Optional[str] = None
    patient_response: Optional[str] = None


class FunctionalMobilityEntry(BaseModel):
    activity: str
    assist_level: Optional[str] = None
    device: Optional[str] = None
    distance: Optional[str] = None
    quality: Optional[str] = None


class ObjectiveData(BaseModel):
    vitals: Optional[VitalsData] = None
    interventions: list[InterventionEntry] = Field(default_factory=list)
    rom: list[ROMEntry] = Field(default_factory=list)
    mmt: list[MMTEntry] = Field(default_factory=list)
    standardized_tests: list[StandardizedTestEntry] = Field(default_factory=list)
    functional_mobility: list[FunctionalMobilityEntry] = Field(default_factory=list)
    tolerance: Optional[str] = None
    other: Optional[str] = None


# ── Assessment ────────────────────────────────────────────────────────────────


class AssessmentData(BaseModel):
    progress: Optional[str] = None
    clinical_impression: Optional[str] = None
    rehab_potential: Optional[str] = None
    skilled_justification: Optional[str] = None


# ── Plan ──────────────────────────────────────────────────────────────────────


class PlanData(BaseModel):
    next_visit: Optional[str] = None
    frequency: Optional[str] = None
    goals_update: Optional[str] = None
    hep_changes: Optional[str] = None
    referrals: Optional[str] = None
    discharge_timeline: Optional[str] = None


# ── Patient History (from memU / Patient-Core) ────────────────────────────────


class PatientHistory(BaseModel):
    last_encounters: list[dict] = Field(default_factory=list)
    active_goals: list[dict] = Field(default_factory=list)
    diagnosis: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)


# ── Encounter State ──────────────────────────────────────────────────────────


class EncounterState(BaseModel):
    """Complete encounter state — maintained server-side."""

    encounter_id: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    note_type: Optional[str] = None
    date_of_service: Optional[str] = None
    discipline: str = "PT"
    phase: EncounterPhase = EncounterPhase.SETUP

    subjective: SubjectiveData = Field(default_factory=SubjectiveData)
    objective: ObjectiveData = Field(default_factory=ObjectiveData)
    assessment: AssessmentData = Field(default_factory=AssessmentData)
    plan: PlanData = Field(default_factory=PlanData)

    history: PatientHistory = Field(default_factory=PatientHistory)

    transcript: list[dict] = Field(default_factory=list)
    turn_count: int = 0

    # ── Completeness helpers ──────────────────────────────────────────────

    def missing_critical(self) -> list[str]:
        """What MUST be documented for a valid note."""
        missing = []
        if not self.subjective.chief_complaint:
            missing.append("chief complaint")
        if not self.objective.interventions:
            missing.append("interventions performed")
        if not self.objective.tolerance:
            missing.append("patient tolerance/response")
        if not self.plan.next_visit and not self.plan.frequency:
            missing.append("plan for continued care")
        return missing

    def missing_recommended(self) -> list[str]:
        """What SHOULD be documented for defensibility."""
        rec = []
        if self.subjective.pain_level is None:
            rec.append("pain level")
        if not self.objective.vitals:
            rec.append("vital signs")
        if not self.objective.rom:
            rec.append("ROM measurements")
        if not self.objective.standardized_tests:
            rec.append("standardized tests (Berg, TUG)")
        if self.subjective.hep_compliance is None:
            rec.append("HEP compliance")
        return rec

    def completeness_score(self) -> float:
        """0.0–1.0 score of how complete the documentation is."""
        total = 8.0
        filled = total - len(self.missing_critical()) - min(len(self.missing_recommended()), 4) * 0.5
        return max(0.0, min(1.0, filled / total))

    # ── Prompt summary ────────────────────────────────────────────────────

    def summary_for_prompt(self) -> str:
        """Build a concise summary of what's collected for the Brain's prompt."""
        lines: list[str] = []

        # Subjective
        if self.subjective.chief_complaint:
            pain = f", pain {self.subjective.pain_level}/10" if self.subjective.pain_level is not None else ""
            loc = f" ({self.subjective.pain_location})" if self.subjective.pain_location else ""
            lines.append(f"✓ Chief complaint: {self.subjective.chief_complaint}{pain}{loc}")
        else:
            lines.append("☐ Chief complaint: [NOT YET]")

        if self.subjective.hep_compliance:
            lines.append(f"✓ HEP compliance: {self.subjective.hep_compliance}")

        # Vitals
        if self.objective.vitals:
            v = self.objective.vitals
            parts = []
            if v.bp:
                parts.append(f"BP {v.bp}")
            if v.pulse:
                parts.append(f"HR {v.pulse}")
            if v.spo2:
                parts.append(f"SpO2 {v.spo2}%")
            lines.append(f"✓ Vitals: {', '.join(parts)}")
        else:
            lines.append("☐ Vitals: [NOT YET]")

        # Interventions
        if self.objective.interventions:
            names = [i.name for i in self.objective.interventions]
            lines.append(f"✓ Interventions: {', '.join(names)}")
        else:
            lines.append("☐ Interventions: [NOT YET]")

        # ROM
        if self.objective.rom:
            rom_str = ", ".join(
                f"{r.side} {r.joint} {r.motion} {r.value}°"
                for r in self.objective.rom
                if r.value
            )
            lines.append(f"✓ ROM: {rom_str}")

        # MMT
        if self.objective.mmt:
            mmt_str = ", ".join(f"{m.muscle_group} {m.grade}" for m in self.objective.mmt)
            lines.append(f"✓ MMT: {mmt_str}")

        # Tests
        if self.objective.standardized_tests:
            test_str = ", ".join(f"{t.name} {t.score}" for t in self.objective.standardized_tests)
            lines.append(f"✓ Tests: {test_str}")

        # Functional mobility
        if self.objective.functional_mobility:
            lines.append(f"✓ Functional mobility: {len(self.objective.functional_mobility)} items")

        # Tolerance
        if self.objective.tolerance:
            lines.append(f"✓ Tolerance: {self.objective.tolerance}")
        else:
            lines.append("☐ Patient tolerance: [NOT YET]")

        # Assessment
        if self.assessment.progress:
            lines.append(f"✓ Assessment: {self.assessment.progress}")
        else:
            lines.append("☐ Assessment: [NOT YET — can auto-generate]")

        # Plan
        if self.plan.next_visit or self.plan.frequency:
            plan_parts = []
            if self.plan.next_visit:
                plan_parts.append(self.plan.next_visit)
            if self.plan.frequency:
                plan_parts.append(self.plan.frequency)
            lines.append(f"✓ Plan: {', '.join(plan_parts)}")
        else:
            lines.append("☐ Plan: [NOT YET]")

        return "\n".join(lines)
