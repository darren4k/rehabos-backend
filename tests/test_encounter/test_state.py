"""Tests for EncounterState — completeness tracking, phase management, prompt summary."""

import pytest

from rehab_os.encounter.state import (
    AssessmentData,
    EncounterPhase,
    EncounterState,
    InterventionEntry,
    MMTEntry,
    ObjectiveData,
    PlanData,
    ROMEntry,
    StandardizedTestEntry,
    SubjectiveData,
    VitalsData,
)


@pytest.fixture
def empty_state():
    return EncounterState(encounter_id="test-001")


@pytest.fixture
def complete_state():
    state = EncounterState(
        encounter_id="test-002",
        patient_name="Maria Santos",
        note_type="daily_note",
        date_of_service="2026-02-17",
        phase=EncounterPhase.PLAN,
    )
    state.subjective = SubjectiveData(
        chief_complaint="right knee pain",
        pain_level=3,
        pain_location="right knee",
        hep_compliance="Compliant with HEP",
    )
    state.objective = ObjectiveData(
        vitals=VitalsData(bp="125/66", pulse=62, spo2=98),
        interventions=[
            InterventionEntry(name="therapeutic exercise", duration_minutes=15),
            InterventionEntry(name="gait training", duration_minutes=10),
            InterventionEntry(name="manual therapy", duration_minutes=10),
        ],
        rom=[ROMEntry(joint="knee", motion="flexion", value=105, side="right")],
        mmt=[MMTEntry(muscle_group="quadriceps", grade="4/5", side="right")],
        standardized_tests=[StandardizedTestEntry(name="TUG", score="12.3")],
        tolerance="Patient tolerated treatment well without adverse reaction",
    )
    state.assessment = AssessmentData(progress="Improving, ROM up from 95 to 105")
    state.plan = PlanData(next_visit="Continue current plan of care", frequency="3x/week")
    return state


class TestMissingCritical:
    def test_empty_state_has_all_critical_missing(self, empty_state):
        missing = empty_state.missing_critical()
        assert "chief complaint" in missing
        assert "interventions performed" in missing
        assert "patient tolerance/response" in missing
        assert "plan for continued care" in missing
        assert len(missing) == 4

    def test_complete_state_has_no_critical_missing(self, complete_state):
        assert complete_state.missing_critical() == []

    def test_partial_state_tracks_remaining(self, empty_state):
        empty_state.subjective.chief_complaint = "knee pain"
        empty_state.objective.interventions.append(InterventionEntry(name="exercise"))
        missing = empty_state.missing_critical()
        assert "chief complaint" not in missing
        assert "interventions performed" not in missing
        assert "patient tolerance/response" in missing
        assert "plan for continued care" in missing

    def test_plan_satisfied_by_frequency_alone(self, empty_state):
        empty_state.subjective.chief_complaint = "pain"
        empty_state.objective.interventions.append(InterventionEntry(name="ex"))
        empty_state.objective.tolerance = "tolerated well"
        empty_state.plan.frequency = "3x/week"
        assert empty_state.missing_critical() == []


class TestMissingRecommended:
    def test_empty_state_has_all_recommended_missing(self, empty_state):
        rec = empty_state.missing_recommended()
        assert "pain level" in rec
        assert "vital signs" in rec
        assert "ROM measurements" in rec
        assert "standardized tests (Berg, TUG)" in rec
        assert "HEP compliance" in rec

    def test_complete_state_has_no_recommended_missing(self, complete_state):
        assert complete_state.missing_recommended() == []


class TestCompletenessScore:
    def test_empty_state_is_low(self, empty_state):
        score = empty_state.completeness_score()
        assert score <= 0.25  # Some recommended items missing counts partial

    def test_complete_state_is_one(self, complete_state):
        assert complete_state.completeness_score() == 1.0

    def test_partial_state_is_between(self, empty_state):
        empty_state.subjective.chief_complaint = "pain"
        empty_state.objective.interventions.append(InterventionEntry(name="ex"))
        score = empty_state.completeness_score()
        assert 0.0 < score < 1.0


class TestSummaryForPrompt:
    def test_empty_state_shows_not_yet(self, empty_state):
        summary = empty_state.summary_for_prompt()
        assert "[NOT YET]" in summary

    def test_complete_state_shows_checkmarks(self, complete_state):
        summary = complete_state.summary_for_prompt()
        assert "right knee pain" in summary
        assert "pain 3/10" in summary
        assert "BP 125/66" in summary
        assert "therapeutic exercise" in summary
        assert "knee flexion 105" in summary
        assert "TUG 12.3" in summary

    def test_rom_entries_formatted(self, complete_state):
        summary = complete_state.summary_for_prompt()
        assert "ROM:" in summary


class TestPhase:
    def test_default_phase_is_setup(self, empty_state):
        assert empty_state.phase == EncounterPhase.SETUP

    def test_all_phases_exist(self):
        phases = [e.value for e in EncounterPhase]
        assert "setup" in phases
        assert "subjective" in phases
        assert "objective" in phases
        assert "assessment" in phases
        assert "plan" in phases
        assert "review" in phases
        assert "complete" in phases
