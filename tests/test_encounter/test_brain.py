"""Tests for EncounterBrain — data extraction, instant responses, phase management."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from rehab_os.encounter.brain import EncounterBrain
from rehab_os.encounter.state import (
    EncounterPhase,
    EncounterState,
    InterventionEntry,
    PlanData,
    SubjectiveData,
)


@pytest.fixture
def brain():
    llm_router = MagicMock()
    return EncounterBrain(llm_router=llm_router)


@pytest.fixture
def state():
    return EncounterState(encounter_id="test-brain-001")


@pytest.fixture
def state_with_note_type():
    s = EncounterState(encounter_id="test-brain-002", note_type="daily_note")
    s.phase = EncounterPhase.SUBJECTIVE
    return s


# ── Instant Responses ─────────────────────────────────────────────────────────


class TestInstantResponses:
    def test_greeting_hi(self, brain, state):
        result = brain._check_instant_response("hi", state)
        assert result is not None
        _, response, suggestions = result
        assert "ready" in response.lower() or "who" in response.lower()

    def test_greeting_good_morning(self, brain, state):
        result = brain._check_instant_response("good morning!", state)
        assert result is not None

    def test_thanks(self, brain, state):
        result = brain._check_instant_response("thanks", state)
        assert result is not None
        _, response, _ = result
        assert "welcome" in response.lower()

    def test_help(self, brain, state):
        result = brain._check_instant_response("help", state)
        assert result is not None
        _, response, _ = result
        assert "SOAP" in response or "document" in response.lower()

    def test_clinical_utterance_not_instant(self, brain, state):
        result = brain._check_instant_response("patient has pain 5 out of 10", state)
        assert result is None


# ── Data Extraction ───────────────────────────────────────────────────────────


class TestExtractPain:
    def test_pain_x_out_of_10(self, brain, state):
        brain._extract_pain("pain 3 out of 10", state)
        assert state.subjective.pain_level == 3

    def test_pain_slash_10(self, brain, state):
        brain._extract_pain("pain 5/10", state)
        assert state.subjective.pain_level == 5

    def test_pain_level_is(self, brain, state):
        brain._extract_pain("pain level is 7 out of 10", state)
        assert state.subjective.pain_level == 7

    def test_no_pain(self, brain, state):
        brain._extract_pain("no pain", state)
        assert state.subjective.pain_level == 0

    def test_pain_location_right_knee(self, brain, state):
        brain._extract_pain("pain in the right knee", state)
        assert state.subjective.pain_location == "right knee"

    def test_pain_location_left_shoulder(self, brain, state):
        brain._extract_pain("complains of left shoulder pain", state)
        assert state.subjective.pain_location == "left shoulder"

    def test_pain_location_lower_back(self, brain, state):
        brain._extract_pain("c/o lower back pain", state)
        assert state.subjective.pain_location == "lower back"


class TestExtractVitals:
    def test_bp(self, brain, state):
        brain._extract_vitals("bp 125/66", state)
        assert state.objective.vitals is not None
        assert state.objective.vitals.bp == "125/66"

    def test_bp_blood_pressure_long(self, brain, state):
        brain._extract_vitals("blood pressure is 130 over 80", state)
        assert state.objective.vitals.bp == "130/80"

    def test_pulse(self, brain, state):
        brain._extract_vitals("pulse 72", state)
        assert state.objective.vitals.pulse == 72

    def test_heart_rate(self, brain, state):
        brain._extract_vitals("heart rate is 65", state)
        assert state.objective.vitals.pulse == 65

    def test_spo2(self, brain, state):
        brain._extract_vitals("spo2 98%", state)
        assert state.objective.vitals.spo2 == 98

    def test_multiple_vitals(self, brain, state):
        brain._extract_vitals("bp 120/80 pulse 70 spo2 97", state)
        v = state.objective.vitals
        assert v.bp == "120/80"
        assert v.pulse == 70
        assert v.spo2 == 97


class TestExtractROM:
    def test_full_rom_spec(self, brain, state):
        brain._extract_rom("right knee flexion 95 degrees", state)
        assert len(state.objective.rom) == 1
        r = state.objective.rom[0]
        assert r.side == "right"
        assert r.joint == "knee"
        assert r.motion == "flexion"
        assert r.value == 95

    def test_rom_with_degree_symbol(self, brain, state):
        brain._extract_rom("left shoulder abduction 90°", state)
        assert len(state.objective.rom) == 1
        assert state.objective.rom[0].value == 90

    def test_simple_rom_uses_pain_location(self, brain, state):
        state.subjective.pain_location = "right knee"
        brain._extract_rom("flexion 105 degrees", state)
        assert len(state.objective.rom) == 1
        assert state.objective.rom[0].value == 105


class TestExtractInterventions:
    def test_therapeutic_exercise(self, brain, state):
        brain._extract_interventions("performed therapeutic exercise", state)
        assert len(state.objective.interventions) == 1
        assert state.objective.interventions[0].name == "therapeutic exercise"

    def test_gait_training(self, brain, state):
        brain._extract_interventions("gait training in hallway", state)
        assert any(i.name == "gait training" for i in state.objective.interventions)

    def test_multiple_interventions(self, brain, state):
        brain._extract_interventions(
            "performed therapeutic exercise, gait training, and balance training", state
        )
        names = [i.name for i in state.objective.interventions]
        assert "therapeutic exercise" in names
        assert "gait training" in names
        assert "balance training" in names

    def test_manual_therapy(self, brain, state):
        brain._extract_interventions("manual therapy to right knee", state)
        assert any(i.name == "manual therapy" for i in state.objective.interventions)

    def test_modalities(self, brain, state):
        brain._extract_interventions("applied hot pack to lumbar", state)
        assert any(i.name == "modalities" for i in state.objective.interventions)

    def test_no_duplicates(self, brain, state):
        brain._extract_interventions("therapeutic exercise", state)
        brain._extract_interventions("therapeutic exercise again", state)
        assert len([i for i in state.objective.interventions if i.name == "therapeutic exercise"]) == 1


class TestExtractTolerance:
    def test_tolerated_well(self, brain, state):
        brain._extract_tolerance("tolerated well", state)
        assert "well" in state.objective.tolerance.lower()

    def test_patient_tolerated(self, brain, state):
        brain._extract_tolerance("patient tolerated treatment without issues", state)
        assert state.objective.tolerance is not None

    def test_no_adverse(self, brain, state):
        brain._extract_tolerance("no adverse reactions noted", state)
        assert state.objective.tolerance is not None


class TestExtractTests:
    def test_berg(self, brain, state):
        brain._extract_tests("berg balance score 45", state)
        assert len(state.objective.standardized_tests) == 1
        assert state.objective.standardized_tests[0].name == "Berg Balance Scale"
        assert state.objective.standardized_tests[0].score == "45"

    def test_tug(self, brain, state):
        brain._extract_tests("tug 12.5 seconds", state)
        assert state.objective.standardized_tests[0].name == "TUG"
        assert state.objective.standardized_tests[0].score == "12.5"

    def test_6mwt(self, brain, state):
        brain._extract_tests("6 minute walk test 300 feet", state)
        assert state.objective.standardized_tests[0].name == "6MWT"
        assert state.objective.standardized_tests[0].score == "300"


class TestExtractSetup:
    def test_note_type_daily(self, brain, state):
        brain._extract_setup("daily note for today", "Daily note for today", state)
        assert state.note_type == "daily_note"

    def test_note_type_eval(self, brain, state):
        brain._extract_setup("initial evaluation", "Initial evaluation", state)
        assert state.note_type == "initial_evaluation"

    def test_patient_name(self, brain, state):
        # Regex matches against raw text: keyword + capitalized name
        brain._extract_setup("seeing maria santos", "seeing Maria Santos", state)
        assert state.patient_name == "Maria Santos"

    def test_date_from_text(self, brain, state):
        brain._extract_setup("date 2/17/2026", "date 2/17/2026", state)
        assert state.date_of_service == "2026-02-17"


class TestExtractPlan:
    def test_continue_poc(self, brain, state):
        brain._extract_plan("continue current poc", state)
        assert state.plan.next_visit is not None

    def test_frequency(self, brain, state):
        brain._extract_plan("3 times per week", state)
        assert state.plan.frequency is not None
        assert "3" in state.plan.frequency

    def test_discharge_timeline(self, brain, state):
        brain._extract_plan("discharge in 4 weeks", state)
        assert state.plan.discharge_timeline is not None


class TestExtractHEP:
    def test_compliant(self, brain, state):
        brain._extract_hep("patient is compliant with exercises", state)
        assert "compliant" in state.subjective.hep_compliance.lower()

    def test_non_compliant(self, brain, state):
        brain._extract_hep("patient hasn't been doing exercises", state)
        assert "non-compliant" in state.subjective.hep_compliance.lower()


# ── Phase Management ──────────────────────────────────────────────────────────


class TestPhaseManagement:
    def test_advances_from_setup_on_note_type(self, brain, state):
        state.note_type = "daily_note"
        state = brain._update_phase(state)
        assert state.phase == EncounterPhase.SUBJECTIVE

    def test_stays_in_setup_without_note_type(self, brain, state):
        state = brain._update_phase(state)
        assert state.phase == EncounterPhase.SETUP


# ── Meta Commands ─────────────────────────────────────────────────────────────


class TestMetaCommands:
    def test_generate_triggers_review(self, brain, state):
        state.turn_count = 5
        result = brain._check_meta_commands("generate the note", state)
        assert result is not None
        s, response, suggestions = result
        assert s.phase == EncounterPhase.REVIEW

    def test_generate_warns_about_missing(self, brain, state):
        state.turn_count = 5
        result = brain._check_meta_commands("that's it", state)
        assert result is not None
        _, response, _ = result
        assert "missing" in response.lower() or "generate" in response.lower()

    def test_next_patient_resets(self, brain, state):
        state.patient_name = "Old Patient"
        result = brain._check_meta_commands("next patient", state)
        assert result is not None
        new_state, _, _ = result
        assert new_state.patient_name is None

    def test_skip_to_objective(self, brain, state):
        result = brain._check_meta_commands("skip to objective", state)
        assert result is not None
        s, _, _ = result
        assert s.phase == EncounterPhase.OBJECTIVE

    def test_normal_utterance_no_meta(self, brain, state):
        result = brain._check_meta_commands("pain is 5 out of 10", state)
        assert result is None


# ── Suggestions ───────────────────────────────────────────────────────────────


class TestSuggestions:
    def test_suggests_tolerance_after_interventions(self, brain, state):
        state.objective.interventions.append(InterventionEntry(name="exercise"))
        suggestions = brain._get_suggestions(state)
        assert any("tolerat" in s.lower() for s in suggestions)

    def test_suggests_rom_after_complaint(self, brain, state):
        state.subjective.chief_complaint = "knee pain"
        suggestions = brain._get_suggestions(state)
        assert any("rom" in s.lower() for s in suggestions)

    def test_suggests_generate_when_ready(self, brain, state):
        state.objective.interventions.append(InterventionEntry(name="exercise"))
        state.plan.next_visit = "continue"
        suggestions = brain._get_suggestions(state)
        assert any("generate" in s.lower() for s in suggestions)
