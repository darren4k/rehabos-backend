"""Tests for rehab_os.clinical.care_transitions — care transition management."""
from __future__ import annotations

import pytest

from rehab_os.clinical.care_transitions import (
    CareTransition,
    TransitionManager,
    VALID_TRANSITIONS,
)
from rehab_os.clinical.settings import ClinicalSetting


@pytest.fixture
def mgr() -> TransitionManager:
    return TransitionManager()


# ---------------------------------------------------------------------------
# Initiation
# ---------------------------------------------------------------------------

class TestInitiateTransition:
    def test_initiate_transition(self, mgr):
        t = mgr.initiate_transition(
            patient_id="pt-1",
            from_setting=ClinicalSetting.SNF,
            to_setting=ClinicalSetting.HOMECARE,
            reason="discharge_home",
            active_goals=["Amb 150ft CGA"],
            medications=["Lisinopril 10mg"],
        )
        assert isinstance(t, CareTransition)
        assert t.status == "pending"
        assert t.from_setting == ClinicalSetting.SNF
        assert t.to_setting == ClinicalSetting.HOMECARE
        assert t.patient_id == "pt-1"
        assert "Amb 150ft CGA" in t.active_goals
        assert t.transition_id  # non-empty

    def test_initiate_returns_unique_ids(self, mgr):
        t1 = mgr.initiate_transition("pt-1", ClinicalSetting.SNF, ClinicalSetting.HOMECARE, "r1")
        t2 = mgr.initiate_transition("pt-2", ClinicalSetting.SNF, ClinicalSetting.OUTPATIENT, "r2")
        assert t1.transition_id != t2.transition_id


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

class TestCompleteTransition:
    def test_complete_transition(self, mgr):
        t = mgr.initiate_transition("pt-1", ClinicalSetting.SNF, ClinicalSetting.HOMECARE, "dc")
        completed = mgr.complete_transition(t.transition_id, clinical_summary="Discharged stable")
        assert completed.status == "completed"
        assert completed.completed_at is not None
        assert completed.clinical_summary == "Discharged stable"

    def test_complete_already_completed_raises(self, mgr):
        t = mgr.initiate_transition("pt-1", ClinicalSetting.SNF, ClinicalSetting.HOMECARE, "dc")
        mgr.complete_transition(t.transition_id)
        with pytest.raises(ValueError, match="already completed"):
            mgr.complete_transition(t.transition_id)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestCancelTransition:
    def test_cancel_transition(self, mgr):
        t = mgr.initiate_transition("pt-1", ClinicalSetting.SNF, ClinicalSetting.HOMECARE, "dc")
        cancelled = mgr.cancel_transition(t.transition_id, reason="Patient readmitted")
        assert cancelled.status == "cancelled"
        assert "Patient readmitted" in cancelled.clinical_summary

    def test_cancel_completed_raises(self, mgr):
        t = mgr.initiate_transition("pt-1", ClinicalSetting.SNF, ClinicalSetting.HOMECARE, "dc")
        mgr.complete_transition(t.transition_id)
        with pytest.raises(ValueError, match="Cannot cancel completed"):
            mgr.cancel_transition(t.transition_id)


# ---------------------------------------------------------------------------
# Pending retrieval
# ---------------------------------------------------------------------------

class TestGetPendingTransitions:
    def test_get_pending_transitions(self, mgr):
        mgr.initiate_transition("pt-1", ClinicalSetting.SNF, ClinicalSetting.HOMECARE, "dc")
        mgr.initiate_transition("pt-2", ClinicalSetting.OUTPATIENT, ClinicalSetting.TELEHEALTH, "tele")
        pending = mgr.get_pending_transitions()
        assert len(pending) == 2

    def test_get_pending_filter_by_patient(self, mgr):
        mgr.initiate_transition("pt-1", ClinicalSetting.SNF, ClinicalSetting.HOMECARE, "dc")
        mgr.initiate_transition("pt-2", ClinicalSetting.OUTPATIENT, ClinicalSetting.TELEHEALTH, "tele")
        pending = mgr.get_pending_transitions(patient_id="pt-1")
        assert len(pending) == 1
        assert pending[0].patient_id == "pt-1"

    def test_completed_not_in_pending(self, mgr):
        t = mgr.initiate_transition("pt-1", ClinicalSetting.SNF, ClinicalSetting.HOMECARE, "dc")
        mgr.complete_transition(t.transition_id)
        pending = mgr.get_pending_transitions()
        assert len(pending) == 0


# ---------------------------------------------------------------------------
# Invalid transition paths
# ---------------------------------------------------------------------------

class TestInvalidTransitionPath:
    def test_invalid_path_raises(self, mgr):
        """SCHOOL -> SNF is not a valid transition."""
        with pytest.raises(ValueError, match="Invalid transition"):
            mgr.initiate_transition("pt-1", ClinicalSetting.SCHOOL, ClinicalSetting.SNF, "invalid")

    def test_telehealth_to_snf_invalid(self, mgr):
        with pytest.raises(ValueError, match="Invalid transition"):
            mgr.initiate_transition("pt-1", ClinicalSetting.TELEHEALTH, ClinicalSetting.SNF, "invalid")

    def test_nonexistent_transition_id_raises(self, mgr):
        with pytest.raises(ValueError, match="not found"):
            mgr.complete_transition("fake-id")
