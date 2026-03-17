"""Tests for rehab_os.clinical.outcomes — longitudinal outcome tracking."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from rehab_os.clinical.outcomes import (
    OUTCOME_MEASURES,
    OutcomeMeasure,
    OutcomeTracker,
    ScoreRecord,
)


@pytest.fixture
def tracker() -> OutcomeTracker:
    return OutcomeTracker()


# ---------------------------------------------------------------------------
# Measure definitions
# ---------------------------------------------------------------------------

class TestMeasureDefinitions:
    def test_outcome_measures_defined(self):
        assert len(OUTCOME_MEASURES) == 22

    def test_all_measures_have_fields(self):
        for key, m in OUTCOME_MEASURES.items():
            assert m.name
            assert m.direction in ("up", "down")
            assert m.mcid > 0
            assert m.mdc > 0
            assert m.discipline in ("PT", "OT", "SLP", "ALL")

    def test_direction_up(self):
        lefs = OUTCOME_MEASURES["LEFS"]
        assert lefs.direction == "up"  # higher is better

    def test_direction_down(self):
        odi = OUTCOME_MEASURES["ODI"]
        assert odi.direction == "down"  # lower is better


# ---------------------------------------------------------------------------
# Recording scores
# ---------------------------------------------------------------------------

class TestRecordScore:
    def test_record_score(self, tracker):
        rec = tracker.record_score("pt-1", "LEFS", 45.0)
        assert isinstance(rec, ScoreRecord)
        assert rec.score == 45.0
        assert rec.measure_name == "LEFS"

    def test_score_clamped_to_range(self, tracker):
        rec = tracker.record_score("pt-1", "NPRS", 15.0)  # max 10
        assert rec.score == 10.0

    def test_score_clamped_below_min(self, tracker):
        rec = tracker.record_score("pt-1", "NPRS", -5.0)  # min 0
        assert rec.score == 0.0

    def test_unknown_measure_raises(self, tracker):
        with pytest.raises(ValueError, match="Unknown measure"):
            tracker.record_score("pt-1", "FAKE_MEASURE", 10.0)


# ---------------------------------------------------------------------------
# Progress retrieval
# ---------------------------------------------------------------------------

class TestGetProgress:
    def test_get_progress_ordered(self, tracker):
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 2, 1, tzinfo=timezone.utc)
        tracker.record_score("pt-1", "LEFS", 40.0, recorded_at=t1)
        tracker.record_score("pt-1", "LEFS", 55.0, recorded_at=t2)
        progress = tracker.get_progress("pt-1", "LEFS")
        assert len(progress) == 2
        assert progress[0].score == 40.0
        assert progress[1].score == 55.0

    def test_get_progress_empty(self, tracker):
        assert tracker.get_progress("pt-1", "ODI") == []


# ---------------------------------------------------------------------------
# MCID / MDC checks
# ---------------------------------------------------------------------------

class TestCheckMCID:
    def test_mcid_met(self, tracker):
        """LEFS MCID=9.0 — delta of 10 should meet it."""
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 1, tzinfo=timezone.utc)
        tracker.record_score("pt-1", "LEFS", 40.0, recorded_at=t1)
        tracker.record_score("pt-1", "LEFS", 50.0, recorded_at=t2)
        result = tracker.check_mcid("pt-1", "LEFS")
        assert result["met_mcid"] is True
        assert result["delta"] == 10.0
        assert result["direction"] == "up"

    def test_mcid_not_met(self, tracker):
        """LEFS MCID=9.0 — delta of 5 should not meet it."""
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 1, tzinfo=timezone.utc)
        tracker.record_score("pt-1", "LEFS", 40.0, recorded_at=t1)
        tracker.record_score("pt-1", "LEFS", 45.0, recorded_at=t2)
        result = tracker.check_mcid("pt-1", "LEFS")
        assert result["met_mcid"] is False
        assert result["delta"] == 5.0

    def test_mdc_met(self, tracker):
        """Berg MDC=5.0 — delta of 6 should meet it."""
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 1, tzinfo=timezone.utc)
        tracker.record_score("pt-1", "Berg", 30.0, recorded_at=t1)
        tracker.record_score("pt-1", "Berg", 36.0, recorded_at=t2)
        result = tracker.check_mcid("pt-1", "Berg")
        assert result["met_mdc"] is True

    def test_mcid_down_direction(self, tracker):
        """ODI direction=down — positive delta means baseline > latest."""
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 1, tzinfo=timezone.utc)
        tracker.record_score("pt-1", "ODI", 50.0, recorded_at=t1)
        tracker.record_score("pt-1", "ODI", 35.0, recorded_at=t2)
        result = tracker.check_mcid("pt-1", "ODI")
        assert result["met_mcid"] is True
        assert result["delta"] == 15.0  # 50 - 35

    def test_mcid_single_score(self, tracker):
        tracker.record_score("pt-1", "NPRS", 7.0)
        result = tracker.check_mcid("pt-1", "NPRS")
        assert result["met_mcid"] is False
        assert result["scores_recorded"] == 1


# ---------------------------------------------------------------------------
# Functional summary
# ---------------------------------------------------------------------------

class TestFunctionalSummary:
    def test_functional_summary(self, tracker):
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 1, tzinfo=timezone.utc)
        tracker.record_score("pt-1", "LEFS", 40.0, recorded_at=t1)
        tracker.record_score("pt-1", "LEFS", 55.0, recorded_at=t2)
        tracker.record_score("pt-1", "NPRS", 7.0, recorded_at=t1)
        tracker.record_score("pt-1", "NPRS", 4.0, recorded_at=t2)
        summary = tracker.get_functional_summary("pt-1")
        assert "LEFS" in summary
        assert "NPRS" in summary
        assert summary["LEFS"]["trend"] == "improving"
        assert summary["NPRS"]["trend"] == "improving"
        assert summary["LEFS"]["met_mcid"] is True  # delta 15 >= 9

    def test_functional_summary_empty(self, tracker):
        assert tracker.get_functional_summary("pt-none") == {}


# ---------------------------------------------------------------------------
# Suggest measures
# ---------------------------------------------------------------------------

class TestSuggestMeasures:
    def test_suggest_measures_pt(self, tracker):
        suggestions = tracker.suggest_measures("PT")
        measure_names = {s["measure"] for s in suggestions}
        assert "LEFS" in measure_names
        assert "ODI" in measure_names
        assert len(suggestions) > 5

    def test_suggest_measures_ot(self, tracker):
        suggestions = tracker.suggest_measures("OT")
        measure_names = {s["measure"] for s in suggestions}
        assert "QuickDASH" in measure_names or "COPM" in measure_names

    def test_suggest_measures_slp(self, tracker):
        suggestions = tracker.suggest_measures("SLP")
        measure_names = {s["measure"] for s in suggestions}
        assert "FOIS" in measure_names
        assert "ASHA_NOMS" in measure_names

    def test_suggest_measures_by_diagnosis(self, tracker):
        suggestions = tracker.suggest_measures("PT", diagnosis="low back pain")
        measure_names = {s["measure"] for s in suggestions}
        assert "ODI" in measure_names
        assert "NPRS" in measure_names
