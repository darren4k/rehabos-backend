"""Tests for rehab_os.clinical.flow_sheets — longitudinal flow sheet tracking."""
from __future__ import annotations

import pytest

from rehab_os.clinical.flow_sheets import (
    FlowSheetEntry,
    FlowSheetService,
    OT_COLUMNS,
    PT_COLUMNS,
    SLP_COLUMNS,
)


@pytest.fixture
def svc() -> FlowSheetService:
    return FlowSheetService()


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

class TestGetColumns:
    def test_get_columns_pt(self, svc):
        cols = svc.get_columns("PT")
        assert len(cols) == 25
        keys = {c.key for c in cols}
        assert "rom_flex" in keys
        assert "pain_current" in keys
        assert "tug" in keys

    def test_get_columns_ot(self, svc):
        cols = svc.get_columns("OT")
        assert len(cols) == len(OT_COLUMNS)
        keys = {c.key for c in cols}
        assert "grip_strength_r" in keys
        assert "adl_feeding" in keys

    def test_get_columns_slp(self, svc):
        cols = svc.get_columns("SLP")
        assert len(cols) == len(SLP_COLUMNS)
        keys = {c.key for c in cols}
        assert "swallow_liquid" in keys
        assert "intelligibility" in keys
        assert "cognition_problem_solving" in keys

    def test_get_columns_case_insensitive(self, svc):
        assert len(svc.get_columns("pt")) == 25
        assert len(svc.get_columns("Pt")) == 25

    def test_get_columns_invalid_raises(self, svc):
        with pytest.raises(ValueError, match="Unknown discipline"):
            svc.get_columns("XYZ")


# ---------------------------------------------------------------------------
# Recording entries
# ---------------------------------------------------------------------------

class TestRecordEntry:
    def test_record_entry(self, svc):
        entry = svc.record_entry(
            patient_id="pt-1",
            encounter_id="enc-1",
            encounter_date="2026-03-01",
            provider_id="prov-1",
            data={"rom_flex": 95, "pain_current": 4},
        )
        assert isinstance(entry, FlowSheetEntry)
        assert entry.encounter_id == "enc-1"
        assert entry.data["rom_flex"] == 95

    def test_multiple_entries_sorted(self, svc):
        svc.record_entry("pt-1", "enc-2", "2026-03-05", "prov-1", {"rom_flex": 100})
        svc.record_entry("pt-1", "enc-1", "2026-03-01", "prov-1", {"rom_flex": 90})
        entries = svc.get_flow_sheet("pt-1")
        assert entries[0].encounter_date == "2026-03-01"
        assert entries[1].encounter_date == "2026-03-05"


# ---------------------------------------------------------------------------
# Retrieval & filtering
# ---------------------------------------------------------------------------

class TestGetFlowSheet:
    def test_get_flow_sheet_in_date_order(self, svc):
        svc.record_entry("pt-1", "enc-a", "2026-01-10", "prov-1", {"rom_flex": 80})
        svc.record_entry("pt-1", "enc-b", "2026-02-10", "prov-1", {"rom_flex": 90})
        svc.record_entry("pt-1", "enc-c", "2026-03-10", "prov-1", {"rom_flex": 100})
        entries = svc.get_flow_sheet("pt-1")
        dates = [e.encounter_date for e in entries]
        assert dates == sorted(dates)

    def test_get_flow_sheet_date_filter(self, svc):
        svc.record_entry("pt-1", "enc-a", "2026-01-10", "prov-1", {"rom_flex": 80})
        svc.record_entry("pt-1", "enc-b", "2026-02-10", "prov-1", {"rom_flex": 90})
        svc.record_entry("pt-1", "enc-c", "2026-03-10", "prov-1", {"rom_flex": 100})
        entries = svc.get_flow_sheet("pt-1", date_from="2026-02-01", date_to="2026-02-28")
        assert len(entries) == 1
        assert entries[0].encounter_id == "enc-b"

    def test_get_flow_sheet_empty_patient(self, svc):
        assert svc.get_flow_sheet("nonexistent") == []


# ---------------------------------------------------------------------------
# Trending
# ---------------------------------------------------------------------------

class TestTrendingData:
    def test_get_trending_data(self, svc):
        svc.record_entry("pt-1", "enc-1", "2026-01-01", "prov-1", {"rom_flex": 80})
        svc.record_entry("pt-1", "enc-2", "2026-02-01", "prov-1", {"rom_flex": 95})
        svc.record_entry("pt-1", "enc-3", "2026-03-01", "prov-1", {"rom_flex": 110})
        trend = svc.get_trending_data("pt-1", "rom_flex")
        assert len(trend) == 3
        assert trend[0]["value"] == 80
        assert trend[2]["value"] == 110
        assert "date" in trend[0]
        assert "encounter_id" in trend[0]

    def test_get_trending_data_missing_column(self, svc):
        svc.record_entry("pt-1", "enc-1", "2026-01-01", "prov-1", {"rom_flex": 80})
        trend = svc.get_trending_data("pt-1", "nonexistent_col")
        assert trend == []


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestGetSummary:
    def test_get_summary_with_data(self, svc):
        svc.record_entry("pt-1", "enc-1", "2026-01-01", "prov-1",
                          {"rom_flex": 80, "pain_current": 7})
        svc.record_entry("pt-1", "enc-2", "2026-03-01", "prov-1",
                          {"rom_flex": 110, "pain_current": 3})
        summary = svc.get_summary("pt-1", "PT")
        assert summary["total_entries"] == 2
        # ROM improving (higher is better)
        rom_col = summary["columns"]["rom_flex"]
        assert rom_col["first_value"] == 80
        assert rom_col["last_value"] == 110
        assert rom_col["trend"] == "improving"
        # Pain improving (lower is better, category=pain)
        pain_col = summary["columns"]["pain_current"]
        assert pain_col["first_value"] == 7
        assert pain_col["last_value"] == 3
        assert pain_col["trend"] == "improving"

    def test_get_summary_empty(self, svc):
        summary = svc.get_summary("pt-none", "PT")
        assert summary["total_entries"] == 0
        assert summary["columns"] == {}
