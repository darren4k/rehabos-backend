"""Tests for cross-namespace memU queries."""

from unittest.mock import AsyncMock, MagicMock, patch
import json
import pytest

from rehab_os.memory.cross_namespace import (
    EncounterRecord,
    get_patient_history,
    format_cross_namespace_context,
    _parse_timestamp,
)
from rehab_os.memory.session_memory import SessionMemoryService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_disabled():
    return SessionMemoryService(enabled=False)


@pytest.fixture
def memory_with_cache(memory_disabled):
    """Memory service with some in-memory cached records."""
    memory_disabled._cache["PT001"] = [
        {
            "timestamp": "2026-02-10T09:00:00+00:00",
            "patient_id": "PT001",
            "primary_diagnosis": "Lumbar radiculopathy",
            "confidence": 0.85,
            "goals": ["Reduce pain", "Improve mobility"],
        },
        {
            "timestamp": "2026-02-12T10:00:00+00:00",
            "patient_id": "PT001",
            "primary_diagnosis": "Lumbar radiculopathy",
            "confidence": 0.90,
            "interventions": ["Therapeutic exercise", "Manual therapy"],
        },
    ]
    return memory_disabled


@pytest.fixture
def mock_memu_service():
    """A mock memU MemoryService."""
    svc = MagicMock()
    svc.retrieve = AsyncMock()
    return svc


@pytest.fixture
def memory_with_memu(mock_memu_service):
    """Memory service with mocked memU backend."""
    mem = SessionMemoryService(enabled=False)
    mem._memu_available = True
    mem._memu_service = mock_memu_service
    return mem


# ---------------------------------------------------------------------------
# Tests: get_patient_history
# ---------------------------------------------------------------------------


class TestGetPatientHistory:
    def test_fallback_rehab_namespace_only(self, memory_with_cache):
        """When memU unavailable, returns in-memory cache for rehab namespace."""
        records = get_patient_history(memory_with_cache, "PT001", namespaces=["rehab"])
        assert len(records) == 2
        assert all(r.namespace == "rehab" for r in records)
        assert records[0].data["primary_diagnosis"] == "Lumbar radiculopathy"

    def test_fallback_docpilot_returns_empty(self, memory_with_cache):
        """When memU unavailable, docpilot namespace returns nothing."""
        records = get_patient_history(memory_with_cache, "PT001", namespaces=["docpilot"])
        assert records == []

    def test_combined_fallback(self, memory_with_cache):
        """Combined query returns rehab cache + empty docpilot."""
        records = get_patient_history(memory_with_cache, "PT001")
        assert len(records) == 2

    def test_unknown_patient_returns_empty(self, memory_disabled):
        records = get_patient_history(memory_disabled, "UNKNOWN")
        assert records == []

    def test_sorted_by_timestamp(self, memory_with_cache):
        records = get_patient_history(memory_with_cache, "PT001")
        timestamps = [r.timestamp for r in records]
        assert timestamps == sorted(timestamps)

    def test_memu_query_both_namespaces(self, memory_with_memu, mock_memu_service):
        """When memU available, queries both namespaces."""
        rehab_items = {
            "items": [
                {
                    "summary": json.dumps({
                        "timestamp": "2026-02-10T09:00:00+00:00",
                        "patient_id": "PT001",
                        "primary_diagnosis": "LBP",
                    }),
                }
            ]
        }
        docpilot_items = {
            "items": [
                {
                    "summary": json.dumps({
                        "timestamp": "2026-02-08T14:00:00+00:00",
                        "patient_id": "PT001",
                        "primary_diagnosis": "Acute LBP",
                    }),
                }
            ]
        }

        async def fake_retrieve(queries, where, method):
            if where["user_id"] == "rehab:PT001":
                return rehab_items
            elif where["user_id"] == "docpilot:PT001":
                return docpilot_items
            return {"items": []}

        mock_memu_service.retrieve = AsyncMock(side_effect=fake_retrieve)

        records = get_patient_history(memory_with_memu, "PT001")
        assert len(records) == 2
        # DocPilot encounter is older, should come first
        assert records[0].namespace == "docpilot"
        assert records[1].namespace == "rehab"

    def test_memu_query_error_returns_empty(self, memory_with_memu, mock_memu_service):
        """memU errors are caught gracefully."""
        mock_memu_service.retrieve = AsyncMock(side_effect=RuntimeError("connection lost"))
        records = get_patient_history(memory_with_memu, "PT001")
        assert records == []


# ---------------------------------------------------------------------------
# Tests: format_cross_namespace_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_empty_records(self):
        assert format_cross_namespace_context([]) == ""

    def test_basic_formatting(self):
        records = [
            EncounterRecord(
                namespace="rehab",
                timestamp="2026-02-10T09:00:00+00:00",
                patient_id="PT001",
                data={"primary_diagnosis": "LBP", "confidence": 0.85},
            ),
            EncounterRecord(
                namespace="docpilot",
                timestamp="2026-02-12T09:00:00+00:00",
                patient_id="PT001",
                data={"primary_diagnosis": "Chronic LBP", "goals": ["Pain mgmt"]},
            ),
        ]
        result = format_cross_namespace_context(records)
        assert "REHAB" in result
        assert "DOCPILOT" in result
        assert "LBP" in result
        assert "Cross-Service Patient History" in result

    def test_max_records_limit(self):
        records = [
            EncounterRecord(
                namespace="rehab",
                timestamp=f"2026-02-{i:02d}T09:00:00+00:00",
                patient_id="PT001",
                data={"summary": f"visit {i}"},
            )
            for i in range(1, 16)
        ]
        result = format_cross_namespace_context(records, max_records=5)
        assert "showing last 5" in result


# ---------------------------------------------------------------------------
# Tests: _parse_timestamp
# ---------------------------------------------------------------------------


class TestParseTimestamp:
    def test_iso_with_tz(self):
        dt = _parse_timestamp("2026-02-10T09:00:00+00:00")
        assert dt.year == 2026

    def test_iso_without_tz(self):
        dt = _parse_timestamp("2026-02-10T09:00:00")
        assert dt.month == 2

    def test_invalid_returns_min(self):
        from datetime import datetime
        dt = _parse_timestamp("not-a-date")
        assert dt == datetime.min


# ---------------------------------------------------------------------------
# Tests: EncounterRecord
# ---------------------------------------------------------------------------


class TestEncounterRecord:
    def test_to_dict(self):
        rec = EncounterRecord(
            namespace="rehab",
            timestamp="2026-02-10T09:00:00",
            patient_id="PT001",
            data={"dx": "LBP"},
        )
        d = rec.to_dict()
        assert d["namespace"] == "rehab"
        assert d["data"]["dx"] == "LBP"
