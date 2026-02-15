"""Tests for SessionMemoryService."""

from unittest.mock import MagicMock, patch

import pytest

from rehab_os.memory.session_memory import SessionMemoryService, _NAMESPACE_PREFIX
from rehab_os.models.output import (
    ConsultationResponse,
    DiagnosisResult,
    OutcomeRecommendations,
    OutcomeMeasure,
    QAResult,
    SafetyAssessment,
    UrgencyLevel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_disabled():
    """SessionMemoryService with memU disabled (pure in-memory)."""
    return SessionMemoryService(enabled=False)


@pytest.fixture
def mock_consultation() -> ConsultationResponse:
    """Minimal consultation response for testing."""
    return ConsultationResponse(
        safety=SafetyAssessment(
            is_safe_to_treat=True,
            urgency_level=UrgencyLevel.ROUTINE,
            summary="No concerns",
        ),
        diagnosis=DiagnosisResult(
            primary_diagnosis="Lumbar radiculopathy",
            icd_codes=["M54.17"],
            rationale="Consistent with presentation",
            confidence=0.85,
        ),
        outcomes=OutcomeRecommendations(
            primary_measures=[
                OutcomeMeasure(
                    name="Oswestry Disability Index",
                    description="Low back function",
                    rationale="Standard for LBP",
                    frequency="Every 4 weeks",
                )
            ],
            reassessment_schedule="Every 4 weeks",
            rationale="Standard measures",
        ),
        qa_review=QAResult(
            overall_quality=0.9,
            strengths=["Good evidence"],
        ),
        processing_notes=["Test consultation"],
    )


# ---------------------------------------------------------------------------
# Tests: in-memory fallback
# ---------------------------------------------------------------------------


class TestInMemoryFallback:
    def test_init_disabled(self, memory_disabled: SessionMemoryService):
        assert not memory_disabled.is_memu_available

    def test_store_and_retrieve(
        self,
        memory_disabled: SessionMemoryService,
        mock_consultation: ConsultationResponse,
    ):
        memory_disabled.store_consultation("pt-001", mock_consultation)
        history = memory_disabled.get_patient_history("pt-001")

        assert len(history) == 1
        assert history[0]["primary_diagnosis"] == "Lumbar radiculopathy"
        assert history[0]["confidence"] == 0.85
        assert history[0]["icd_codes"] == ["M54.17"]

    def test_empty_history(self, memory_disabled: SessionMemoryService):
        assert memory_disabled.get_patient_history("nonexistent") == []

    def test_multiple_consultations(
        self,
        memory_disabled: SessionMemoryService,
        mock_consultation: ConsultationResponse,
    ):
        memory_disabled.store_consultation("pt-001", mock_consultation)
        memory_disabled.store_consultation("pt-001", mock_consultation)
        assert len(memory_disabled.get_patient_history("pt-001")) == 2


# ---------------------------------------------------------------------------
# Tests: namespace prefixing
# ---------------------------------------------------------------------------


class TestNamespacePrefixing:
    def test_prefix_added(self):
        assert SessionMemoryService._scoped_id("patient-1") == "rehab:patient-1"

    def test_no_double_prefix(self):
        assert SessionMemoryService._scoped_id("rehab:patient-1") == "rehab:patient-1"

    def test_prefix_value(self):
        assert _NAMESPACE_PREFIX == "rehab:"


# ---------------------------------------------------------------------------
# Tests: longitudinal context formatting
# ---------------------------------------------------------------------------


class TestLongitudinalContext:
    def test_empty_returns_empty_string(self, memory_disabled: SessionMemoryService):
        assert memory_disabled.get_longitudinal_context("unknown") == ""

    def test_formatted_output(
        self,
        memory_disabled: SessionMemoryService,
        mock_consultation: ConsultationResponse,
    ):
        memory_disabled.store_consultation("pt-002", mock_consultation)
        ctx = memory_disabled.get_longitudinal_context("pt-002")

        assert "Longitudinal Context" in ctx
        assert "1 prior visit" in ctx
        assert "Lumbar radiculopathy" in ctx
        assert "85%" in ctx
        assert "End Longitudinal Context" in ctx

    def test_max_10_visits(self, memory_disabled: SessionMemoryService):
        """Only last 10 visits should appear."""
        minimal = ConsultationResponse(
            safety=SafetyAssessment(
                is_safe_to_treat=True,
                urgency_level=UrgencyLevel.ROUTINE,
                summary="ok",
            ),
        )
        for _ in range(15):
            memory_disabled.store_consultation("pt-many", minimal)

        ctx = memory_disabled.get_longitudinal_context("pt-many")
        assert "15 prior visit" in ctx
        # Should only have 10 visit detail blocks
        assert ctx.count("--- Visit") == 10


# ---------------------------------------------------------------------------
# Tests: outcome tracking
# ---------------------------------------------------------------------------


class TestOutcomeTracking:
    def test_store_and_trend(self, memory_disabled: SessionMemoryService):
        memory_disabled.store_outcome("pt-003", {"measure_name": "ODI", "value": 45})
        memory_disabled.store_outcome("pt-003", {"measure_name": "ODI", "value": 30})

        trends = memory_disabled.get_outcome_trends("pt-003")
        assert "ODI" in trends
        assert len(trends["ODI"]) == 2
        assert trends["ODI"][0]["value"] == 45
        assert trends["ODI"][1]["value"] == 30

    def test_empty_trends(self, memory_disabled: SessionMemoryService):
        assert memory_disabled.get_outcome_trends("nobody") == {}


# ---------------------------------------------------------------------------
# Tests: memU initialization (mocked)
# ---------------------------------------------------------------------------


class TestMemUInit:
    @patch("rehab_os.memory.session_memory.SessionMemoryService._init_memu")
    def test_init_called_when_enabled(self, mock_init):
        SessionMemoryService(enabled=True)
        mock_init.assert_called_once()

    @patch("rehab_os.memory.session_memory.SessionMemoryService._init_memu")
    def test_init_not_called_when_disabled(self, mock_init):
        SessionMemoryService(enabled=False)
        mock_init.assert_not_called()
