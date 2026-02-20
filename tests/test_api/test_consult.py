"""Tests for the consultation API endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.conftest import apply_auth_override
from rehab_os.api.routes.consult import router
from rehab_os.models.output import (
    ConsultationResponse,
    SafetyAssessment,
    DiagnosisResult,
    UrgencyLevel,
    RedFlag,
)
from rehab_os.models.patient import Discipline, CareSetting


@pytest.fixture
def mock_orchestrator_service():
    """Mock orchestrator stored in app.state."""
    return MagicMock()


@pytest.fixture
def app(mock_orchestrator_service):
    """Create a minimal FastAPI app with the consult router."""
    # Reset the rate limiter so tests don't interfere with each other
    from rehab_os.api.routes.consult import _consult_limiter
    _consult_limiter._requests.clear()

    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1")
    test_app.state.orchestrator = mock_orchestrator_service
    apply_auth_override(test_app)
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


def _safe_response(**overrides) -> ConsultationResponse:
    defaults = dict(
        safety=SafetyAssessment(
            is_safe_to_treat=True,
            urgency_level=UrgencyLevel.ROUTINE,
            summary="Safe to treat.",
        ),
        processing_notes=["Safety screening: routine"],
    )
    defaults.update(overrides)
    return ConsultationResponse(**defaults)


class TestConsultEndpoint:
    """Tests for POST /api/v1/consult."""

    def test_successful_consult(self, client, mock_orchestrator_service):
        """Full consult returns 200 with expected fields."""
        resp = _safe_response(
            diagnosis=DiagnosisResult(
                primary_diagnosis="OA knee",
                icd_codes=["M17.11"],
                rationale="Post-TKA",
                confidence=0.92,
            ),
        )
        mock_orchestrator_service.process = AsyncMock(return_value=resp)

        r = client.post("/api/v1/consult", json={
            "query": "Evaluate post-TKA knee",
            "discipline": "PT",
            "setting": "inpatient",
        })

        assert r.status_code == 200
        data = r.json()
        assert data["safety"]["is_safe_to_treat"] is True
        assert data["diagnosis"]["primary_diagnosis"] == "OA knee"

    def test_safety_only(self, client, mock_orchestrator_service):
        """Safety-only task returns safety but no diagnosis/plan."""
        resp = _safe_response()
        mock_orchestrator_service.process = AsyncMock(return_value=resp)

        r = client.post("/api/v1/consult", json={
            "query": "Check safety",
            "task_type": "safety_only",
        })

        assert r.status_code == 200
        data = r.json()
        assert data["safety"] is not None
        assert data["diagnosis"] is None
        assert data["plan"] is None

    def test_critical_red_flags(self, client, mock_orchestrator_service):
        """Critical red flags stop the pipeline."""
        resp = ConsultationResponse(
            safety=SafetyAssessment(
                is_safe_to_treat=False,
                urgency_level=UrgencyLevel.EMERGENT,
                red_flags=[
                    RedFlag(
                        finding="Cauda equina",
                        description="Saddle anesthesia",
                        rationale="Classic presentation",
                        recommended_action="Immediate MRI",
                        urgency=UrgencyLevel.EMERGENT,
                    )
                ],
                summary="Critical.",
                referral_recommended=True,
                referral_to="ED",
            ),
            processing_notes=["CRITICAL RED FLAGS DETECTED"],
        )
        mock_orchestrator_service.process = AsyncMock(return_value=resp)

        r = client.post("/api/v1/consult", json={
            "query": "Low back pain with saddle anesthesia",
            "patient": {
                "age": 45,
                "sex": "male",
                "chief_complaint": "LBP with saddle anesthesia",
                "discipline": "PT",
                "setting": "outpatient",
            },
        })

        assert r.status_code == 200
        data = r.json()
        assert data["safety"]["is_safe_to_treat"] is False
        assert len(data["safety"]["red_flags"]) == 1
        assert data["plan"] is None

    def test_missing_patient_creates_default(self, client, mock_orchestrator_service):
        """When patient context is omitted, endpoint creates a default."""
        resp = _safe_response()
        mock_orchestrator_service.process = AsyncMock(return_value=resp)

        r = client.post("/api/v1/consult", json={"query": "General question"})

        assert r.status_code == 200
        # Orchestrator should have been called
        mock_orchestrator_service.process.assert_called_once()

    def test_invalid_request_missing_query(self, client):
        """Missing required 'query' field returns 422."""
        r = client.post("/api/v1/consult", json={"discipline": "PT"})
        assert r.status_code == 422

    def test_invalid_discipline(self, client):
        """Invalid discipline value returns 422."""
        r = client.post("/api/v1/consult", json={
            "query": "test",
            "discipline": "INVALID",
        })
        assert r.status_code == 422

    def test_orchestrator_exception_returns_500(self, client, mock_orchestrator_service):
        """Internal orchestrator error returns 500."""
        mock_orchestrator_service.process = AsyncMock(side_effect=RuntimeError("boom"))

        r = client.post("/api/v1/consult", json={"query": "test"})
        assert r.status_code == 500
