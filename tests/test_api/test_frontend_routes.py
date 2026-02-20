"""Tests for frontend-facing API routes (sessions, streaming, mobile)."""

import pytest
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import apply_auth_override
from rehab_os.api.routes import sessions, streaming, mobile
from rehab_os.models.output import (
    ConsultationResponse,
    SafetyAssessment,
    DiagnosisResult,
    UrgencyLevel,
)
from rehab_os.models.evidence import EvidenceSummary
from rehab_os.models.plan import PlanOfCare


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator."""
    orchestrator = MagicMock()
    orchestrator.process = AsyncMock()
    orchestrator._run_safety_check = AsyncMock()
    orchestrator._run_diagnosis_and_evidence = AsyncMock()
    orchestrator._run_planning = AsyncMock()
    orchestrator._run_outcome_selection = AsyncMock()
    orchestrator._run_qa_review = AsyncMock()
    return orchestrator


@pytest.fixture
def mock_llm_router():
    """Create mock LLM router."""
    llm = MagicMock()
    llm.complete = AsyncMock()
    return llm


@pytest.fixture
def sessions_app():
    """Create test app for sessions routes."""
    app = FastAPI()
    app.include_router(sessions.router, prefix="/api/v1")
    # Clear sessions between tests
    sessions._sessions.clear()
    apply_auth_override(app)
    return app


@pytest.fixture
def sessions_client(sessions_app):
    """Create test client for sessions."""
    return TestClient(sessions_app)


@pytest.fixture
def mobile_app(mock_orchestrator):
    """Create test app for mobile routes."""
    app = FastAPI()
    app.include_router(mobile.router, prefix="/api/v1")
    app.state.orchestrator = mock_orchestrator
    apply_auth_override(app)
    return app


@pytest.fixture
def mobile_client(mobile_app):
    """Create test client for mobile routes."""
    return TestClient(mobile_app)


@pytest.fixture
def streaming_app(mock_orchestrator):
    """Create test app for streaming routes."""
    app = FastAPI()
    app.include_router(streaming.router, prefix="/api/v1")
    app.state.orchestrator = mock_orchestrator
    apply_auth_override(app)
    return app


@pytest.fixture
def streaming_client(streaming_app):
    """Create test client for streaming routes."""
    return TestClient(streaming_app)


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    def test_create_session(self, sessions_client):
        """Test POST /api/v1/sessions/create creates a new session."""
        response = sessions_client.post(
            "/api/v1/sessions/create",
            json={
                "discipline": "PT",
                "care_setting": "outpatient",
                "metadata": {"source": "test"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["discipline"] == "PT"
        assert data["care_setting"] == "outpatient"
        assert data["consult_count"] == 0

    def test_create_session_with_user_id(self, sessions_client):
        """Test creating session with user ID."""
        response = sessions_client.post(
            "/api/v1/sessions/create",
            json={
                "user_id": "user-123",
                "discipline": "OT",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-123"

    def test_get_session(self, sessions_client):
        """Test GET /api/v1/sessions/{session_id} retrieves session."""
        # Create session first
        create_response = sessions_client.post(
            "/api/v1/sessions/create",
            json={"discipline": "PT"},
        )
        session_id = create_response.json()["session_id"]

        # Get session
        response = sessions_client.get(f"/api/v1/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id

    def test_get_nonexistent_session(self, sessions_client):
        """Test getting nonexistent session returns 404."""
        response = sessions_client.get("/api/v1/sessions/nonexistent")

        assert response.status_code == 404

    def test_add_consult_to_session(self, sessions_client):
        """Test POST /api/v1/sessions/{session_id}/consult records consultation."""
        # Create session first
        create_response = sessions_client.post(
            "/api/v1/sessions/create",
            json={"discipline": "PT"},
        )
        session_id = create_response.json()["session_id"]

        # Add consult
        response = sessions_client.post(
            f"/api/v1/sessions/{session_id}/consult",
            params={
                "consult_id": "consult-001",
                "query_summary": "Low back pain evaluation",
                "diagnosis": "M54.5 - Low back pain",
                "has_red_flags": False,
                "qa_score": 0.85,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"
        assert data["consult_count"] == 1

    def test_get_session_history(self, sessions_client):
        """Test GET /api/v1/sessions/{session_id}/history returns consultations."""
        # Create session and add consults
        create_response = sessions_client.post(
            "/api/v1/sessions/create",
            json={"discipline": "PT"},
        )
        session_id = create_response.json()["session_id"]

        # Add multiple consults
        for i in range(3):
            sessions_client.post(
                f"/api/v1/sessions/{session_id}/consult",
                params={
                    "consult_id": f"consult-{i:03d}",
                    "query_summary": f"Consultation {i}",
                },
            )

        # Get history
        response = sessions_client.get(f"/api/v1/sessions/{session_id}/history")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_get_session_history_with_limit(self, sessions_client):
        """Test history limit parameter."""
        create_response = sessions_client.post(
            "/api/v1/sessions/create",
            json={"discipline": "PT"},
        )
        session_id = create_response.json()["session_id"]

        # Add consults
        for i in range(5):
            sessions_client.post(
                f"/api/v1/sessions/{session_id}/consult",
                params={
                    "consult_id": f"consult-{i:03d}",
                    "query_summary": f"Consultation {i}",
                },
            )

        # Get limited history
        response = sessions_client.get(
            f"/api/v1/sessions/{session_id}/history",
            params={"limit": 2},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_end_session(self, sessions_client):
        """Test DELETE /api/v1/sessions/{session_id} ends session."""
        # Create session
        create_response = sessions_client.post(
            "/api/v1/sessions/create",
            json={"discipline": "PT"},
        )
        session_id = create_response.json()["session_id"]

        # End session
        response = sessions_client.delete(f"/api/v1/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ended"

        # Verify session is gone
        get_response = sessions_client.get(f"/api/v1/sessions/{session_id}")
        assert get_response.status_code == 404

    def test_get_session_logs(self, sessions_client, tmp_path):
        """Test GET /api/v1/sessions/{session_id}/logs returns logs."""
        # Create session
        create_response = sessions_client.post(
            "/api/v1/sessions/create",
            json={"discipline": "PT"},
        )
        session_id = create_response.json()["session_id"]

        # Get logs (will return empty since no log files exist)
        response = sessions_client.get(
            f"/api/v1/sessions/{session_id}/logs",
            params={"log_type": "orchestrator"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert "total" in data


class TestMobileEndpoints:
    """Tests for mobile-optimized endpoints."""

    def test_get_disciplines(self, mobile_client):
        """Test GET /api/v1/mobile/disciplines returns discipline list."""
        response = mobile_client.get("/api/v1/mobile/disciplines")

        assert response.status_code == 200
        data = response.json()
        assert "disciplines" in data
        assert len(data["disciplines"]) == 3

        codes = [d["code"] for d in data["disciplines"]]
        assert "PT" in codes
        assert "OT" in codes
        assert "SLP" in codes

    def test_get_care_settings(self, mobile_client):
        """Test GET /api/v1/mobile/settings returns care settings."""
        response = mobile_client.get("/api/v1/mobile/settings")

        assert response.status_code == 200
        data = response.json()
        assert "settings" in data
        assert len(data["settings"]) == 5

        codes = [s["code"] for s in data["settings"]]
        assert "outpatient" in codes
        assert "inpatient" in codes
        assert "acute" in codes

    def test_hep_endpoint_known_condition(self, mobile_client):
        """Test POST /api/v1/mobile/hep returns home exercise program."""
        response = mobile_client.post(
            "/api/v1/mobile/hep",
            json={
                "condition": "low back pain",
                "discipline": "PT",
                "difficulty_level": "moderate",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["condition"] == "low back pain"
        assert len(data["exercises"]) >= 1
        assert "precautions" in data
        assert "progression_criteria" in data

    def test_hep_endpoint_unknown_condition(self, mobile_client):
        """Test HEP endpoint with unknown condition returns generic exercises."""
        response = mobile_client.post(
            "/api/v1/mobile/hep",
            json={
                "condition": "unusual condition xyz",
                "discipline": "PT",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["exercises"]) >= 1

    def test_hep_knee_pain(self, mobile_client):
        """Test HEP endpoint for knee pain."""
        response = mobile_client.post(
            "/api/v1/mobile/hep",
            json={
                "condition": "knee pain after surgery",
                "discipline": "PT",
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Should match "knee pain" template
        exercises = [e["name"] for e in data["exercises"]]
        assert any("Quad" in name or "Leg" in name or "Heel" in name for name in exercises)

    @pytest.mark.asyncio
    async def test_quick_consult_endpoint(self, mobile_app, mock_orchestrator):
        """Test POST /api/v1/mobile/quick-consult returns condensed response."""
        from rehab_os.models.output import ConsultationResponse, SafetyAssessment, DiagnosisResult, UrgencyLevel
        from rehab_os.models.evidence import EvidenceSummary
        from rehab_os.models.plan import PlanOfCare, Intervention

        mock_response = ConsultationResponse(
            safety=SafetyAssessment(
                is_safe_to_treat=True,
                urgency_level=UrgencyLevel.ROUTINE,
                summary="No red flags",
            ),
            diagnosis=DiagnosisResult(
                primary_diagnosis="Low back pain",
                icd_codes=["M54.5"],
                rationale="Clinical findings consistent with mechanical LBP",
                confidence=0.85,
            ),
            evidence=EvidenceSummary(
                query="low back pain",
                total_sources=3,
            ),
            plan=PlanOfCare(
                clinical_summary="Acute LBP",
                clinical_impression="Mechanical",
                prognosis="Good",
                rehab_potential="Good",
                visit_frequency="2x/week",
                expected_duration="4 weeks",
                interventions=[
                    Intervention(
                        name="Manual therapy",
                        category="manual_therapy",
                        description="Soft tissue mobilization",
                        rationale="Pain reduction",
                    ),
                    Intervention(
                        name="Therapeutic exercise",
                        category="therapeutic_exercise",
                        description="Core stabilization exercises",
                        rationale="Core stability",
                    ),
                ],
            ),
        )

        with patch("rehab_os.llm.create_router_from_settings") as mock_create, \
             patch("rehab_os.agents.Orchestrator") as mock_orch_class:
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm
            mock_orch = MagicMock()
            mock_orch.process = AsyncMock(return_value=mock_response)
            mock_orch_class.return_value = mock_orch

            client = TestClient(mobile_app)
            response = client.post(
                "/api/v1/mobile/quick-consult",
                json={
                    "chief_complaint": "Lower back pain, 3 days",
                    "age": 45,
                    "discipline": "PT",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "consult_id" in data
            assert data["is_safe"] is True
            assert data["diagnosis"] == "Low back pain"
            assert len(data["key_interventions"]) <= 3

    @pytest.mark.asyncio
    async def test_safety_check_endpoint(self, mobile_app, mock_orchestrator):
        """Test POST /api/v1/mobile/safety-check returns safety assessment."""
        from rehab_os.agents.red_flag import SafetyAssessment
        from rehab_os.models.output import UrgencyLevel

        mock_safety = SafetyAssessment(
            is_safe_to_treat=True,
            urgency_level=UrgencyLevel.ROUTINE,
            summary="No red flags identified",
        )

        with patch("rehab_os.llm.create_router_from_settings") as mock_create, \
             patch("rehab_os.agents.red_flag.RedFlagAgent") as mock_agent_class:
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(return_value=mock_safety)
            mock_agent_class.return_value = mock_agent

            client = TestClient(mobile_app)
            response = client.post(
                "/api/v1/mobile/safety-check",
                json={
                    "symptoms": ["knee pain", "swelling"],
                    "discipline": "PT",
                    "age": 50,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["is_safe"] is True
            assert "recommendation" in data


class TestStreamingEndpoints:
    """Tests for WebSocket streaming endpoints."""

    def test_websocket_status(self, streaming_client):
        """Test GET /api/v1/ws/status returns connection info."""
        response = streaming_client.get("/api/v1/ws/status")

        assert response.status_code == 200
        data = response.json()
        assert "active_connections" in data
        assert "connection_ids" in data

    def test_websocket_connect_disconnect(self, streaming_app):
        """Test WebSocket connection and disconnection."""
        client = TestClient(streaming_app)

        with client.websocket_connect("/api/v1/ws/consult/test-client-123") as websocket:
            # Connection should be established
            pass

        # After disconnect, check status
        response = client.get("/api/v1/ws/status")
        data = response.json()
        assert "test-client-123" not in data["connection_ids"]

    def test_websocket_invalid_json(self, streaming_app):
        """Test WebSocket handles invalid JSON gracefully."""
        client = TestClient(streaming_app)

        with client.websocket_connect("/api/v1/ws/consult/test-client") as websocket:
            websocket.send_text("not valid json")
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Invalid JSON" in response["message"]


class TestConnectionManager:
    """Tests for WebSocket ConnectionManager."""

    def test_connection_manager_init(self):
        """Test ConnectionManager initialization."""
        manager = streaming.ConnectionManager()
        assert len(manager.active_connections) == 0

    @pytest.mark.asyncio
    async def test_connection_manager_send_json(self):
        """Test ConnectionManager send_json to non-existent client."""
        manager = streaming.ConnectionManager()
        # Should not raise when client doesn't exist
        await manager.send_json("nonexistent", {"test": "data"})

    @pytest.mark.asyncio
    async def test_connection_manager_disconnect(self):
        """Test ConnectionManager disconnect."""
        manager = streaming.ConnectionManager()
        # Should not raise when client doesn't exist
        manager.disconnect("nonexistent")


class TestMobileModels:
    """Tests for mobile API models."""

    def test_quick_consult_request_defaults(self):
        """Test QuickConsultRequest default values."""
        from rehab_os.api.routes.mobile import QuickConsultRequest

        request = QuickConsultRequest(chief_complaint="Test")
        assert request.age == 50
        assert request.discipline == "PT"
        assert request.setting == "outpatient"
        assert request.red_flag_symptoms == []

    def test_quick_consult_response_fields(self):
        """Test QuickConsultResponse required fields."""
        from rehab_os.api.routes.mobile import QuickConsultResponse

        response = QuickConsultResponse(
            consult_id="test-123",
            is_safe=True,
            urgency="routine",
            diagnosis="Test diagnosis",
            icd_code="Z99.9",
            confidence=0.85,
            key_interventions=["Exercise", "Manual therapy"],
            visit_frequency="2x/week",
        )

        assert response.consult_id == "test-123"
        assert response.red_flag_alert is None
        assert response.patient_instructions is None

    def test_hep_request_defaults(self):
        """Test HEPRequest default values."""
        from rehab_os.api.routes.mobile import HEPRequest

        request = HEPRequest(condition="Test condition")
        assert request.discipline == "PT"
        assert request.difficulty_level == "moderate"
        assert request.equipment_available == []

    def test_exercise_model(self):
        """Test Exercise model."""
        from rehab_os.api.routes.mobile import Exercise

        exercise = Exercise(
            name="Quad Sets",
            description="Tighten thigh muscle",
            sets=3,
            reps="10 holds",
            frequency="2x daily",
        )

        assert exercise.name == "Quad Sets"
        assert exercise.precautions is None
        assert exercise.image_url is None


class TestSessionModels:
    """Tests for session API models."""

    def test_session_create_defaults(self):
        """Test SessionCreate default values."""
        from rehab_os.api.routes.sessions import SessionCreate

        request = SessionCreate()
        assert request.user_id is None
        assert request.discipline == "PT"
        assert request.care_setting == "outpatient"
        assert request.metadata == {}

    def test_consult_history_item(self):
        """Test ConsultHistoryItem model."""
        from rehab_os.api.routes.sessions import ConsultHistoryItem

        item = ConsultHistoryItem(
            consult_id="c-001",
            timestamp="2025-01-01T00:00:00",
            query_summary="Back pain evaluation",
        )

        assert item.diagnosis is None
        assert item.has_red_flags is False
        assert item.qa_score is None
