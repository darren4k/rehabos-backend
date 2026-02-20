"""API endpoint integration tests."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import apply_auth_override
from rehab_os.api.routes import consult, agents, health
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
    orchestrator.run_single_agent = AsyncMock()
    return orchestrator


@pytest.fixture
def mock_app_state(mock_orchestrator):
    """Create mock app state with all dependencies."""
    llm_router = MagicMock()
    llm_router.health_check = AsyncMock(return_value={"primary": True, "fallback": True})

    vector_store = MagicMock()
    vector_store.count = 150

    guideline_repo = MagicMock()

    return {
        "orchestrator": mock_orchestrator,
        "llm_router": llm_router,
        "vector_store": vector_store,
        "guideline_repo": guideline_repo,
    }


@pytest.fixture
def test_app(mock_app_state):
    """Create test FastAPI application."""
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(consult.router, prefix="/api/v1")
    app.include_router(agents.router, prefix="/api/v1/agents")

    # Set up app state
    app.state.orchestrator = mock_app_state["orchestrator"]
    app.state.llm_router = mock_app_state["llm_router"]
    app.state.vector_store = mock_app_state["vector_store"]
    app.state.guideline_repo = mock_app_state["guideline_repo"]

    apply_auth_override(app)
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def mock_consultation_response():
    """Create a complete consultation response."""
    return ConsultationResponse(
        safety=SafetyAssessment(
            is_safe_to_treat=True,
            urgency_level=UrgencyLevel.ROUTINE,
            summary="No red flags. Safe to proceed.",
        ),
        diagnosis=DiagnosisResult(
            primary_diagnosis="Low back pain",
            icd_codes=["M54.5"],
            rationale="Clinical findings consistent with mechanical LBP",
            confidence=0.85,
        ),
        evidence=EvidenceSummary(
            query="low back pain treatment",
            total_sources=3,
            synthesis="Exercise therapy recommended",
        ),
        plan=PlanOfCare(
            clinical_summary="45yo with chronic LBP",
            clinical_impression="Movement impairment",
            prognosis="Good",
            rehab_potential="Good",
            visit_frequency="2x/week",
            expected_duration="6 weeks",
        ),
        processing_notes=["Safety: routine", "Diagnosis confidence: 0.85"],
    )


class TestConsultEndpoints:
    """Tests for consultation endpoints."""

    def test_consult_endpoint_success(
        self, client, mock_orchestrator, mock_consultation_response
    ):
        """Test POST /api/v1/consult returns consultation response."""
        mock_orchestrator.process.return_value = mock_consultation_response

        response = client.post(
            "/api/v1/consult",
            json={
                "query": "Evaluate patient with low back pain",
                "discipline": "PT",
                "setting": "outpatient",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["safety"]["is_safe_to_treat"] is True
        assert data["diagnosis"]["primary_diagnosis"] == "Low back pain"
        assert "disclaimer" in data

    def test_consult_with_patient_context(
        self, client, mock_orchestrator, mock_consultation_response
    ):
        """Test consultation with full patient context."""
        mock_orchestrator.process.return_value = mock_consultation_response

        response = client.post(
            "/api/v1/consult",
            json={
                "query": "Evaluate for PT",
                "patient": {
                    "age": 55,
                    "sex": "male",
                    "chief_complaint": "Knee pain after fall",
                    "discipline": "PT",
                    "setting": "outpatient",
                },
                "discipline": "PT",
                "setting": "outpatient",
            },
        )

        assert response.status_code == 200

    def test_quick_consult_endpoint(
        self, client, mock_orchestrator, mock_consultation_response
    ):
        """Test POST /api/v1/consult/quick for fast consultations."""
        mock_orchestrator.process.return_value = mock_consultation_response

        response = client.post(
            "/api/v1/consult/quick",
            json={
                "query": "Low back pain, acute onset",
                "age": 40,
                "sex": "female",
                "discipline": "PT",
            },
        )

        assert response.status_code == 200
        # Quick consult should have been called with skip_qa=True
        mock_orchestrator.process.assert_called_once()
        call_args = mock_orchestrator.process.call_args
        assert call_args.kwargs.get("skip_qa") is True

    def test_safety_check_endpoint(
        self, client, mock_orchestrator, mock_consultation_response
    ):
        """Test POST /api/v1/consult/safety for safety-only check."""
        mock_orchestrator.process.return_value = mock_consultation_response

        response = client.post(
            "/api/v1/consult/safety",
            json={
                "query": "Patient with chest pain and shortness of breath",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "safety" in data

    def test_consult_with_documentation(
        self, client, mock_orchestrator, mock_consultation_response
    ):
        """Test consultation with documentation generation."""
        mock_orchestrator.process.return_value = mock_consultation_response

        response = client.post(
            "/api/v1/consult",
            json={
                "query": "Evaluate patient",
                "include_documentation": True,
                "documentation_type": "initial_evaluation",
            },
        )

        assert response.status_code == 200

    def test_consult_invalid_discipline(self, client):
        """Test that invalid discipline returns error."""
        response = client.post(
            "/api/v1/consult",
            json={
                "query": "Test query",
                "discipline": "INVALID",
            },
        )

        assert response.status_code == 422  # Validation error


class TestAgentEndpoints:
    """Tests for direct agent access endpoints."""

    def test_list_agents(self, client):
        """Test GET /api/v1/agents/list returns agent list."""
        response = client.get("/api/v1/agents/list")

        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert len(data["agents"]) == 7  # All 7 agents

        agent_names = [a["name"] for a in data["agents"]]
        assert "red_flag" in agent_names
        assert "diagnosis" in agent_names
        assert "evidence" in agent_names
        assert "plan" in agent_names
        assert "outcome" in agent_names
        assert "documentation" in agent_names
        assert "qa" in agent_names

    def test_run_single_agent(self, client, mock_orchestrator):
        """Test POST /api/v1/agents/{agent_name} runs single agent."""
        mock_result = DiagnosisResult(
            primary_diagnosis="Test diagnosis",
            icd_codes=["Z99.9"],
            rationale="Test",
            confidence=0.9,
        )
        mock_orchestrator.run_single_agent.return_value = mock_result

        response = client.post(
            "/api/v1/agents/diagnosis",
            json={
                "inputs": {
                    "patient": {
                        "age": 50,
                        "sex": "male",
                        "chief_complaint": "Test",
                        "discipline": "PT",
                        "setting": "outpatient",
                    },
                    "subjective": "Pain reported",
                    "objective": "Limited ROM",
                },
                "discipline": "PT",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["agent"] == "diagnosis"
        assert "result" in data

    def test_run_invalid_agent(self, client, mock_orchestrator):
        """Test that invalid agent name returns error."""
        mock_orchestrator.run_single_agent.side_effect = ValueError("Unknown agent")

        response = client.post(
            "/api/v1/agents/invalid_agent",
            json={"inputs": {}},
        )

        assert response.status_code == 400

    def test_evidence_search_endpoint(self, client, test_app):
        """Test POST /api/v1/agents/evidence/search returns proper structure."""
        from rehab_os.models.evidence import EvidenceSummary
        from rehab_os.agents.evidence import EvidenceAgent

        # Mock the agent at the class level before importing in route
        mock_evidence_result = EvidenceSummary(
            query="rotator cuff",
            total_sources=5,
            synthesis="Conservative management recommended initially",
        )

        # Patch the agent class in the agents module
        with patch.object(EvidenceAgent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_evidence_result

            response = client.post(
                "/api/v1/agents/evidence/search",
                json={
                    "condition": "rotator cuff tear",
                    "clinical_question": "conservative vs surgical management",
                    "discipline": "PT",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "total_sources" in data


class TestHealthEndpointsFull:
    """Extended health endpoint tests."""

    def test_health_returns_service_info(self, client):
        """Test health endpoint returns service information."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "rehab-os"

    def test_readiness_with_empty_knowledge_base(self, client, test_app):
        """Test readiness check when knowledge base is empty."""
        test_app.state.vector_store.count = 0

        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_ready"
        assert "Vector store empty" in data["errors"]

    def test_readiness_with_llm_failure(self, client, test_app):
        """Test readiness check when LLM is unavailable."""
        test_app.state.llm_router.health_check = AsyncMock(
            return_value={"primary": False, "fallback": False}
        )

        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_ready"
        assert "No LLM available" in data["errors"]


class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_orchestrator_error_returns_500(self, client, mock_orchestrator):
        """Test that orchestrator errors return 500."""
        mock_orchestrator.process.side_effect = Exception("Internal error")

        response = client.post(
            "/api/v1/consult",
            json={"query": "Test query"},
        )

        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()

    def test_missing_required_field(self, client):
        """Test that missing required fields return 422."""
        response = client.post(
            "/api/v1/consult",
            json={},  # Missing required 'query' field
        )

        assert response.status_code == 422
