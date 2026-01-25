"""Tests for health endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_app():
    """Create a test application with mocked dependencies."""
    from fastapi import FastAPI
    from rehab_os.api.routes import health

    app = FastAPI()
    app.include_router(health.router)

    # Mock app state
    app.state.llm_router = MagicMock()
    app.state.llm_router.health_check = AsyncMock(
        return_value={"primary": True, "fallback": True}
    )

    app.state.vector_store = MagicMock()
    app.state.vector_store.count = 100

    return app


@pytest.fixture
def client(mock_app):
    """Create a test client."""
    return TestClient(mock_app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["service"] == "rehab-os"

    def test_liveness_check(self, client):
        """Test liveness check."""
        response = client.get("/health/live")

        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_check_success(self, client):
        """Test readiness check when all systems are healthy."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert "llm" in response.json()
        assert response.json()["knowledge_base_docs"] == 100
