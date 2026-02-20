"""Pytest configuration and fixtures."""

import uuid

import pytest
from unittest.mock import AsyncMock, MagicMock

from rehab_os.llm import LLMRouter, LLMResponse, Message, MessageRole
from rehab_os.models.patient import PatientContext, Discipline, CareSetting


# ---------------------------------------------------------------------------
# Auth override for tests — bypass get_current_user dependency
# ---------------------------------------------------------------------------

class _MockProvider:
    """Lightweight stand-in for Provider ORM model used in tests."""

    def __init__(self):
        self.id = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
        self.first_name = "Test"
        self.last_name = "Therapist"
        self.discipline = "PT"
        self.role = "owner"
        self.active = True
        self.organization_id = None
        self.credentials = "DPT"
        self.npi = None
        self.email = "test@example.com"
        self.password_hash = None
        self.must_change_password = False


def _fake_current_user():
    return _MockProvider()


@pytest.fixture
def mock_current_user():
    """Return a mock provider for auth bypass."""
    return _MockProvider()


def apply_auth_override(app):
    """Apply get_current_user override to a FastAPI test app."""
    from rehab_os.api.dependencies import get_current_user
    app.dependency_overrides[get_current_user] = _fake_current_user
    return app


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return LLMResponse(
        content='{"test": "response"}',
        model="test-model",
        usage={"input_tokens": 100, "output_tokens": 50},
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns structured responses."""
    llm = MagicMock()
    llm.complete = AsyncMock()
    llm.complete_structured = AsyncMock()
    llm.health_check = AsyncMock(return_value=True)
    llm.model_name = "mock-model"
    llm.provider = "mock"
    return llm


@pytest.fixture
def mock_llm_router(mock_llm):
    """Create a mock LLM router."""
    router = MagicMock(spec=LLMRouter)
    router.complete = mock_llm.complete
    router.complete_structured = mock_llm.complete_structured
    router.health_check = AsyncMock(return_value={"primary": True})
    return router


@pytest.fixture
def sample_patient():
    """Create a sample patient context."""
    return PatientContext(
        age=68,
        sex="male",
        chief_complaint="Left knee pain s/p TKA POD 2",
        diagnosis=["Total knee arthroplasty", "Osteoarthritis"],
        comorbidities=["Hypertension", "Type 2 diabetes"],
        medications=["Lisinopril", "Metformin", "Aspirin"],
        precautions=["WBAT L LE", "CPM as ordered"],
        discipline=Discipline.PT,
        setting=CareSetting.INPATIENT,
        prior_level_of_function="Independent with all ADLs and IADLs, walked 1 mile daily",
    )


@pytest.fixture
def sample_patient_slp():
    """Create a sample SLP patient context."""
    return PatientContext(
        age=72,
        sex="female",
        chief_complaint="Dysphagia following stroke",
        diagnosis=["Left MCA CVA", "Dysphagia"],
        comorbidities=["Atrial fibrillation", "Hypertension"],
        medications=["Warfarin", "Metoprolol"],
        discipline=Discipline.SLP,
        setting=CareSetting.ACUTE,
    )


@pytest.fixture
def sample_messages():
    """Create sample messages for LLM calls."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are a clinical assistant."),
        Message(role=MessageRole.USER, content="Assess this patient."),
    ]
