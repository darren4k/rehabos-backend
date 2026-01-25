"""Tests for DiagnosisAgent."""

import pytest
from unittest.mock import AsyncMock

from rehab_os.agents.diagnosis import DiagnosisAgent, DiagnosisInput
from rehab_os.agents.base import AgentContext
from rehab_os.models.output import DiagnosisResult


@pytest.fixture
def diagnosis_agent(mock_llm_router):
    """Create a DiagnosisAgent with mock LLM."""
    return DiagnosisAgent(llm=mock_llm_router)


@pytest.fixture
def mock_diagnosis_response():
    """Create a mock diagnosis response."""
    return DiagnosisResult(
        primary_diagnosis="Status post total knee arthroplasty, left",
        icd_codes=["Z96.651", "M17.11"],
        classification="Acute post-operative, POD 2",
        differential_diagnoses=["DVT", "Infection"],
        rationale="Patient is 2 days post-operative from left TKA with expected post-surgical presentation",
        key_findings=["Limited ROM", "Surgical site intact", "Mild edema"],
        confidence=0.85,
        uncertainties=["Monitor for signs of DVT"],
    )


class TestDiagnosisAgent:
    """Tests for DiagnosisAgent."""

    def test_agent_initialization(self, diagnosis_agent):
        """Test agent initializes correctly."""
        assert diagnosis_agent.name == "diagnosis"
        assert "classification" in diagnosis_agent.description.lower()

    def test_system_prompt_includes_icd(self, diagnosis_agent):
        """Test system prompt mentions ICD coding."""
        prompt = diagnosis_agent.system_prompt
        assert "ICD" in prompt or "icd" in prompt.lower()
        assert "diagnosis" in prompt.lower()

    @pytest.mark.asyncio
    async def test_run_returns_diagnosis_result(
        self, diagnosis_agent, sample_patient, mock_diagnosis_response, mock_llm_router
    ):
        """Test that run returns a DiagnosisResult."""
        mock_llm_router.complete_structured = AsyncMock(return_value=mock_diagnosis_response)

        input_data = DiagnosisInput(
            patient=sample_patient,
            subjective="Patient reports knee pain and stiffness",
            objective="ROM 0-90 degrees, mild edema, surgical site clean",
        )

        result = await diagnosis_agent.run(input_data)

        assert isinstance(result, DiagnosisResult)
        assert result.confidence > 0
        assert len(result.icd_codes) > 0

    def test_format_input_includes_exam_findings(self, diagnosis_agent, sample_patient):
        """Test format_input includes subjective and objective."""
        input_data = DiagnosisInput(
            patient=sample_patient,
            subjective="Knee pain, 4/10",
            objective="ROM 0-90, strength 3+/5",
            special_tests={"Lachman": "negative", "McMurray": "negative"},
        )

        context = AgentContext(discipline="PT")
        formatted = diagnosis_agent.format_input(input_data, context)

        assert "Subjective" in formatted
        assert "Objective" in formatted
        assert "Special Tests" in formatted
        assert "Lachman" in formatted

    def test_icd_code_reference_included(self, diagnosis_agent, sample_patient):
        """Test that ICD code reference is included for PT."""
        sample_patient.discipline = sample_patient.discipline  # Ensure PT

        input_data = DiagnosisInput(
            patient=sample_patient,
            subjective="Low back pain",
            objective="Limited flexion",
        )

        context = AgentContext(discipline="PT")
        formatted = diagnosis_agent.format_input(input_data, context)

        assert "ICD-10" in formatted
        assert "M54" in formatted  # Common LBP code prefix
