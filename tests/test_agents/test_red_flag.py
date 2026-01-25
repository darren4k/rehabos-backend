"""Tests for RedFlagAgent."""

import pytest
from unittest.mock import AsyncMock

from rehab_os.agents.red_flag import RedFlagAgent, RedFlagInput
from rehab_os.agents.base import AgentContext
from rehab_os.models.output import SafetyAssessment, UrgencyLevel
from rehab_os.models.patient import Discipline


@pytest.fixture
def red_flag_agent(mock_llm_router):
    """Create a RedFlagAgent with mock LLM."""
    return RedFlagAgent(llm=mock_llm_router)


@pytest.fixture
def mock_safety_response():
    """Create a mock safety assessment response."""
    return SafetyAssessment(
        is_safe_to_treat=True,
        urgency_level=UrgencyLevel.ROUTINE,
        summary="No red flags identified. Patient is safe for rehabilitation.",
        precautions=["Monitor vitals during activity"],
    )


class TestRedFlagAgent:
    """Tests for RedFlagAgent."""

    def test_agent_initialization(self, red_flag_agent):
        """Test agent initializes correctly."""
        assert red_flag_agent.name == "red_flag"
        assert red_flag_agent.description == "Safety screening and red flag identification"

    def test_system_prompt_contains_key_elements(self, red_flag_agent):
        """Test system prompt includes critical safety elements."""
        prompt = red_flag_agent.system_prompt
        assert "red flag" in prompt.lower()
        assert "safety" in prompt.lower()
        assert "emergent" in prompt.lower() or "urgent" in prompt.lower()

    def test_rule_based_screening_cauda_equina(self, red_flag_agent, sample_patient):
        """Test rule-based detection of cauda equina symptoms."""
        sample_patient.discipline = Discipline.PT
        input_data = RedFlagInput(
            patient=sample_patient,
            chief_complaint="Low back pain with saddle anesthesia and bladder dysfunction",
            subjective_report="Patient reports numbness in groin area and difficulty urinating",
        )

        findings = red_flag_agent.apply_rules(input_data)

        assert "cauda_equina" in findings
        assert "emergent" in findings["cauda_equina"].lower()

    def test_rule_based_screening_dvt(self, red_flag_agent, sample_patient):
        """Test rule-based detection of DVT symptoms."""
        sample_patient.discipline = Discipline.PT
        input_data = RedFlagInput(
            patient=sample_patient,
            chief_complaint="Calf pain and swelling",
            subjective_report="Patient had recent surgery, notes calf swelling and warmth",
        )

        findings = red_flag_agent.apply_rules(input_data)

        assert "dvt" in findings

    def test_rule_based_screening_vital_signs(self, red_flag_agent, sample_patient):
        """Test rule-based detection of abnormal vitals."""
        from rehab_os.models.patient import Vitals

        sample_patient.vitals = Vitals(
            oxygen_saturation=85,  # Hypoxia
            heart_rate=130,  # Tachycardia
        )

        input_data = RedFlagInput(
            patient=sample_patient,
            chief_complaint="General weakness",
        )

        findings = red_flag_agent.apply_rules(input_data)

        assert "hypoxia" in findings
        assert "abnormal_hr" in findings

    def test_rule_based_no_findings(self, red_flag_agent, sample_patient):
        """Test rule-based screening with no red flags."""
        input_data = RedFlagInput(
            patient=sample_patient,
            chief_complaint="Knee stiffness after surgery",
            subjective_report="Mild discomfort with ROM exercises",
        )

        findings = red_flag_agent.apply_rules(input_data)

        # Should have no critical findings
        assert not any("emergent" in str(v).lower() for v in findings.values())

    @pytest.mark.asyncio
    async def test_run_returns_safety_assessment(
        self, red_flag_agent, sample_patient, mock_safety_response, mock_llm_router
    ):
        """Test that run returns a SafetyAssessment."""
        mock_llm_router.complete_structured = AsyncMock(return_value=mock_safety_response)

        input_data = RedFlagInput(
            patient=sample_patient,
            chief_complaint="Knee pain after surgery",
        )

        result = await red_flag_agent.run(input_data)

        assert isinstance(result, SafetyAssessment)
        assert result.is_safe_to_treat is True

    def test_format_input_includes_all_sections(self, red_flag_agent, sample_patient):
        """Test that format_input includes all patient information."""
        from rehab_os.models.patient import Vitals

        sample_patient.vitals = Vitals(heart_rate=80, pain_level=4)

        input_data = RedFlagInput(
            patient=sample_patient,
            chief_complaint="Knee pain",
            subjective_report="Patient reports moderate pain",
            objective_findings="ROM limited, mild swelling",
        )

        context = AgentContext(discipline="PT")
        formatted = red_flag_agent.format_input(input_data, context)

        assert "Patient Information" in formatted
        assert str(sample_patient.age) in formatted
        assert "Diagnoses" in formatted
        assert "Comorbidities" in formatted
        assert "Medications" in formatted
        assert "Vital Signs" in formatted
        assert "Subjective Report" in formatted
        assert "Objective Findings" in formatted

    def test_temperature_is_low_for_safety(self, red_flag_agent):
        """Test that temperature is low for safety-critical reasoning."""
        assert red_flag_agent.temperature <= 0.3
