"""Tests for Orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rehab_os.agents.orchestrator import Orchestrator
from rehab_os.models.output import (
    ClinicalRequest,
    ConsultationResponse,
    SafetyAssessment,
    DiagnosisResult,
    UrgencyLevel,
)
from rehab_os.models.patient import Discipline, CareSetting


@pytest.fixture
def mock_orchestrator(mock_llm_router):
    """Create an orchestrator with mocked agents."""
    orchestrator = Orchestrator(llm=mock_llm_router, knowledge_base=None)
    return orchestrator


@pytest.fixture
def sample_request(sample_patient):
    """Create a sample clinical request."""
    return ClinicalRequest(
        query="Evaluate for PT after TKA",
        patient=sample_patient,
        discipline=Discipline.PT,
        setting=CareSetting.INPATIENT,
    )


@pytest.fixture
def mock_safe_assessment():
    """Create a safe assessment response."""
    return SafetyAssessment(
        is_safe_to_treat=True,
        urgency_level=UrgencyLevel.ROUTINE,
        summary="No red flags. Safe to proceed with PT.",
    )


@pytest.fixture
def mock_critical_assessment():
    """Create a critical assessment response."""
    from rehab_os.models.output import RedFlag

    return SafetyAssessment(
        is_safe_to_treat=False,
        urgency_level=UrgencyLevel.EMERGENT,
        red_flags=[
            RedFlag(
                finding="Cauda equina symptoms",
                description="Saddle anesthesia and bladder dysfunction",
                rationale="Classic cauda equina presentation",
                recommended_action="Immediate MRI and surgical consult",
                urgency=UrgencyLevel.EMERGENT,
            )
        ],
        summary="Critical red flags detected. Immediate medical evaluation required.",
        referral_recommended=True,
        referral_to="Emergency department",
    )


class TestOrchestrator:
    """Tests for Orchestrator."""

    def test_initialization(self, mock_orchestrator):
        """Test orchestrator initializes all agents."""
        assert mock_orchestrator.red_flag_agent is not None
        assert mock_orchestrator.diagnosis_agent is not None
        assert mock_orchestrator.evidence_agent is not None
        assert mock_orchestrator.plan_agent is not None
        assert mock_orchestrator.outcome_agent is not None
        assert mock_orchestrator.documentation_agent is not None
        assert mock_orchestrator.qa_agent is not None

    @pytest.mark.asyncio
    async def test_safety_check_runs_first(
        self, mock_orchestrator, sample_request, mock_safe_assessment
    ):
        """Test that safety check always runs first."""
        from rehab_os.models.evidence import EvidenceSummary
        from rehab_os.models.plan import PlanOfCare
        from rehab_os.models.output import OutcomeRecommendations

        mock_diagnosis = DiagnosisResult(
            primary_diagnosis="Test",
            icd_codes=["Z99.9"],
            rationale="Test",
            confidence=0.9,
        )
        mock_evidence = EvidenceSummary(query="test", total_sources=0)
        mock_plan = PlanOfCare(
            clinical_summary="Test",
            clinical_impression="Test",
            prognosis="Good",
            rehab_potential="Good",
            visit_frequency="2x/week",
            expected_duration="6 weeks",
        )
        mock_outcomes = OutcomeRecommendations(
            reassessment_schedule="Every 2 weeks",
            rationale="Standard protocol",
        )

        with patch.object(
            mock_orchestrator.red_flag_agent,
            "run",
            new_callable=AsyncMock,
            return_value=mock_safe_assessment,
        ) as mock_safety:
            with patch.object(
                mock_orchestrator.diagnosis_agent,
                "run",
                new_callable=AsyncMock,
                return_value=mock_diagnosis,
            ):
                with patch.object(
                    mock_orchestrator.evidence_agent,
                    "run",
                    new_callable=AsyncMock,
                    return_value=mock_evidence,
                ):
                    with patch.object(
                        mock_orchestrator.plan_agent,
                        "run",
                        new_callable=AsyncMock,
                        return_value=mock_plan,
                    ):
                        with patch.object(
                            mock_orchestrator.outcome_agent,
                            "run",
                            new_callable=AsyncMock,
                            return_value=mock_outcomes,
                        ):
                            sample_request.task_type = "plan_only"
                            await mock_orchestrator.process(sample_request, skip_qa=True)

            mock_safety.assert_called_once()

    @pytest.mark.asyncio
    async def test_critical_findings_stop_pipeline(
        self, mock_orchestrator, sample_request, mock_critical_assessment
    ):
        """Test that critical red flags stop further processing."""
        with patch.object(
            mock_orchestrator.red_flag_agent,
            "run",
            new_callable=AsyncMock,
            return_value=mock_critical_assessment,
        ):
            result = await mock_orchestrator.process(sample_request)

            assert result.safety.has_critical_findings
            assert result.plan is None  # Should not have generated a plan
            assert "CRITICAL" in result.processing_notes[0]

    @pytest.mark.asyncio
    async def test_safety_only_task(
        self, mock_orchestrator, sample_request, mock_safe_assessment
    ):
        """Test safety_only task type returns early."""
        sample_request.task_type = "safety_only"

        with patch.object(
            mock_orchestrator.red_flag_agent,
            "run",
            new_callable=AsyncMock,
            return_value=mock_safe_assessment,
        ):
            result = await mock_orchestrator.process(sample_request)

            assert result.safety is not None
            assert result.diagnosis is None
            assert result.plan is None

    @pytest.mark.asyncio
    async def test_run_single_agent(self, mock_orchestrator, sample_patient):
        """Test running a single agent directly."""
        mock_response = DiagnosisResult(
            primary_diagnosis="Test diagnosis",
            icd_codes=["Z99.9"],
            rationale="Test rationale",
            confidence=0.9,
        )

        with patch.object(
            mock_orchestrator.diagnosis_agent,
            "run",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await mock_orchestrator.run_single_agent(
                agent_name="diagnosis",
                inputs={
                    "patient": sample_patient.model_dump(),
                    "subjective": "Test",
                    "objective": "Test",
                },
            )

            assert isinstance(result, DiagnosisResult)

    @pytest.mark.asyncio
    async def test_invalid_agent_name(self, mock_orchestrator):
        """Test that invalid agent name raises error."""
        with pytest.raises(ValueError) as exc:
            await mock_orchestrator.run_single_agent(
                agent_name="invalid_agent",
                inputs={},
            )

        assert "Unknown agent" in str(exc.value)
