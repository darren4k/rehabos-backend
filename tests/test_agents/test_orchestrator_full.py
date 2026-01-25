"""Full orchestrator integration tests with mocked LLM responses."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from rehab_os.agents.orchestrator import Orchestrator
from rehab_os.models.output import (
    ClinicalRequest,
    ConsultationResponse,
    SafetyAssessment,
    DiagnosisResult,
    OutcomeRecommendations,
    QAResult,
    RedFlag,
    UrgencyLevel,
)
from rehab_os.models.evidence import EvidenceSummary, Evidence, EvidenceLevel
from rehab_os.models.plan import PlanOfCare, SMARTGoal, GoalTimeframe, Intervention
from rehab_os.models.patient import PatientContext, Discipline, CareSetting


@pytest.fixture
def mock_knowledge_base():
    """Create a mock knowledge base."""
    kb = MagicMock()
    kb.search = AsyncMock(return_value=[
        Evidence(
            content="Exercise therapy is recommended for LBP",
            source="JOSPT Guidelines",
            evidence_level=EvidenceLevel.LEVEL_1A,
            relevance_score=0.95,
        )
    ])
    return kb


@pytest.fixture
def orchestrator_with_mocks(mock_llm_router, mock_knowledge_base):
    """Create orchestrator with all dependencies mocked."""
    return Orchestrator(llm=mock_llm_router, knowledge_base=mock_knowledge_base)


@pytest.fixture
def clean_patient():
    """Patient with no red flags."""
    return PatientContext(
        age=45,
        sex="female",
        chief_complaint="Chronic low back pain, 6 months duration",
        diagnosis=["Low back pain", "Lumbar degenerative disc disease"],
        comorbidities=["Hypertension (controlled)"],
        medications=["Lisinopril 10mg"],
        discipline=Discipline.PT,
        setting=CareSetting.OUTPATIENT,
        prior_level_of_function="Independent, works desk job",
        subjective_notes="Dull aching pain 4/10, worse with prolonged sitting",
        objective_findings="ROM limited in flexion, no radicular signs",
    )


@pytest.fixture
def red_flag_patient():
    """Patient with critical red flags."""
    return PatientContext(
        age=55,
        sex="male",
        chief_complaint="Severe low back pain with saddle anesthesia",
        diagnosis=["Low back pain"],
        comorbidities=[],
        medications=[],
        discipline=Discipline.PT,
        setting=CareSetting.OUTPATIENT,
        subjective_notes="Sudden onset, numbness in groin, difficulty urinating",
        objective_findings="Decreased sensation perineal area, weak ankle DF bilaterally",
    )


@pytest.fixture
def mock_clean_safety():
    """Safe assessment for clean patient."""
    return SafetyAssessment(
        is_safe_to_treat=True,
        urgency_level=UrgencyLevel.ROUTINE,
        summary="No red flags identified. Patient appropriate for outpatient PT.",
        precautions=["Monitor for radicular symptoms"],
    )


@pytest.fixture
def mock_critical_safety():
    """Critical assessment requiring immediate referral."""
    return SafetyAssessment(
        is_safe_to_treat=False,
        urgency_level=UrgencyLevel.EMERGENT,
        red_flags=[
            RedFlag(
                finding="Cauda equina syndrome",
                description="Saddle anesthesia and bladder dysfunction",
                rationale="Classic presentation of cauda equina compression",
                recommended_action="Immediate ED referral for emergent MRI",
                urgency=UrgencyLevel.EMERGENT,
            )
        ],
        summary="CRITICAL: Cauda equina syndrome suspected. Immediate surgical evaluation required.",
        referral_recommended=True,
        referral_to="Emergency Department",
        referral_rationale="Emergent MRI and neurosurgical consultation needed",
    )


@pytest.fixture
def mock_diagnosis():
    """Standard diagnosis result."""
    return DiagnosisResult(
        primary_diagnosis="Chronic low back pain with lumbar degenerative disc disease",
        icd_codes=["M54.5", "M51.36"],
        classification="Chronic, mechanical, non-specific",
        differential_diagnoses=["Facet arthropathy", "SI joint dysfunction"],
        rationale="History and exam consistent with chronic mechanical LBP without radiculopathy",
        key_findings=["Limited flexion ROM", "No neurological deficits", "Pain with sustained postures"],
        confidence=0.85,
        uncertainties=["Consider imaging if no improvement in 4-6 weeks"],
    )


@pytest.fixture
def mock_evidence():
    """Evidence summary with guidelines."""
    return EvidenceSummary(
        query="chronic low back pain physical therapy",
        total_sources=5,
        evidence_items=[
            Evidence(
                content="Structured exercise programs are strongly recommended for chronic LBP",
                source="JOSPT Clinical Practice Guidelines",
                evidence_level=EvidenceLevel.LEVEL_1A,
                relevance_score=0.95,
            ),
            Evidence(
                content="Manual therapy combined with exercise shows moderate benefit",
                source="Cochrane Review",
                evidence_level=EvidenceLevel.LEVEL_1A,
                relevance_score=0.88,
            ),
        ],
        synthesis="Strong evidence supports exercise therapy and manual therapy for chronic LBP",
        confidence=0.9,
    )


@pytest.fixture
def mock_plan():
    """Complete plan of care."""
    return PlanOfCare(
        clinical_summary="45yo female with chronic mechanical LBP, appropriate for conservative PT",
        clinical_impression="Movement system impairment with mobility deficit",
        prognosis="Good - expected to achieve functional goals within 6-8 weeks",
        rehab_potential="Good - motivated patient, no significant barriers",
        smart_goals=[
            SMARTGoal(
                description="Reduce pain from 4/10 to 2/10 during prolonged sitting",
                timeframe=GoalTimeframe.SHORT_TERM,
                specific="Decrease sitting-related pain",
                measurable="NPRS rating",
                achievable="Typical response to intervention",
                relevant="Patient's primary functional limitation",
                time_bound="4 weeks",
            ),
            SMARTGoal(
                description="Return to 8-hour work day without pain limiting function",
                timeframe=GoalTimeframe.LONG_TERM,
                specific="Full work tolerance",
                measurable="Hours worked without modification",
                achievable="Good prognosis",
                relevant="Patient's primary goal",
                time_bound="8 weeks",
            ),
        ],
        interventions=[
            Intervention(
                name="Lumbar stabilization exercises",
                category="therapeutic_exercise",
                description="Core strengthening with neutral spine",
                rationale="Address motor control deficits",
                cpt_codes=["97110"],
            ),
            Intervention(
                name="Manual therapy - lumbar mobilization",
                category="manual_therapy",
                description="Grade III-IV PA mobilization L3-S1",
                rationale="Address joint mobility restrictions",
                cpt_codes=["97140"],
            ),
        ],
        visit_frequency="2x/week for 4 weeks, then 1x/week for 4 weeks",
        expected_duration="8 weeks",
        discharge_criteria=["Pain < 2/10", "Full work tolerance", "Independent HEP"],
    )


@pytest.fixture
def mock_outcomes():
    """Outcome measure recommendations."""
    from rehab_os.models.output import OutcomeMeasure

    return OutcomeRecommendations(
        primary_measures=[
            OutcomeMeasure(
                name="Oswestry Disability Index",
                abbreviation="ODI",
                description="Self-report measure of LBP-related disability",
                rationale="Gold standard for LBP functional assessment",
                frequency="Initial, 4 weeks, discharge",
                mcid="10%",
                mdc="10%",
            ),
        ],
        secondary_measures=[
            OutcomeMeasure(
                name="Patient-Specific Functional Scale",
                abbreviation="PSFS",
                description="Patient-identified functional activities",
                rationale="Captures individual patient priorities",
                frequency="Each visit",
                mcid="2 points",
                mdc="2 points",
            ),
        ],
        reassessment_schedule="Every 4 weeks",
        rationale="Standard protocol for chronic LBP management",
    )


@pytest.fixture
def mock_qa_good():
    """Good QA review."""
    return QAResult(
        overall_quality=0.88,
        strengths=[
            "Evidence-based intervention selection",
            "Clear, measurable goals",
            "Appropriate outcome measures",
        ],
        suggestions=[
            "Consider adding patient education component",
            "May benefit from ergonomic assessment",
        ],
        concerns=[],
        uncertainty_flags=[],
        evidence_gaps=[],
        alternative_approaches=["Could consider McKenzie approach if centralization noted"],
    )


@pytest.fixture
def mock_qa_with_concerns():
    """QA review with uncertainty flags."""
    return QAResult(
        overall_quality=0.65,
        strengths=["Appropriate safety screening"],
        suggestions=["Clarify exercise progression criteria"],
        concerns=["Goals may be overly ambitious for timeline"],
        uncertainty_flags=[
            "Insufficient evidence for specific manual therapy technique",
            "Patient's work demands not fully characterized",
        ],
        evidence_gaps=["Limited evidence for this specific population (sedentary workers)"],
        alternative_approaches=["Consider multidisciplinary approach if plateau"],
    )


class TestOrchestratorFullPipeline:
    """Integration tests for full orchestrator pipeline."""

    @pytest.mark.asyncio
    async def test_clean_case_full_pipeline(
        self,
        orchestrator_with_mocks,
        clean_patient,
        mock_clean_safety,
        mock_diagnosis,
        mock_evidence,
        mock_plan,
        mock_outcomes,
        mock_qa_good,
    ):
        """Test complete pipeline for clean case: Safety → Diagnosis → Evidence → Plan → QA."""
        request = ClinicalRequest(
            query="Evaluate and treat chronic low back pain",
            patient=clean_patient,
            discipline=Discipline.PT,
            setting=CareSetting.OUTPATIENT,
        )

        with patch.object(
            orchestrator_with_mocks.red_flag_agent, "run",
            new_callable=AsyncMock, return_value=mock_clean_safety
        ):
            with patch.object(
                orchestrator_with_mocks.diagnosis_agent, "run",
                new_callable=AsyncMock, return_value=mock_diagnosis
            ):
                with patch.object(
                    orchestrator_with_mocks.evidence_agent, "run",
                    new_callable=AsyncMock, return_value=mock_evidence
                ):
                    with patch.object(
                        orchestrator_with_mocks.plan_agent, "run",
                        new_callable=AsyncMock, return_value=mock_plan
                    ):
                        with patch.object(
                            orchestrator_with_mocks.outcome_agent, "run",
                            new_callable=AsyncMock, return_value=mock_outcomes
                        ):
                            with patch.object(
                                orchestrator_with_mocks.qa_agent, "run",
                                new_callable=AsyncMock, return_value=mock_qa_good
                            ):
                                result = await orchestrator_with_mocks.process(request)

        # Verify complete response
        assert isinstance(result, ConsultationResponse)
        assert result.safety.is_safe_to_treat is True
        assert result.diagnosis is not None
        assert result.diagnosis.confidence >= 0.8
        assert result.evidence is not None
        assert result.evidence.total_sources > 0
        assert result.plan is not None
        assert len(result.plan.smart_goals) >= 2
        assert result.outcomes is not None
        assert result.qa_review is not None
        assert result.qa_review.overall_quality >= 0.8

    @pytest.mark.asyncio
    async def test_red_flag_aborts_pipeline(
        self,
        orchestrator_with_mocks,
        red_flag_patient,
        mock_critical_safety,
    ):
        """Test that critical red flags abort the pipeline immediately."""
        request = ClinicalRequest(
            query="Evaluate low back pain",
            patient=red_flag_patient,
            discipline=Discipline.PT,
            setting=CareSetting.OUTPATIENT,
        )

        with patch.object(
            orchestrator_with_mocks.red_flag_agent, "run",
            new_callable=AsyncMock, return_value=mock_critical_safety
        ):
            result = await orchestrator_with_mocks.process(request)

        # Verify emergency response
        assert result.safety.is_safe_to_treat is False
        assert result.safety.has_critical_findings is True
        assert result.safety.referral_recommended is True
        assert "Emergency" in result.safety.referral_to

        # Verify pipeline was aborted
        assert result.diagnosis is None
        assert result.plan is None
        assert "CRITICAL" in result.processing_notes[0]

    @pytest.mark.asyncio
    async def test_conflicting_evidence_qa_flags_uncertainty(
        self,
        orchestrator_with_mocks,
        clean_patient,
        mock_clean_safety,
        mock_diagnosis,
        mock_plan,
        mock_outcomes,
        mock_qa_with_concerns,
    ):
        """Test that conflicting/limited evidence results in QA uncertainty flags."""
        # Evidence with low confidence
        limited_evidence = EvidenceSummary(
            query="specific treatment query",
            total_sources=1,
            evidence_items=[
                Evidence(
                    content="Limited evidence available",
                    source="Single study",
                    evidence_level=EvidenceLevel.LEVEL_4,
                    relevance_score=0.5,
                ),
            ],
            synthesis="Insufficient evidence to make strong recommendations",
            confidence=0.4,
            limitations="Only case series available, no RCTs",
        )

        request = ClinicalRequest(
            query="Evaluate patient",
            patient=clean_patient,
            discipline=Discipline.PT,
            setting=CareSetting.OUTPATIENT,
        )

        with patch.object(
            orchestrator_with_mocks.red_flag_agent, "run",
            new_callable=AsyncMock, return_value=mock_clean_safety
        ):
            with patch.object(
                orchestrator_with_mocks.diagnosis_agent, "run",
                new_callable=AsyncMock, return_value=mock_diagnosis
            ):
                with patch.object(
                    orchestrator_with_mocks.evidence_agent, "run",
                    new_callable=AsyncMock, return_value=limited_evidence
                ):
                    with patch.object(
                        orchestrator_with_mocks.plan_agent, "run",
                        new_callable=AsyncMock, return_value=mock_plan
                    ):
                        with patch.object(
                            orchestrator_with_mocks.outcome_agent, "run",
                            new_callable=AsyncMock, return_value=mock_outcomes
                        ):
                            with patch.object(
                                orchestrator_with_mocks.qa_agent, "run",
                                new_callable=AsyncMock, return_value=mock_qa_with_concerns
                            ):
                                result = await orchestrator_with_mocks.process(request)

        # Verify QA flagged uncertainties
        assert result.qa_review is not None
        assert len(result.qa_review.uncertainty_flags) > 0
        assert len(result.qa_review.evidence_gaps) > 0
        assert result.qa_review.overall_quality < 0.8

    @pytest.mark.asyncio
    async def test_diagnosis_only_task_type(
        self,
        orchestrator_with_mocks,
        clean_patient,
        mock_clean_safety,
        mock_diagnosis,
        mock_evidence,
    ):
        """Test diagnosis_only task type returns early."""
        request = ClinicalRequest(
            query="Diagnose condition",
            patient=clean_patient,
            discipline=Discipline.PT,
            setting=CareSetting.OUTPATIENT,
            task_type="diagnosis_only",
        )

        with patch.object(
            orchestrator_with_mocks.red_flag_agent, "run",
            new_callable=AsyncMock, return_value=mock_clean_safety
        ):
            with patch.object(
                orchestrator_with_mocks.diagnosis_agent, "run",
                new_callable=AsyncMock, return_value=mock_diagnosis
            ):
                with patch.object(
                    orchestrator_with_mocks.evidence_agent, "run",
                    new_callable=AsyncMock, return_value=mock_evidence
                ):
                    result = await orchestrator_with_mocks.process(request)

        assert result.safety is not None
        assert result.diagnosis is not None
        assert result.evidence is not None
        assert result.plan is None  # Should not generate plan
        assert result.qa_review is None

    @pytest.mark.asyncio
    async def test_skip_qa_flag(
        self,
        orchestrator_with_mocks,
        clean_patient,
        mock_clean_safety,
        mock_diagnosis,
        mock_evidence,
        mock_plan,
        mock_outcomes,
    ):
        """Test skip_qa flag bypasses QA review."""
        request = ClinicalRequest(
            query="Quick consult",
            patient=clean_patient,
            discipline=Discipline.PT,
            setting=CareSetting.OUTPATIENT,
            task_type="plan_only",
        )

        with patch.object(
            orchestrator_with_mocks.red_flag_agent, "run",
            new_callable=AsyncMock, return_value=mock_clean_safety
        ):
            with patch.object(
                orchestrator_with_mocks.diagnosis_agent, "run",
                new_callable=AsyncMock, return_value=mock_diagnosis
            ):
                with patch.object(
                    orchestrator_with_mocks.evidence_agent, "run",
                    new_callable=AsyncMock, return_value=mock_evidence
                ):
                    with patch.object(
                        orchestrator_with_mocks.plan_agent, "run",
                        new_callable=AsyncMock, return_value=mock_plan
                    ):
                        with patch.object(
                            orchestrator_with_mocks.outcome_agent, "run",
                            new_callable=AsyncMock, return_value=mock_outcomes
                        ):
                            result = await orchestrator_with_mocks.process(
                                request, skip_qa=True
                            )

        assert result.plan is not None
        assert result.qa_review is None  # Should be skipped

    @pytest.mark.asyncio
    async def test_no_patient_context_minimal_response(
        self, orchestrator_with_mocks
    ):
        """Test handling request without patient context."""
        request = ClinicalRequest(
            query="General question about LBP treatment",
            patient=None,
            discipline=Discipline.PT,
            setting=CareSetting.OUTPATIENT,
            task_type="safety_only",
        )

        result = await orchestrator_with_mocks.process(request)

        # Should return minimal safety response
        assert result.safety is not None
        assert result.safety.is_safe_to_treat is True
        assert "No patient context" in result.safety.summary
