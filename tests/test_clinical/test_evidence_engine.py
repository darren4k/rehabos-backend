"""Tests for the evidence-based practice engine."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from rehab_os.clinical.evidence_engine import (
    EvidenceSuggestion,
    TreatmentPlanReview,
    suggest_evidence_based_treatments,
)


@pytest.fixture
def sample_review():
    return TreatmentPlanReview(
        suggested_interventions=[
            EvidenceSuggestion(
                intervention="LSVT BIG program",
                evidence_level="Level I - Systematic Review",
                rationale="Patient has Parkinson's with bradykinesia affecting gait",
                source_summary="Fox et al. systematic review supports amplitude-based training",
                relevance_score=0.95,
                category="intervention",
            )
        ],
        suggested_outcome_measures=[
            EvidenceSuggestion(
                intervention="Timed Up and Go (TUG)",
                evidence_level="Level I - Systematic Review",
                rationale="Gold standard for fall risk in Parkinson's",
                source_summary="MDC = 3.5 sec, MCID = 3.4 sec for PD population",
                relevance_score=0.9,
                category="outcome_measure",
            )
        ],
        suggested_goals=[
            EvidenceSuggestion(
                intervention="Patient will ambulate 300ft on level surfaces with SBA and RW in 4 weeks",
                evidence_level="Expert Consensus",
                rationale="Based on current 150ft distance, 100% improvement achievable per CPG",
                source_summary="APTA CPG recommends progressive gait distance goals",
                relevance_score=0.85,
                category="goal_suggestion",
            )
        ],
        missing_elements=["No fall risk assessment documented", "No caregiver training plan"],
        defensibility_notes=["Add skilled rationale for balance training interventions"],
    )


@pytest.fixture
def mock_llm(sample_review):
    llm = MagicMock()
    llm.complete_structured = AsyncMock(return_value=sample_review)
    llm.complete = AsyncMock()
    return llm


@pytest.mark.asyncio
async def test_suggest_evidence_based_treatments(mock_llm, sample_review):
    result = await suggest_evidence_based_treatments(
        diagnosis=["Parkinson's disease", "Gait dysfunction"],
        current_interventions=["Gait training", "Balance exercises"],
        functional_deficits=["Ambulation limited to 150ft with SBA"],
        patient_context={"age": 72, "setting": "outpatient", "comorbidities": ["HTN"]},
        note_type="evaluation",
        llm=mock_llm,
    )

    assert isinstance(result, TreatmentPlanReview)
    assert len(result.suggested_interventions) == 1
    assert result.suggested_interventions[0].intervention == "LSVT BIG program"
    assert result.suggested_interventions[0].relevance_score == 0.95
    assert len(result.suggested_outcome_measures) == 1
    assert len(result.suggested_goals) == 1
    assert "No fall risk assessment documented" in result.missing_elements
    mock_llm.complete_structured.assert_called_once()


@pytest.mark.asyncio
async def test_suggest_fallback_to_raw_completion(sample_review):
    """When structured fails, falls back to raw completion + JSON parse."""
    llm = MagicMock()
    llm.complete_structured = AsyncMock(side_effect=Exception("structured not supported"))
    llm.complete = AsyncMock()
    llm.complete.return_value = MagicMock(content=sample_review.model_dump_json())

    result = await suggest_evidence_based_treatments(
        diagnosis=["CVA"],
        current_interventions=[],
        functional_deficits=["R hemiparesis"],
        patient_context={"age": 65},
        note_type="evaluation",
        llm=llm,
    )

    assert isinstance(result, TreatmentPlanReview)
    assert len(result.suggested_interventions) == 1
    llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_suggest_handles_unparseable_response():
    """Returns safe fallback when both structured and raw parsing fail."""
    llm = MagicMock()
    llm.complete_structured = AsyncMock(side_effect=Exception("fail"))
    llm.complete = AsyncMock()
    llm.complete.return_value = MagicMock(content="not json at all")

    result = await suggest_evidence_based_treatments(
        diagnosis=["TBI"],
        current_interventions=[],
        functional_deficits=[],
        patient_context={},
        note_type="daily_note",
        llm=llm,
    )

    assert isinstance(result, TreatmentPlanReview)
    assert len(result.missing_elements) > 0


def test_evidence_suggestion_validation():
    """Test pydantic validation on EvidenceSuggestion."""
    s = EvidenceSuggestion(
        intervention="Test",
        evidence_level="Level II - RCT",
        rationale="Because",
        source_summary="A study",
        relevance_score=0.5,
        category="intervention",
    )
    assert s.relevance_score == 0.5

    with pytest.raises(Exception):
        EvidenceSuggestion(
            intervention="Test",
            evidence_level="Level II",
            rationale="x",
            source_summary="x",
            relevance_score=1.5,  # out of range
            category="intervention",
        )


def test_treatment_plan_review_defaults():
    """Empty review should have all empty lists."""
    r = TreatmentPlanReview()
    assert r.suggested_interventions == []
    assert r.suggested_outcome_measures == []
    assert r.suggested_goals == []
    assert r.missing_elements == []
    assert r.defensibility_notes == []
