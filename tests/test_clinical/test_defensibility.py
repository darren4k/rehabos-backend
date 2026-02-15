"""Tests for the Medicare & Joint Commission defensibility checker."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from rehab_os.clinical.defensibility import (
    DefensibilityCheck,
    MedicareRequirement,
    check_defensibility,
)


@pytest.fixture
def good_check():
    return DefensibilityCheck(
        overall_score=88.0,
        category_scores={
            "medical_necessity": 95.0,
            "skilled_care_justification": 90.0,
            "functional_outcomes": 85.0,
            "progress_documentation": 80.0,
            "plan_of_care": 90.0,
            "safety_documentation": 85.0,
            "discharge_planning": 75.0,
        },
        passed=[
            "Skilled care justification present",
            "Medical necessity documented",
            "Functional goals present",
        ],
        warnings=["Discharge plan could be more specific"],
        failures=[],
        recommendations=["Add specific discharge criteria with measurable benchmarks"],
        medicare_requirements=[
            MedicareRequirement(
                requirement="Skilled care justification",
                met=True,
                evidence_in_note="Patient requires skilled neuromuscular re-education for gait deviations",
            ),
            MedicareRequirement(
                requirement="Medical necessity",
                met=True,
                evidence_in_note="Gait dysfunction limits community ambulation and fall risk is elevated",
            ),
        ],
    )


@pytest.fixture
def bad_check():
    return DefensibilityCheck(
        overall_score=35.0,
        category_scores={
            "medical_necessity": 40.0,
            "skilled_care_justification": 20.0,
            "functional_outcomes": 30.0,
        },
        passed=[],
        warnings=["Goals are impairment-level only"],
        failures=[
            "No skilled care justification",
            "Uses 'maintenance therapy' language",
            "No functional goals documented",
        ],
        recommendations=[
            "Add skilled rationale for each intervention",
            "Replace 'maintenance therapy' with 'skilled maintenance' and provide justification",
            "Convert impairment goals to functional goals",
        ],
        medicare_requirements=[
            MedicareRequirement(
                requirement="Skilled care justification",
                met=False,
                fix_suggestion="Document why a licensed therapist is required for these interventions",
            ),
        ],
    )


@pytest.fixture
def mock_llm(good_check):
    llm = MagicMock()
    llm.complete_structured = AsyncMock(return_value=good_check)
    llm.complete = AsyncMock()
    return llm


SAMPLE_NOTE = {
    "subjective": "Patient reports improved balance, less fear of falling. Pain 3/10 lumbar.",
    "objective": "Gait training 200ft with SBA and RW. TUG 15.2 sec. Berg 44/56. NMR for gait deviations. Therapeutic exercise for LE strengthening.",
    "assessment": "Patient demonstrates measurable improvement in gait endurance and balance. Continued skilled PT warranted for fall risk reduction and community ambulation goals.",
    "plan": "Continue PT 2x/week x 4 weeks. Progress gait distance to 500ft. Advance to CGA. Caregiver training for home safety.",
}


@pytest.mark.asyncio
async def test_check_defensibility_good_note(mock_llm, good_check):
    result = await check_defensibility(
        note_content=SAMPLE_NOTE,
        note_type="progress_note",
        structured_data={"tug": 15.2, "berg": 44},
        patient_context={"age": 72, "setting": "outpatient"},
        llm=mock_llm,
    )

    assert isinstance(result, DefensibilityCheck)
    assert result.overall_score == 88.0
    assert len(result.passed) == 3
    assert len(result.failures) == 0
    assert result.category_scores["medical_necessity"] == 95.0
    mock_llm.complete_structured.assert_called_once()


@pytest.mark.asyncio
async def test_check_defensibility_bad_note(bad_check):
    llm = MagicMock()
    llm.complete_structured = AsyncMock(return_value=bad_check)

    result = await check_defensibility(
        note_content={"subjective": "Patient doing well", "objective": "Exercises performed", "assessment": "Continue", "plan": "Same"},
        note_type="daily_note",
        structured_data={},
        patient_context={},
        llm=llm,
    )

    assert result.overall_score < 60
    assert len(result.failures) == 3
    assert any("maintenance therapy" in f for f in result.failures)


@pytest.mark.asyncio
async def test_check_defensibility_fallback_to_raw(good_check):
    llm = MagicMock()
    llm.complete_structured = AsyncMock(side_effect=Exception("structured fail"))
    llm.complete = AsyncMock()
    llm.complete.return_value = MagicMock(content=good_check.model_dump_json())

    result = await check_defensibility(
        note_content=SAMPLE_NOTE,
        note_type="progress_note",
        structured_data={},
        patient_context={},
        llm=llm,
    )

    assert isinstance(result, DefensibilityCheck)
    assert result.overall_score == 88.0


@pytest.mark.asyncio
async def test_check_defensibility_unparseable():
    llm = MagicMock()
    llm.complete_structured = AsyncMock(side_effect=Exception("fail"))
    llm.complete = AsyncMock()
    llm.complete.return_value = MagicMock(content="garbage")

    result = await check_defensibility(
        note_content=SAMPLE_NOTE,
        note_type="daily_note",
        structured_data={},
        patient_context={},
        llm=llm,
    )

    assert isinstance(result, DefensibilityCheck)
    assert result.overall_score == 0.0
    assert len(result.failures) > 0


def test_medicare_requirement_model():
    req = MedicareRequirement(
        requirement="Skilled care justification",
        met=True,
        evidence_in_note="Skilled NMR required for gait deviations",
    )
    assert req.met is True
    assert req.fix_suggestion is None

    req2 = MedicareRequirement(
        requirement="Discharge plan",
        met=False,
        fix_suggestion="Add discharge criteria",
    )
    assert req2.met is False
    assert req2.evidence_in_note is None


def test_defensibility_check_defaults():
    d = DefensibilityCheck(overall_score=50.0)
    assert d.passed == []
    assert d.warnings == []
    assert d.failures == []
    assert d.recommendations == []
    assert d.medicare_requirements == []


def test_denial_pattern_maintenance_therapy():
    """Verify the model can represent maintenance therapy denial flags."""
    check = DefensibilityCheck(
        overall_score=30.0,
        failures=["Note uses 'maintenance therapy' language which triggers automatic Medicare denial"],
        recommendations=["Replace with 'skilled maintenance program' and document: (1) complexity requiring therapist, (2) risk of decline without skilled oversight, (3) specific skilled techniques used"],
    )
    assert "maintenance therapy" in check.failures[0]
    assert len(check.recommendations) == 1
