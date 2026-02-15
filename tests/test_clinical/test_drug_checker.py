"""Tests for clinical intelligence — drug checker & chronic management."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from rehab_os.clinical.drug_checker import (
    DrugCheckResult,
    DrugInteraction,
    SideEffectCorrelation,
    check_drug_interactions,
    correlate_symptoms,
    _extract_json,
    _parse_drug_check_json,
)
from rehab_os.clinical.chronic_management import (
    ClinicalAlert,
    store_clinical_snapshot,
    check_for_alerts,
    get_patient_snapshots,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_DRUG_CHECK = DrugCheckResult(
    interactions=[
        DrugInteraction(
            drug_a="Warfarin",
            drug_b="Aspirin",
            severity="major",
            description="Increased bleeding risk",
            clinical_significance="Concurrent use significantly increases hemorrhagic risk",
            recommendation="Monitor INR closely; avoid concurrent use if possible",
        )
    ],
    side_effect_correlations=[
        SideEffectCorrelation(
            medication="Metoprolol",
            symptom="fatigue",
            likelihood="common",
            description="Beta-blockers commonly cause fatigue and exercise intolerance",
            recommendation="monitor",
        )
    ],
    therapy_considerations=[
        "Metoprolol may limit HR response during exercise — monitor RPE instead of target HR"
    ],
    alerts=["Patient on Warfarin + Aspirin — high bleeding risk during manual therapy"],
    fall_risk_medications=["Metoprolol", "Lorazepam"],
)

MOCK_CORRELATIONS = [
    {
        "medication": "Lisinopril",
        "symptom": "dizziness",
        "likelihood": "common",
        "description": "ACE inhibitors can cause orthostatic hypotension",
        "recommendation": "monitor",
    }
]


def _make_llm_mock(structured_return=None, complete_content="[]"):
    """Create a mock LLMRouter."""
    llm = AsyncMock()
    if structured_return is not None:
        llm.complete_structured = AsyncMock(return_value=structured_return)
    else:
        llm.complete_structured = AsyncMock(side_effect=Exception("no structured"))
    resp = MagicMock()
    resp.content = complete_content
    llm.complete = AsyncMock(return_value=resp)
    return llm


def _make_memory_mock():
    """Create a mock SessionMemoryService."""
    mem = MagicMock()
    mem.is_memu_available = False
    mem._cache = {}
    return mem


# ---------------------------------------------------------------------------
# Drug checker tests
# ---------------------------------------------------------------------------

class TestDrugChecker:

    @pytest.mark.asyncio
    async def test_check_drug_interactions_structured(self):
        llm = _make_llm_mock(structured_return=MOCK_DRUG_CHECK)
        result = await check_drug_interactions(["Warfarin", "Aspirin", "Metoprolol"], llm)

        assert len(result.interactions) == 1
        assert result.interactions[0].severity == "major"
        assert "Metoprolol" in result.fall_risk_medications
        assert len(result.therapy_considerations) == 1
        llm.complete_structured.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_drug_interactions_fallback(self):
        """When structured call fails, falls back to unstructured JSON parsing."""
        raw_json = MOCK_DRUG_CHECK.model_dump_json()
        llm = _make_llm_mock(structured_return=None, complete_content=raw_json)
        result = await check_drug_interactions(["Warfarin", "Aspirin"], llm)

        assert len(result.interactions) == 1
        llm.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_empty_medications(self):
        llm = _make_llm_mock()
        result = await check_drug_interactions([], llm)
        assert result == DrugCheckResult()
        llm.complete_structured.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_correlate_symptoms(self):
        llm = _make_llm_mock(complete_content=json.dumps(MOCK_CORRELATIONS))
        result = await correlate_symptoms(
            ["Lisinopril"], ["dizziness"], ["Hypertension"], llm
        )
        assert len(result) == 1
        assert result[0].medication == "Lisinopril"
        assert result[0].likelihood == "common"

    @pytest.mark.asyncio
    async def test_correlate_empty(self):
        llm = _make_llm_mock()
        result = await correlate_symptoms([], ["dizziness"], ["HTN"], llm)
        assert result == []

    def test_extract_json_plain(self):
        assert _extract_json('{"a":1}') == '{"a":1}'

    def test_extract_json_fenced(self):
        text = "```json\n{\"a\":1}\n```"
        assert _extract_json(text) == '{"a":1}'

    def test_parse_drug_check_json_valid(self):
        raw = MOCK_DRUG_CHECK.model_dump_json()
        result = _parse_drug_check_json(raw)
        assert len(result.interactions) == 1

    def test_parse_drug_check_json_invalid(self):
        result = _parse_drug_check_json("not json at all")
        assert len(result.alerts) == 1
        assert "parsing error" in result.alerts[0].lower()


# ---------------------------------------------------------------------------
# Chronic management tests
# ---------------------------------------------------------------------------

class TestChronicManagement:

    @pytest.mark.asyncio
    async def test_store_and_retrieve_snapshot(self):
        mem = _make_memory_mock()
        await store_clinical_snapshot(
            patient_id="P001",
            medications=["Metformin"],
            symptoms=["fatigue"],
            vitals={"bp": "130/80"},
            functional_status={"berg_balance": 48},
            memory_service=mem,
        )
        snapshots = await get_patient_snapshots("P001", mem)
        assert len(snapshots) == 1
        assert snapshots[0]["data"]["medications"] == ["Metformin"]

    @pytest.mark.asyncio
    async def test_check_for_alerts_no_history(self):
        mem = _make_memory_mock()
        llm = _make_llm_mock(complete_content="[]")
        alerts = await check_for_alerts("P001", {}, mem, llm)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_check_for_alerts_with_data(self):
        mem = _make_memory_mock()
        # Pre-populate history
        await store_clinical_snapshot("P002", ["Warfarin"], ["bruising"], {"bp": "140/90"}, {}, mem)

        alert_json = json.dumps([{
            "alert_type": "vital_trend",
            "severity": "warning",
            "description": "BP trending upward",
            "recommendation": "Notify physician",
            "related_data": {"bp_history": ["130/85", "140/90"]},
        }])
        llm = _make_llm_mock(complete_content=alert_json)

        alerts = await check_for_alerts(
            "P002",
            {"medications": ["Warfarin"], "vitals": {"bp": "150/95"}},
            mem,
            llm,
        )
        assert len(alerts) == 1
        assert alerts[0].alert_type == "vital_trend"
        assert alerts[0].severity == "warning"
