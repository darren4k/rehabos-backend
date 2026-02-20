"""Tests for comprehensive clinical data models in note generation."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.conftest import apply_auth_override
from rehab_os.api.routes.notes import (
    router, _generate_from_transcript, GeneratedNote, ExtractedClinicalData,
)
from rehab_os.models.clinical import (
    ROMEntry, MMTEntry, StandardizedTest, FunctionalDeficit,
    Vitals, BalanceAssessment, GoalWithBaseline,
)


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    apply_auth_override(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def _full_evaluation_payload():
    """Build a payload matching the richness of the DocPilot sample data."""
    return {
        "note_type": "evaluation",
        "discipline": "pt",
        "age": 72,
        "diagnosis": ["R26.2 - Difficulty in walking", "G20 - Parkinson's disease"],
        "precautions": ["Fall risk (high)", "Cardiac precautions", "Orthostatic hypotension"],
        "setting": "Home - 12",
        "session_date": "2025-11-29",
        "session_duration_minutes": 90,
        "rom": [
            {"joint": "cervical", "motion": "general", "qualitative": "WFL with mild rigidity", "side": "bilateral"},
            {"joint": "trunk", "motion": "rotation", "qualitative": "Decreased bilaterally, mild rigidity", "side": "bilateral"},
            {"joint": "bilateral_hips", "motion": "general", "qualitative": "WFL", "side": "bilateral"},
            {"joint": "bilateral_knees", "motion": "general", "qualitative": "WFL", "side": "bilateral"},
            {"joint": "bilateral_ankles", "motion": "general", "qualitative": "WFL, mild tightness in gastrocnemius", "side": "bilateral"},
        ],
        "mmt": [
            {"muscle_group": "hip_flexion", "grade": "4/5", "side": "right"},
            {"muscle_group": "hip_flexion", "grade": "4/5", "side": "left"},
            {"muscle_group": "hip_extension", "grade": "3+/5", "side": "right"},
            {"muscle_group": "hip_extension", "grade": "3+/5", "side": "left"},
            {"muscle_group": "knee_extension", "grade": "4/5", "side": "bilateral"},
            {"muscle_group": "trunk_flexion", "grade": "3+/5", "side": "bilateral"},
        ],
        "standardized_tests": [
            {"name": "TUG", "score": 18.5, "unit": "seconds", "interpretation": "High fall risk (>14 sec)"},
            {"name": "Berg Balance Scale", "score": 42, "max_score": 56, "interpretation": "Medium fall risk"},
            {"name": "Tinetti", "score": 20, "max_score": 28, "interpretation": "High fall risk"},
            {"name": "5xSTS", "score": 16.2, "unit": "seconds", "interpretation": "Abnormal (>12 sec)"},
            {"name": "Functional Reach", "score": 7, "unit": "inches", "interpretation": "Fall risk (<10 inches)"},
        ],
        "functional_deficits": [
            {"category": "bed_mobility", "activity": "supine_to_sit", "prior_level": "independence", "current_level": "minimal assistance (1-25%)"},
            {"category": "transfers", "activity": "sit_to_stand", "prior_level": "independence", "current_level": "minimal assistance (1-25%)"},
            {"category": "gait", "activity": "level_surfaces", "prior_level": "independence", "current_level": "standby assistance", "assistive_device": "Rolling Walker", "distance": "150 feet", "quality_notes": "Shuffling gait, decreased stride length"},
            {"category": "gait", "activity": "stairs", "prior_level": "independence", "current_level": "minimal assistance"},
        ],
        "vitals": {
            "blood_pressure_sitting": "138/82",
            "blood_pressure_standing": "130/78",
            "heart_rate": 72,
            "spo2": 97,
            "respiratory_rate": 16,
            "pain_level": 2,
            "pain_location": "lower back",
        },
        "balance": {
            "static_sitting": "Good",
            "dynamic_sitting": "Good",
            "static_standing": "Fair",
            "dynamic_standing": "Poor",
            "single_leg_stance_right": "3 seconds",
            "single_leg_stance_left": "2 seconds",
            "tandem_stance": "Unable >5 seconds",
        },
        "social_history": "Lives at home with spouse in single-story house. 2 steps to enter with bilateral railings.",
        "past_medical_history": ["Parkinson's disease", "Hypertension", "Hyperlipidemia"],
        "medications": ["Carbidopa-Levodopa 25/100mg TID", "Lisinopril 10mg daily", "Atorvastatin 20mg daily"],
        "goals_with_baselines": [
            {"area": "Strength/MMT", "goal": "Hip extension 4/5 bilateral in 2 weeks", "baseline": "3+/5", "timeframe": "2 weeks", "type": "short_term"},
            {"area": "Balance", "goal": "Berg 48/56 in 4 weeks", "baseline": "42/56", "timeframe": "4 weeks", "type": "long_term"},
        ],
        "functional_status": [
            {"activity": "Ambulation", "level": "Standby Assist", "equipment": "Rolling Walker", "distance": "150 ft"},
        ],
        "interventions": [
            {"intervention": "Therapeutic Exercise", "duration_minutes": 15, "parameters": "LE strengthening"},
            {"intervention": "Gait Training", "duration_minutes": 15, "parameters": "with RW, visual cueing"},
            {"intervention": "Neuromuscular Re-education", "duration_minutes": 15, "parameters": "Balance training"},
        ],
    }


class TestFullEvaluationGeneration:
    def test_generates_with_all_clinical_data(self, client):
        resp = client.post("/api/v1/notes/generate", json=_full_evaluation_payload())
        assert resp.status_code == 200
        data = resp.json()

        # All clinical sections present
        assert "rom" in data["sections"]
        assert "mmt" in data["sections"]
        assert "standardized_tests" in data["sections"]
        assert "functional_deficits" in data["sections"]
        assert "vitals" in data["sections"]
        assert "social_history" in data["sections"]

    def test_rom_in_content(self, client):
        resp = client.post("/api/v1/notes/generate", json=_full_evaluation_payload())
        content = resp.json()["content"]
        assert "RANGE OF MOTION" in content
        assert "WFL" in content

    def test_mmt_in_content(self, client):
        resp = client.post("/api/v1/notes/generate", json=_full_evaluation_payload())
        content = resp.json()["content"]
        assert "MANUAL MUSCLE TESTING" in content
        assert "3+/5" in content

    def test_standardized_tests_in_content(self, client):
        resp = client.post("/api/v1/notes/generate", json=_full_evaluation_payload())
        content = resp.json()["content"]
        assert "STANDARDIZED TESTS" in content
        assert "TUG" in content
        assert "18.5" in content
        assert "Berg" in content

    def test_functional_deficits_in_content(self, client):
        resp = client.post("/api/v1/notes/generate", json=_full_evaluation_payload())
        content = resp.json()["content"]
        assert "FUNCTIONAL DEFICITS" in content
        assert "Rolling Walker" in content

    def test_vitals_in_content(self, client):
        resp = client.post("/api/v1/notes/generate", json=_full_evaluation_payload())
        content = resp.json()["content"]
        assert "138/82" in content
        assert "72 bpm" in content

    def test_goals_with_baselines_in_content(self, client):
        resp = client.post("/api/v1/notes/generate", json=_full_evaluation_payload())
        content = resp.json()["content"]
        assert "Baseline:" in content


class TestMedicareComplianceWithClinicalData:
    def test_full_eval_compliant(self, client):
        resp = client.post("/api/v1/notes/generate", json=_full_evaluation_payload())
        data = resp.json()
        checklist = data["compliance_checklist"]
        assert checklist.get("rom_documented") is True
        assert checklist.get("mmt_documented") is True
        assert checklist.get("standardized_tests_documented") is True
        assert checklist.get("functional_deficits_documented") is True

    def test_eval_without_clinical_data_warns(self, client):
        payload = {
            "note_type": "evaluation",
            "discipline": "pt",
            "diagnosis": ["R26.2"],
            "functional_status": [{"activity": "Ambulation", "level": "Min A"}],
            "interventions": [{"intervention": "Gait training", "duration_minutes": 15}],
        }
        resp = client.post("/api/v1/notes/generate", json=payload)
        data = resp.json()
        warnings = data["compliance_warnings"]
        assert any("ROM" in w for w in warnings)
        assert any("MMT" in w for w in warnings)
        assert any("standardized tests" in w for w in warnings)


class TestGeneratedNoteStructuredData:
    """Verify GeneratedNote includes machine-readable clinical_data."""

    def test_clinical_data_populated_from_structured_request(self, client):
        resp = client.post("/api/v1/notes/generate", json=_full_evaluation_payload())
        data = resp.json()
        cd = data["clinical_data"]
        assert cd is not None
        assert len(cd["rom"]) == 5
        assert len(cd["mmt"]) == 6
        assert len(cd["standardized_tests"]) == 5
        assert len(cd["functional_deficits"]) == 4
        assert cd["vitals"]["heart_rate"] == 72
        assert cd["balance"]["dynamic_standing"] == "Poor"
        assert "Parkinson" in cd["past_medical_history"][0]

    def test_clinical_data_none_when_no_clinical_fields(self, client):
        resp = client.post("/api/v1/notes/generate", json={
            "note_type": "daily_note",
            "discipline": "pt",
            "interventions": [{"intervention": "Gait training", "duration_minutes": 15}],
        })
        assert resp.json()["clinical_data"] is None

    def test_clinical_data_mmt_grades_preserved(self, client):
        resp = client.post("/api/v1/notes/generate", json=_full_evaluation_payload())
        mmt = resp.json()["clinical_data"]["mmt"]
        grades = {e["muscle_group"]: e["grade"] for e in mmt}
        assert grades["hip_extension"] == "3+/5"
        assert grades["knee_extension"] == "4/5"

    def test_clinical_data_standardized_test_interpretation(self, client):
        resp = client.post("/api/v1/notes/generate", json=_full_evaluation_payload())
        tests = {t["name"]: t for t in resp.json()["clinical_data"]["standardized_tests"]}
        assert tests["TUG"]["score"] == 18.5
        assert tests["TUG"]["unit"] == "seconds"
        assert "fall risk" in tests["TUG"]["interpretation"].lower()
        assert tests["Berg Balance Scale"]["max_score"] == 56


class TestTranscriptClinicalDataExtraction:
    """Verify that LLM-extracted clinical_data flows through."""

    @pytest.mark.asyncio
    async def test_clinical_data_extracted_from_transcript(self):
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "subjective": "Pt reports feeling unsteady.",
            "objective": "ROM: WFL bilateral LE. MMT: hip flexors 4/5 bilat. TUG 18.5 sec.",
            "assessment": "High fall risk per TUG. Skilled PT warranted.",
            "plan": "PT 2x/wk x 4 wks.",
            "clinical_data": {
                "rom": [
                    {"joint": "bilateral_hips", "motion": "general", "qualitative": "WFL", "side": "bilateral"},
                ],
                "mmt": [
                    {"muscle_group": "hip_flexion", "grade": "4/5", "side": "bilateral"},
                ],
                "standardized_tests": [
                    {"name": "TUG", "score": 18.5, "unit": "seconds", "interpretation": "High fall risk"},
                ],
                "functional_deficits": [
                    {"category": "gait", "activity": "level_surfaces", "prior_level": "independent", "current_level": "SBA with RW"},
                ],
                "vitals": {"heart_rate": 72, "spo2": 97},
            }
        })
        mock_llm.complete = AsyncMock(return_value=mock_response)

        result = await _generate_from_transcript(
            transcript="Patient eval, ROM WFL bilateral, hip flexors 4/5, TUG 18.5 seconds...",
            note_type="evaluation",
            patient_context=None,
            preferences=None,
            llm_router=mock_llm,
        )
        assert isinstance(result, GeneratedNote)
        assert result.clinical_data is not None
        assert len(result.clinical_data.rom) == 1
        assert result.clinical_data.rom[0].qualitative == "WFL"
        assert len(result.clinical_data.mmt) == 1
        assert result.clinical_data.mmt[0].grade == "4/5"
        assert result.clinical_data.standardized_tests[0].score == 18.5
        assert result.clinical_data.vitals.heart_rate == 72

    @pytest.mark.asyncio
    async def test_no_clinical_data_from_simple_transcript(self):
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "subjective": "Pt doing well.",
            "objective": "Gait training x 15 min.",
            "assessment": "Progressing.",
            "plan": "Continue.",
        })
        mock_llm.complete = AsyncMock(return_value=mock_response)

        result = await _generate_from_transcript(
            transcript="Quick session, gait training.",
            note_type="daily_note",
            patient_context=None,
            preferences=None,
            llm_router=mock_llm,
        )
        assert result.clinical_data is None


class TestClinicalModelsValidation:
    def test_rom_entry_qualitative_only(self):
        e = ROMEntry(joint="cervical", motion="general", qualitative="WFL")
        assert e.value is None
        assert e.qualitative == "WFL"

    def test_rom_entry_numeric(self):
        e = ROMEntry(joint="right_knee", motion="flexion", value=120.0, side="right")
        assert e.value == 120.0

    def test_mmt_entry(self):
        e = MMTEntry(muscle_group="hip_flexion", grade="4/5", side="left")
        assert e.grade == "4/5"

    def test_standardized_test_with_sub_scores(self):
        t = StandardizedTest(
            name="Tinetti", score=20, max_score=28,
            sub_scores={"balance": 12, "gait": 8},
        )
        assert t.sub_scores["balance"] == 12

    def test_functional_deficit(self):
        d = FunctionalDeficit(
            category="gait", activity="level_surfaces",
            prior_level="independence", current_level="SBA",
            assistive_device="RW", distance="150 ft",
        )
        assert d.assistive_device == "RW"

    def test_balance_assessment(self):
        b = BalanceAssessment(static_standing="Fair", dynamic_standing="Poor")
        assert b.static_standing == "Fair"

    def test_goal_with_baseline(self):
        g = GoalWithBaseline(area="TUG", goal="<14 sec in 4 weeks", baseline="18.5 sec")
        assert g.type == "short_term"
