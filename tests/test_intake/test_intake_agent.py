"""Tests for the intake module — all with mocks, no real LLM calls."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rehab_os.intake.agent import IntakeAgent, IntakeInput, IntakeResult
from rehab_os.intake.extractor import extract_text
from rehab_os.intake.pipeline import IntakePipeline
from rehab_os.models.patient import CareSetting, Discipline, PatientContext

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_REFERRAL = (FIXTURES / "sample_referral.txt").read_text()

# ---------------------------------------------------------------------------
# A realistic mock IntakeResult for reuse
# ---------------------------------------------------------------------------

MOCK_PATIENT = PatientContext(
    age=67,
    sex="female",
    chief_complaint="Left total knee arthroplasty rehabilitation",
    diagnosis=["Left total knee arthroplasty", "Primary osteoarthritis left knee"],
    icd_codes=["Z96.652", "M17.12"],
    comorbidities=["HTN", "T2DM", "hypothyroidism"],
    medications=[
        "Lisinopril 10mg",
        "Metformin 1000mg BID",
        "Levothyroxine 50mcg",
        "Oxycodone 5mg PRN",
    ],
    allergies=["Sulfa drugs", "Latex"],
    surgical_history=["Right TKA (2024)", "Cholecystectomy (2019)"],
    precautions=[
        "WBAT L LE",
        "CPM 0-90 degrees",
        "No active knee extension exercises x6 weeks",
    ],
    setting=CareSetting.OUTPATIENT,
    discipline=Discipline.PT,
    prior_level_of_function="Independent with all ADLs/IADLs, walked 1.5 miles daily",
    physician_orders="PT evaluate and treat. Focus on ROM, strengthening, gait training, stair negotiation.",
)

MOCK_RESULT = IntakeResult(
    patient=MOCK_PATIENT,
    referral_summary="67yo female referred for PT s/p left TKA POD 2 by Dr. James Chen.",
    referring_provider="Dr. James Chen, MD",
    referring_diagnosis=["Left total knee arthroplasty", "Primary osteoarthritis left knee"],
    icd_codes_extracted=["Z96.652", "M17.12"],
    insurance_info={
        "payer": "Blue Cross Blue Shield",
        "member_id": "BCB882991",
        "group_id": "GRP-445",
        "auth_number": "PA-2026-88431",
    },
    visit_authorization={
        "authorized_visits": 12,
        "frequency": "2-3x/week",
        "duration": "6 weeks",
    },
    extraction_confidence=0.92,
    missing_fields=["height_cm", "weight_kg", "vitals"],
    raw_text_snippet=SAMPLE_REFERRAL[:500],
)


# ===========================================================================
# Extractor tests
# ===========================================================================


class TestExtractor:
    """Test text extraction routing."""

    def test_extract_text_pdf_route(self):
        """PDF content type routes to PDF extractor."""
        with patch("rehab_os.intake.extractor.extract_text_from_pdf", return_value="pdf text"):
            result = extract_text(b"fake-pdf", "application/pdf")
            assert result == "pdf text"

    def test_extract_text_image_routes(self):
        """Image content types route to image extractor."""
        for ct in ("image/png", "image/jpeg", "image/tiff"):
            with patch(
                "rehab_os.intake.extractor.extract_text_from_image",
                return_value="img text",
            ):
                result = extract_text(b"fake-img", ct)
                assert result == "img text"

    def test_extract_text_unsupported(self):
        """Unsupported content type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported content type"):
            extract_text(b"data", "application/zip")

    def test_extract_text_case_insensitive(self):
        """Content type matching is case-insensitive."""
        with patch("rehab_os.intake.extractor.extract_text_from_pdf", return_value="ok"):
            result = extract_text(b"data", "Application/PDF")
            assert result == "ok"


# ===========================================================================
# IntakeAgent tests
# ===========================================================================


class TestIntakeAgent:
    """Test IntakeAgent with mock LLM."""

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.complete_structured = AsyncMock(return_value=MOCK_RESULT)
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        return IntakeAgent(mock_llm)

    async def test_agent_returns_patient_context(self, agent):
        result = await agent.run(IntakeInput(raw_text=SAMPLE_REFERRAL))
        assert isinstance(result, IntakeResult)
        assert isinstance(result.patient, PatientContext)
        assert result.patient.age == 67
        assert result.patient.sex == "female"

    async def test_agent_extracts_diagnoses(self, agent):
        result = await agent.run(IntakeInput(raw_text=SAMPLE_REFERRAL))
        assert len(result.referring_diagnosis) > 0
        assert len(result.icd_codes_extracted) > 0

    async def test_agent_insurance_info(self, agent):
        result = await agent.run(IntakeInput(raw_text=SAMPLE_REFERRAL))
        assert result.insurance_info is not None
        assert "payer" in result.insurance_info

    async def test_agent_visit_authorization(self, agent):
        result = await agent.run(IntakeInput(raw_text=SAMPLE_REFERRAL))
        assert result.visit_authorization is not None
        assert result.visit_authorization["authorized_visits"] == 12

    async def test_missing_fields_detected(self, agent):
        result = await agent.run(IntakeInput(raw_text=SAMPLE_REFERRAL))
        assert "height_cm" in result.missing_fields
        assert "weight_kg" in result.missing_fields

    async def test_confidence_scoring(self, agent):
        result = await agent.run(IntakeInput(raw_text=SAMPLE_REFERRAL))
        assert 0 <= result.extraction_confidence <= 1

    async def test_raw_text_snippet(self, agent):
        result = await agent.run(IntakeInput(raw_text=SAMPLE_REFERRAL))
        assert len(result.raw_text_snippet) <= 500

    async def test_agent_properties(self, agent):
        assert agent.temperature == 0.2
        assert agent.model_tier.value == "standard"
        assert agent.name == "intake"

    async def test_format_input_includes_source_type(self, agent):
        from rehab_os.agents.base import AgentContext

        inp = IntakeInput(
            raw_text="test", source_type="prescription", referring_provider="Dr. Smith"
        )
        formatted = agent.format_input(inp, AgentContext())
        assert "prescription" in formatted
        assert "Dr. Smith" in formatted


# ===========================================================================
# Pipeline tests
# ===========================================================================


class TestIntakePipeline:
    """Test IntakePipeline end-to-end with mock LLM."""

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.complete_structured = AsyncMock(return_value=MOCK_RESULT)
        return llm

    @pytest.fixture
    def pipeline(self, mock_llm):
        return IntakePipeline(llm=mock_llm)

    async def test_process_raw_text(self, pipeline):
        result = await pipeline.process_raw_text(SAMPLE_REFERRAL)
        assert isinstance(result, IntakeResult)
        assert result.patient.age == 67

    async def test_process_referral_pdf(self, pipeline):
        with patch(
            "rehab_os.intake.extractor.extract_text_from_pdf",
            return_value=SAMPLE_REFERRAL,
        ):
            result = await pipeline.process_referral(
                b"fake-pdf", "application/pdf", {"source_type": "referral"}
            )
            assert isinstance(result, IntakeResult)

    async def test_pipeline_stores_to_memory(self, mock_llm):
        memory = MagicMock()
        memory._cache = {}
        pipeline = IntakePipeline(llm=mock_llm, session_memory=memory)
        await pipeline.process_raw_text(SAMPLE_REFERRAL)
        # Should have stored something in the cache
        assert len(memory._cache) > 0


# ===========================================================================
# API route tests
# ===========================================================================


class TestIntakeAPI:
    """Test intake API endpoints with TestClient."""

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from rehab_os.api.routes.intake import router

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")

        # Mock app state
        mock_llm = MagicMock()
        mock_llm.complete_structured = AsyncMock(return_value=MOCK_RESULT)
        app.state.llm_router = mock_llm
        app.state.session_memory = None

        return TestClient(app)

    def test_get_templates(self, client):
        resp = client.get("/api/v1/intake/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "source_types" in data
        assert len(data["source_types"]) > 0

    def test_post_text(self, client):
        resp = client.post(
            "/api/v1/intake/text",
            json={"text": SAMPLE_REFERRAL, "source_type": "referral"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "patient" in data
        assert data["patient"]["age"] == 67

    def test_post_text_empty(self, client):
        resp = client.post("/api/v1/intake/text", json={"text": "", "source_type": "referral"})
        assert resp.status_code == 400

    def test_upload_unsupported_type(self, client):
        resp = client.post(
            "/api/v1/intake/upload",
            files={"file": ("test.zip", b"data", "application/zip")},
        )
        assert resp.status_code == 400

    def test_upload_pdf(self, client):
        with patch(
            "rehab_os.intake.extractor.extract_text_from_pdf",
            return_value=SAMPLE_REFERRAL,
        ):
            resp = client.post(
                "/api/v1/intake/upload",
                files={"file": ("referral.pdf", b"fake-pdf", "application/pdf")},
                data={"source_type": "referral"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "patient" in data
