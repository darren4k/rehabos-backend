"""Tests for transcript-based SOAP note generation."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rehab_os.api.routes.notes import router, _generate_from_transcript, GeneratedNote
from tests.conftest import apply_auth_override


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Mock LLM router
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "subjective": "Patient reports decreased pain in right knee, 4/10 today vs 6/10 last visit.",
        "objective": "Therapeutic exercise x 15 min, gait training x 10 min w/ rolling walker. Pt ambulated 150 ft w/ min A, improved from 100 ft last session.",
        "assessment": "Patient demonstrating progress toward functional mobility goals. Continued skilled physical therapy warranted for neuromuscular re-education and functional mobility training.",
        "plan": "Continue PT 3x/wk. Progress gait training to least restrictive device. Patient education on HEP compliance."
    })
    mock_llm.complete = AsyncMock(return_value=mock_response)
    app.state.llm_router = mock_llm

    apply_auth_override(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestGenerateFromTranscript:
    def test_basic_transcript(self, client):
        resp = client.post("/api/v1/notes/generate-from-transcript", json={
            "transcript": "Patient came in today said knee feels better. Did exercises and walking with walker. Walked 150 feet with minimal help.",
            "note_type": "daily_note",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "subjective" in data["sections"]
        assert "objective" in data["sections"]
        assert data["note_type"] == "daily_note"
        assert data["word_count"] > 0

    def test_with_patient_context(self, client):
        resp = client.post("/api/v1/notes/generate-from-transcript", json={
            "transcript": "Worked on balance and transfers today.",
            "note_type": "daily_note",
            "patient_context": {
                "diagnosis": ["s/p R TKA"],
                "age": 72,
                "setting": "SNF"
            },
        })
        assert resp.status_code == 200

    def test_with_preferences(self, client):
        resp = client.post("/api/v1/notes/generate-from-transcript", json={
            "transcript": "Did gait training and therapeutic exercise.",
            "note_type": "progress_note",
            "preferences": {"style": "concise"},
        })
        assert resp.status_code == 200
        assert resp.json()["style_used"] == "concise"

    def test_medicare_compliance_checked(self, client):
        resp = client.post("/api/v1/notes/generate-from-transcript", json={
            "transcript": "Gait training with patient today.",
            "note_type": "daily_note",
        })
        data = resp.json()
        assert "compliance_checklist" in data
        assert "skilled_interventions_documented" in data["compliance_checklist"]
        assert isinstance(data["medicare_compliant"], bool)


class TestGenerateWithTranscriptAdapter:
    def test_existing_endpoint_with_transcript(self, client):
        resp = client.post("/api/v1/notes/generate", json={
            "note_type": "daily_note",
            "discipline": "pt",
            "transcript": "Patient tolerated treatment well. Gait training 150 ft.",
        })
        assert resp.status_code == 200
        data = resp.json()
        # Should have gone through LLM path
        assert "subjective" in data["sections"]

    def test_existing_endpoint_without_transcript(self, client):
        """Original structured path still works."""
        resp = client.post("/api/v1/notes/generate", json={
            "note_type": "daily_note",
            "discipline": "pt",
            "functional_status": [{"activity": "Ambulation", "level": "Min A"}],
            "interventions": [{"intervention": "Gait training", "duration_minutes": 15}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "skilled_interventions" in data["sections"]


class TestGenerateFromTranscriptUnit:
    @pytest.mark.asyncio
    async def test_handles_invalid_json_from_llm(self):
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "This is not JSON, just a plain text note."
        mock_llm.complete = AsyncMock(return_value=mock_response)

        result = await _generate_from_transcript(
            transcript="Some dictation",
            note_type="daily_note",
            patient_context=None,
            preferences=None,
            llm_router=mock_llm,
        )
        assert isinstance(result, GeneratedNote)
        assert "raw_output" in result.sections

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '```json\n{"subjective":"test","objective":"test","assessment":"test","plan":"test"}\n```'
        mock_llm.complete = AsyncMock(return_value=mock_response)

        result = await _generate_from_transcript(
            transcript="Some dictation",
            note_type="daily_note",
            patient_context=None,
            preferences=None,
            llm_router=mock_llm,
        )
        assert result.sections["subjective"] == "test"
