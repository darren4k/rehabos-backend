"""Tests for export API routes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date
from uuid import uuid4

from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from rehab_os.api.routes.export import router
from tests.conftest import apply_auth_override


# Minimal test app
def _create_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    apply_auth_override(app)
    return app


@pytest.fixture
def app():
    return _create_test_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestPostExportPDF:
    async def test_returns_pdf(self, client: AsyncClient):
        payload = {
            "note_type": "daily_note",
            "note_date": "2026-02-18",
            "discipline": "pt",
            "therapist_name": "Test Therapist",
            "soap_subjective": "Patient reports improvement.",
            "soap_objective": "AROM WNL.",
            "soap_assessment": "Progressing well.",
            "soap_plan": "Continue POC.",
        }
        resp = await client.post("/api/v1/notes/export/pdf", json=payload)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert resp.content[:5] == b"%PDF-"

    async def test_returns_pdf_minimal(self, client: AsyncClient):
        payload = {
            "note_type": "progress_note",
            "note_date": "2026-02-18",
            "discipline": "slp",
        }
        resp = await client.post("/api/v1/notes/export/pdf", json=payload)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"

    async def test_content_disposition_header(self, client: AsyncClient):
        payload = {
            "note_type": "evaluation",
            "note_date": "2026-02-18",
            "discipline": "ot",
        }
        resp = await client.post("/api/v1/notes/export/pdf", json=payload)
        assert "attachment" in resp.headers.get("content-disposition", "")


class TestGetExportPDF:
    async def test_404_missing_note(self, client: AsyncClient, app: FastAPI):
        """GET endpoint requires DB — mock the dependency to test 404 path."""
        mock_db = AsyncMock()
        mock_repo_cls = MagicMock()
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_by_id = AsyncMock(return_value=None)
        mock_repo_cls.return_value = mock_repo_instance

        with patch("rehab_os.api.routes.export.ClinicalNoteRepository", mock_repo_cls):
            with patch("rehab_os.api.routes.export.get_db") as mock_get_db:
                async def fake_db():
                    yield mock_db
                mock_get_db.return_value = fake_db()

                # Re-create app with patched deps
                patched_app = _create_test_app()
                transport = ASGITransport(app=patched_app)
                async with AsyncClient(transport=transport, base_url="http://test") as c:
                    note_id = str(uuid4())
                    resp = await c.get(f"/api/v1/notes/{note_id}/export/pdf")
                    assert resp.status_code == 404

    async def test_200_with_note(self, client: AsyncClient):
        """GET endpoint with a mocked note should return PDF."""
        mock_note = MagicMock()
        mock_note.note_type = "daily_note"
        mock_note.note_date = date(2026, 2, 18)
        mock_note.discipline = "pt"
        mock_note.therapist_name = "Test Therapist"
        mock_note.soap_subjective = "Subjective text"
        mock_note.soap_objective = "Objective text"
        mock_note.soap_assessment = "Assessment text"
        mock_note.soap_plan = "Plan text"
        mock_note.structured_data = None
        mock_note.compliance_score = 90.0
        mock_note.compliance_warnings = []

        mock_repo_cls = MagicMock()
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_by_id = AsyncMock(return_value=mock_note)
        mock_repo_cls.return_value = mock_repo_instance

        mock_db = AsyncMock()

        with patch("rehab_os.api.routes.export.ClinicalNoteRepository", mock_repo_cls):
            with patch("rehab_os.api.routes.export.get_db") as mock_get_db:
                async def fake_db():
                    yield mock_db
                mock_get_db.return_value = fake_db()

                patched_app = _create_test_app()
                transport = ASGITransport(app=patched_app)
                async with AsyncClient(transport=transport, base_url="http://test") as c:
                    note_id = str(uuid4())
                    resp = await c.get(f"/api/v1/notes/{note_id}/export/pdf")
                    assert resp.status_code == 200
                    assert resp.headers["content-type"] == "application/pdf"
                    assert resp.content[:5] == b"%PDF-"
