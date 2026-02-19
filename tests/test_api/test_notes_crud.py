"""Tests for Clinical Note CRUD endpoints."""

import uuid
from datetime import date, datetime, timezone

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rehab_os.core.models import ClinicalNote
from rehab_os.core.schemas import ClinicalNoteCreate, ClinicalNoteRead, ClinicalNoteUpdate


@pytest.fixture
def sample_note_data():
    return {
        "patient_id": uuid.uuid4(),
        "note_type": "daily_note",
        "note_date": date.today(),
        "discipline": "pt",
        "therapist_name": "Dr. Smith",
        "soap_subjective": "Patient reports pain 3/10",
        "soap_objective": "AROM R knee flex 95°",
        "soap_assessment": "Patient progressing well",
        "soap_plan": "Continue PT 3x/week",
        "structured_data": {
            "rom": [{"joint": "right_knee", "motion": "flexion", "value": 95, "side": "right"}],
            "mmt": [{"muscle_group": "quads", "grade": "4/5", "side": "right"}],
            "standardized_tests": [{"name": "TUG", "score": 14, "unit": "seconds"}],
        },
        "compliance_score": 85.0,
        "compliance_warnings": ["Consider adding prior level of function"],
        "status": "final",
    }


class TestClinicalNoteSchemas:
    def test_create_schema(self, sample_note_data):
        schema = ClinicalNoteCreate(**sample_note_data)
        assert schema.note_type == "daily_note"
        assert schema.discipline == "pt"
        assert schema.soap_subjective == "Patient reports pain 3/10"

    def test_create_schema_defaults(self):
        schema = ClinicalNoteCreate(
            patient_id=uuid.uuid4(),
            note_type="evaluation",
            note_date=date.today(),
        )
        assert schema.discipline == "pt"
        assert schema.status == "final"
        assert schema.emr_synced is False

    def test_update_schema_partial(self):
        schema = ClinicalNoteUpdate(soap_subjective="Updated subjective")
        assert schema.soap_subjective == "Updated subjective"
        assert schema.note_type is None

    def test_read_schema(self):
        now = datetime.now(timezone.utc)
        schema = ClinicalNoteRead(
            id=uuid.uuid4(),
            patient_id=uuid.uuid4(),
            note_type="progress_note",
            note_date=date.today(),
            discipline="pt",
            status="final",
            emr_synced=False,
            created_at=now,
        )
        assert schema.note_type == "progress_note"


class TestClinicalNoteRepository:
    @pytest.mark.asyncio
    async def test_create_note(self, sample_note_data):
        from rehab_os.core.repository import ClinicalNoteRepository

        mock_session = AsyncMock()
        repo = ClinicalNoteRepository(mock_session)
        
        note = await repo.create(**sample_note_data)
        assert note.note_type == "daily_note"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id(self):
        from rehab_os.core.repository import ClinicalNoteRepository

        mock_session = AsyncMock()
        note_id = uuid.uuid4()
        repo = ClinicalNoteRepository(mock_session)
        await repo.get_by_id(note_id)
        mock_session.get.assert_awaited_once_with(ClinicalNote, note_id)

    @pytest.mark.asyncio
    async def test_list_by_patient(self):
        from rehab_os.core.repository import ClinicalNoteRepository

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = ClinicalNoteRepository(mock_session)
        result = await repo.list_by_patient(uuid.uuid4())
        assert result == []

    @pytest.mark.asyncio
    async def test_list_by_patient_with_type_filter(self):
        from rehab_os.core.repository import ClinicalNoteRepository

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = ClinicalNoteRepository(mock_session)
        result = await repo.list_by_patient(uuid.uuid4(), note_type="evaluation")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_notes(self):
        from rehab_os.core.repository import ClinicalNoteRepository

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = ClinicalNoteRepository(mock_session)
        result = await repo.search_notes(uuid.uuid4(), "knee")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_latest_by_type(self):
        from rehab_os.core.repository import ClinicalNoteRepository

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = ClinicalNoteRepository(mock_session)
        result = await repo.get_latest_by_type(uuid.uuid4(), "evaluation")
        assert result is None
