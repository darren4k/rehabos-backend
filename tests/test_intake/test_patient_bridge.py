"""Tests for Intake → Patient-Core bridge."""

import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rehab_os.intake.pipeline import IntakePipeline, _estimate_dob
from rehab_os.models.patient import PatientContext


def _make_patient_context(**overrides) -> PatientContext:
    defaults = dict(
        age=65,
        sex="male",
        chief_complaint="Low back pain",
        diagnosis=["Lumbar radiculopathy"],
        icd_codes=["M54.16"],
    )
    defaults.update(overrides)
    return PatientContext(**defaults)


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def mock_db():
    return AsyncMock()


class TestPatientContextNewFields:
    def test_name_optional(self):
        ctx = _make_patient_context()
        assert ctx.name is None

    def test_name_set(self):
        ctx = _make_patient_context(name="John Doe")
        assert ctx.name == "John Doe"

    def test_date_of_birth_optional(self):
        ctx = _make_patient_context()
        assert ctx.date_of_birth is None

    def test_date_of_birth_set(self):
        ctx = _make_patient_context(date_of_birth=date(1960, 5, 15))
        assert ctx.date_of_birth == date(1960, 5, 15)


class TestEstimateDob:
    def test_estimate_dob(self):
        dob = _estimate_dob(65)
        assert dob.year == date.today().year - 65
        assert dob.month == 1
        assert dob.day == 1


class TestCreatePatientRecord:
    @pytest.mark.asyncio
    async def test_no_db_session_returns_none(self, mock_llm):
        pipeline = IntakePipeline(llm=mock_llm, db_session=None)
        ctx = _make_patient_context(name="John Doe")
        result = await pipeline._create_patient_record(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_name_returns_none(self, mock_llm, mock_db):
        pipeline = IntakePipeline(llm=mock_llm, db_session=mock_db)
        ctx = _make_patient_context()
        result = await pipeline._create_patient_record(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_creates_new_patient(self, mock_llm, mock_db):
        pipeline = IntakePipeline(llm=mock_llm, db_session=mock_db)
        ctx = _make_patient_context(name="John Doe", date_of_birth=date(1960, 5, 15))

        fake_id = uuid.uuid4()
        fake_patient = MagicMock()
        fake_patient.id = fake_id

        with patch("rehab_os.intake.pipeline.PatientRepository") as MockRepo:
            repo_instance = MockRepo.return_value
            repo_instance.search_by_name = AsyncMock(return_value=[])
            repo_instance.create = AsyncMock(return_value=fake_patient)

            result = await pipeline._create_patient_record(ctx)

        assert result == fake_id
        repo_instance.create.assert_called_once()
        call_kwargs = repo_instance.create.call_args[1]
        assert call_kwargs["first_name"] == "John"
        assert call_kwargs["last_name"] == "Doe"
        assert call_kwargs["dob"] == date(1960, 5, 15)
        assert call_kwargs["sex"] == "male"

    @pytest.mark.asyncio
    async def test_finds_existing_patient(self, mock_llm, mock_db):
        pipeline = IntakePipeline(llm=mock_llm, db_session=mock_db)
        ctx = _make_patient_context(name="John Doe", date_of_birth=date(1960, 5, 15))

        existing_id = uuid.uuid4()
        existing = MagicMock()
        existing.id = existing_id
        existing.first_name = "John"
        existing.last_name = "Doe"
        existing.dob = date(1960, 5, 15)

        with patch("rehab_os.intake.pipeline.PatientRepository") as MockRepo:
            repo_instance = MockRepo.return_value
            repo_instance.search_by_name = AsyncMock(return_value=[existing])
            repo_instance.create = AsyncMock()

            result = await pipeline._create_patient_record(ctx)

        assert result == existing_id
        repo_instance.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_name_handled(self, mock_llm, mock_db):
        pipeline = IntakePipeline(llm=mock_llm, db_session=mock_db)
        ctx = _make_patient_context(name="Madonna")

        fake_patient = MagicMock()
        fake_patient.id = uuid.uuid4()

        with patch("rehab_os.intake.pipeline.PatientRepository") as MockRepo:
            repo_instance = MockRepo.return_value
            repo_instance.search_by_name = AsyncMock(return_value=[])
            repo_instance.create = AsyncMock(return_value=fake_patient)

            result = await pipeline._create_patient_record(ctx)

        call_kwargs = repo_instance.create.call_args[1]
        assert call_kwargs["first_name"] == "Madonna"
        assert call_kwargs["last_name"] == ""

    @pytest.mark.asyncio
    async def test_no_dob_estimates_from_age(self, mock_llm, mock_db):
        pipeline = IntakePipeline(llm=mock_llm, db_session=mock_db)
        ctx = _make_patient_context(name="John Doe")

        fake_patient = MagicMock()
        fake_patient.id = uuid.uuid4()

        with patch("rehab_os.intake.pipeline.PatientRepository") as MockRepo:
            repo_instance = MockRepo.return_value
            repo_instance.search_by_name = AsyncMock(return_value=[])
            repo_instance.create = AsyncMock(return_value=fake_patient)

            await pipeline._create_patient_record(ctx)

        call_kwargs = repo_instance.create.call_args[1]
        assert call_kwargs["dob"] == date(date.today().year - 65, 1, 1)
