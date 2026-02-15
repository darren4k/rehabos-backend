"""Tests for Patient-Core repositories using async SQLite."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rehab_os.core.models import Base
from rehab_os.core.repository import (
    AuditRepository,
    EncounterRepository,
    InsuranceRepository,
    PatientRepository,
    ReferralRepository,
)


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as sess:
        yield sess
        await sess.rollback()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# --- Patient ---

async def test_patient_create_and_get(session: AsyncSession):
    repo = PatientRepository(session)
    p = await repo.create(
        first_name="John", last_name="Doe", dob=date(1990, 1, 15), sex="male"
    )
    assert p.id is not None
    assert p.first_name == "John"

    fetched = await repo.get_by_id(p.id)
    assert fetched is not None
    assert fetched.last_name == "Doe"


async def test_patient_update(session: AsyncSession):
    repo = PatientRepository(session)
    p = await repo.create(first_name="Jane", last_name="Smith", dob=date(1985, 6, 1), sex="female")
    updated = await repo.update(p.id, phone="555-1234")
    assert updated is not None
    assert updated.phone == "555-1234"


async def test_patient_search(session: AsyncSession):
    repo = PatientRepository(session)
    await repo.create(first_name="Alice", last_name="Johnson", dob=date(2000, 3, 10), sex="female")
    await repo.create(first_name="Bob", last_name="Williams", dob=date(1975, 11, 22), sex="male")
    await session.flush()

    results = await repo.search_by_name("john")
    assert len(results) == 1
    assert results[0].first_name == "Alice"


async def test_patient_list_and_deactivate(session: AsyncSession):
    repo = PatientRepository(session)
    p = await repo.create(first_name="Carl", last_name="Test", dob=date(1960, 1, 1), sex="male")
    await session.flush()

    all_active = await repo.list()
    assert any(x.id == p.id for x in all_active)

    await repo.deactivate(p.id)
    await session.flush()

    all_active = await repo.list(active_only=True)
    assert not any(x.id == p.id for x in all_active)


# --- Encounter ---

async def test_encounter_create_and_list(session: AsyncSession):
    prepo = PatientRepository(session)
    patient = await prepo.create(first_name="Test", last_name="Pat", dob=date(1990, 1, 1), sex="male")
    await session.flush()

    erepo = EncounterRepository(session)
    enc = await erepo.create(
        patient_id=patient.id,
        encounter_date=datetime.now(timezone.utc),
        discipline="PT",
        encounter_type="initial_eval",
    )
    assert enc.status == "scheduled"

    encs = await erepo.list_by_patient(patient.id)
    assert len(encs) == 1

    await erepo.update_status(enc.id, "in_progress")
    updated = await erepo.get_by_id(enc.id)
    assert updated is not None
    assert updated.status == "in_progress"


# --- Referral ---

async def test_referral_status_workflow(session: AsyncSession):
    prepo = PatientRepository(session)
    patient = await prepo.create(first_name="Ref", last_name="Test", dob=date(1980, 5, 5), sex="female")
    await session.flush()

    rrepo = ReferralRepository(session)
    ref = await rrepo.create(
        patient_id=patient.id,
        referring_provider_name="Dr. Smith",
        diagnosis_codes=["M54.5"],
        status="pending",
    )

    pending = await rrepo.list_pending()
    assert len(pending) == 1

    await rrepo.update_status(ref.id, "accepted")
    await session.flush()

    pending = await rrepo.list_pending()
    assert len(pending) == 0

    updated = await rrepo.get_by_id(ref.id)
    assert updated is not None
    assert updated.status == "accepted"


# --- Audit ---

async def test_audit_logging(session: AsyncSession):
    arepo = AuditRepository(session)
    entry = await arepo.log_action(
        action="create",
        resource_type="patient",
        resource_id="test-123",
        user_id="admin",
        details={"note": "test"},
    )
    assert entry.id is not None

    logs = await arepo.get_by_resource("patient", "test-123")
    assert len(logs) == 1
    assert logs[0].action == "create"
