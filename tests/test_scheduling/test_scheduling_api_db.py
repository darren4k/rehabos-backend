"""DB-backed integration tests for scheduling API endpoints."""

import uuid
from datetime import datetime, timedelta, timezone, date

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rehab_os.core.models import Base, Patient, Provider, Insurance, AppointmentDB
from rehab_os.core.database import get_db
from rehab_os.api.routes.scheduling import router

from fastapi import FastAPI
from tests.conftest import apply_auth_override


# ---------------------------------------------------------------------------
# Fixtures: in-memory SQLite engine + session override
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as sess:
        yield sess


@pytest_asyncio.fixture
async def seed_data(session: AsyncSession):
    """Seed a patient, provider, and insurance record."""
    patient = Patient(
        id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        first_name="Jane", last_name="Doe",
        dob=date(1960, 5, 15), sex="female",
    )
    provider = Provider(
        id=uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        first_name="Sarah", last_name="Chen",
        discipline="PT",
    )
    insurance = Insurance(
        patient_id=patient.id,
        payer_name="Aetna",
        member_id="MEM-001",
        authorized_visits=20,
        visits_used=17,
        is_primary=True,
        expiry_date=date(2027, 12, 31),
    )
    session.add_all([patient, provider, insurance])
    await session.commit()
    return {"patient": patient, "provider": provider, "insurance": insurance}


@pytest_asyncio.fixture
async def client(engine, seed_data):
    """AsyncClient bound to a FastAPI app using test DB session."""
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_get_db():
        async with factory() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_db] = _override_get_db
    apply_auth_override(app)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


PAT_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
PROV_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _appt_body(
    start_offset_hours: int = 0,
    discipline: str = "PT",
    encounter_type: str = "treatment",
    location: str | None = None,
) -> dict:
    base = datetime(2026, 4, 6, 9, 0, tzinfo=timezone.utc) + timedelta(hours=start_offset_hours)
    return {
        "patient_id": PAT_ID,
        "provider_id": PROV_ID,
        "start_time": base.isoformat(),
        "end_time": (base + timedelta(minutes=45)).isoformat(),
        "discipline": discipline,
        "encounter_type": encounter_type,
        "location": location,
    }


# ---------------------------------------------------------------------------
# Appointment CRUD
# ---------------------------------------------------------------------------

class TestAppointmentCRUD:
    @pytest.mark.asyncio
    async def test_create_appointment(self, client: AsyncClient):
        resp = await client.post("/api/v1/scheduling/appointments", json=_appt_body())
        assert resp.status_code == 201
        data = resp.json()
        assert data["patient_id"] == PAT_ID
        assert data["status"] == "scheduled"

    @pytest.mark.asyncio
    async def test_list_by_provider(self, client: AsyncClient):
        await client.post("/api/v1/scheduling/appointments", json=_appt_body(0))
        await client.post("/api/v1/scheduling/appointments", json=_appt_body(1))
        resp = await client.get("/api/v1/scheduling/appointments", params={
            "provider_id": PROV_ID,
            "date_from": "2026-04-06",
            "date_to": "2026-04-06",
        })
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    @pytest.mark.asyncio
    async def test_get_single(self, client: AsyncClient):
        resp = await client.post("/api/v1/scheduling/appointments", json=_appt_body())
        appt_id = resp.json()["id"]
        resp = await client.get(f"/api/v1/scheduling/appointments/{appt_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == appt_id

    @pytest.mark.asyncio
    async def test_cancel_appointment(self, client: AsyncClient):
        resp = await client.post("/api/v1/scheduling/appointments", json=_appt_body())
        appt_id = resp.json()["id"]
        resp = await client.request(
            "DELETE", f"/api/v1/scheduling/appointments/{appt_id}",
            json={"reason": "Patient request"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
        assert resp.json()["cancel_reason"] == "Patient request"

    @pytest.mark.asyncio
    async def test_conflict_detection_409(self, client: AsyncClient):
        await client.post("/api/v1/scheduling/appointments", json=_appt_body(0))
        resp = await client.post("/api/v1/scheduling/appointments", json=_appt_body(0))
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_update_appointment(self, client: AsyncClient):
        resp = await client.post("/api/v1/scheduling/appointments", json=_appt_body())
        appt_id = resp.json()["id"]
        resp = await client.put(f"/api/v1/scheduling/appointments/{appt_id}", json={"notes": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["notes"] == "Updated"

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.get(f"/api/v1/scheduling/appointments/{fake_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Insurance authorization
# ---------------------------------------------------------------------------

class TestInsuranceAuth:
    @pytest.mark.asyncio
    async def test_warning_at_low_remaining(self, client: AsyncClient):
        resp = await client.get("/api/v1/scheduling/insurance-check", params={
            "patient_id": PAT_ID, "discipline": "PT",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["remaining"] == 3
        assert "3 authorized visits remaining" in data["warning"]

    @pytest.mark.asyncio
    async def test_create_includes_insurance_warning(self, client: AsyncClient):
        resp = await client.post("/api/v1/scheduling/appointments", json=_appt_body())
        data = resp.json()
        assert data["insurance_warning"] is not None
        assert "remaining" in data["insurance_warning"].lower()


# ---------------------------------------------------------------------------
# Check-in and Complete workflow
# ---------------------------------------------------------------------------

class TestCheckInComplete:
    @pytest.mark.asyncio
    async def test_check_in_creates_encounter(self, client: AsyncClient):
        resp = await client.post("/api/v1/scheduling/appointments", json=_appt_body())
        appt_id = resp.json()["id"]

        resp = await client.post(f"/api/v1/scheduling/appointments/{appt_id}/check-in")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "checked_in"
        assert data["encounter_id"] is not None

    @pytest.mark.asyncio
    async def test_complete_increments_visits(self, client: AsyncClient):
        # Create and check-in
        resp = await client.post("/api/v1/scheduling/appointments", json=_appt_body())
        appt_id = resp.json()["id"]
        await client.post(f"/api/v1/scheduling/appointments/{appt_id}/check-in")

        # Complete
        resp = await client.post(f"/api/v1/scheduling/appointments/{appt_id}/complete")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

        # Verify insurance visits_used incremented
        resp = await client.get("/api/v1/scheduling/insurance-check", params={
            "patient_id": PAT_ID, "discipline": "PT",
        })
        data = resp.json()
        assert data["visits_used"] == 18  # was 17, now 18
        assert data["remaining"] == 2

    @pytest.mark.asyncio
    async def test_cannot_check_in_cancelled(self, client: AsyncClient):
        resp = await client.post("/api/v1/scheduling/appointments", json=_appt_body())
        appt_id = resp.json()["id"]
        await client.delete(f"/api/v1/scheduling/appointments/{appt_id}")

        resp = await client.post(f"/api/v1/scheduling/appointments/{appt_id}/check-in")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_cannot_complete_without_checkin(self, client: AsyncClient):
        resp = await client.post("/api/v1/scheduling/appointments", json=_appt_body())
        appt_id = resp.json()["id"]
        resp = await client.post(f"/api/v1/scheduling/appointments/{appt_id}/complete")
        assert resp.status_code == 400
