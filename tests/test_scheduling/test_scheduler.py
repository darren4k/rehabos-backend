"""Tests for the scheduling service."""

import uuid
from datetime import date, datetime, timedelta

import pytest

from rehab_os.scheduling.models import (
    Appointment,
    AppointmentStatus,
    PreferredTimeRange,
    ScheduleRequest,
    TimeSlot,
)
from rehab_os.scheduling.scheduler import SchedulingService
from rehab_os.scheduling.optimizer import RouteOptimizer


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def svc():
    return SchedulingService(default_slot_minutes=45)


@pytest.fixture
def optimizer():
    return RouteOptimizer()


def _make_appt(
    provider_id: str,
    start: datetime,
    minutes: int = 45,
    location: str | None = None,
    patient_id: str = "pat-1",
    status: AppointmentStatus = AppointmentStatus.SCHEDULED,
) -> Appointment:
    return Appointment(
        id=str(uuid.uuid4()),
        patient_id=patient_id,
        provider_id=provider_id,
        time_slot=TimeSlot(
            start_time=start,
            end_time=start + timedelta(minutes=minutes),
            provider_id=provider_id,
            location=location,
        ),
        discipline="PT",
        encounter_type="treatment",
        status=status,
    )


# --------------------------------------------------------- slot generation

class TestSlotGeneration:
    def test_generates_correct_count(self, svc: SchedulingService):
        # 8:00–17:00 = 9h = 540min / 45min = 12 slots
        slots = svc.generate_provider_slots("prov-1", date(2026, 3, 2))
        assert len(slots) == 12

    def test_slot_boundaries(self, svc: SchedulingService):
        slots = svc.generate_provider_slots("prov-1", date(2026, 3, 2))
        assert slots[0].start_time == datetime(2026, 3, 2, 8, 0)
        assert slots[0].end_time == datetime(2026, 3, 2, 8, 45)
        assert slots[-1].end_time <= datetime(2026, 3, 2, 17, 0)

    def test_custom_slot_duration(self, svc: SchedulingService):
        slots = svc.generate_provider_slots("prov-1", date(2026, 3, 2), slot_minutes=30)
        # 9h / 30min = 18 slots
        assert len(slots) == 18


# -------------------------------------------------------- available slots

class TestAvailableSlots:
    def test_excludes_booked(self, svc: SchedulingService):
        d = date(2026, 3, 2)  # Monday
        booked = [_make_appt("prov-1", datetime(2026, 3, 2, 8, 0))]
        available = svc.find_available_slots("prov-1", (d, d), booked)
        starts = {s.start_time for s in available}
        assert datetime(2026, 3, 2, 8, 0) not in starts
        assert datetime(2026, 3, 2, 8, 45) in starts

    def test_cancelled_not_excluded(self, svc: SchedulingService):
        d = date(2026, 3, 2)
        booked = [_make_appt("prov-1", datetime(2026, 3, 2, 8, 0), status=AppointmentStatus.CANCELLED)]
        available = svc.find_available_slots("prov-1", (d, d), booked)
        starts = {s.start_time for s in available}
        assert datetime(2026, 3, 2, 8, 0) in starts

    def test_skips_weekends(self, svc: SchedulingService):
        # 2026-03-07 is Saturday, 2026-03-08 is Sunday
        available = svc.find_available_slots("prov-1", (date(2026, 3, 7), date(2026, 3, 8)), [])
        assert len(available) == 0


# --------------------------------------------------------- auto-schedule

class TestAutoSchedule:
    def _available_slots(self, svc, provider="prov-1", weeks=8):
        start = date(2026, 3, 2)  # Monday
        end = start + timedelta(weeks=weeks)
        return svc.find_available_slots(provider, (start, end), [])

    def test_3x_week(self, svc: SchedulingService):
        req = ScheduleRequest(
            patient_id="pat-1", discipline="PT",
            frequency="3x/week", duration_weeks=4,
            provider_id="prov-1",
        )
        result = svc.auto_schedule(req, self._available_slots(svc))
        assert len(result.appointments) == 12
        # Check M/W/F pattern
        weekdays = [a.time_slot.start_time.weekday() for a in result.appointments[:3]]
        assert weekdays == [0, 2, 4]  # Mon, Wed, Fri

    def test_2x_week(self, svc: SchedulingService):
        req = ScheduleRequest(
            patient_id="pat-1", discipline="OT",
            frequency="2x/week", duration_weeks=3,
            provider_id="prov-1",
        )
        result = svc.auto_schedule(req, self._available_slots(svc))
        assert len(result.appointments) == 6
        weekdays = [a.time_slot.start_time.weekday() for a in result.appointments[:2]]
        assert weekdays == [1, 3]  # Tue, Thu

    def test_preferred_times(self, svc: SchedulingService):
        req = ScheduleRequest(
            patient_id="pat-1", discipline="PT",
            frequency="2x/week", duration_weeks=2,
            provider_id="prov-1",
            preferred_times=[PreferredTimeRange(start_hour=13, end_hour=17)],
        )
        result = svc.auto_schedule(req, self._available_slots(svc))
        for appt in result.appointments:
            assert appt.time_slot.start_time.hour >= 13

    def test_conflict_detection(self, svc: SchedulingService):
        # Only provide 1 week of slots for a 4-week request
        start = date(2026, 3, 2)
        end = start + timedelta(weeks=1)
        limited_slots = svc.find_available_slots("prov-1", (start, end), [])
        req = ScheduleRequest(
            patient_id="pat-1", discipline="PT",
            frequency="3x/week", duration_weeks=4,
            provider_id="prov-1",
        )
        result = svc.auto_schedule(req, limited_slots)
        assert len(result.conflicts) > 0
        assert len(result.appointments) < 12


# --------------------------------------------------- reschedule suggestions

class TestReschedule:
    def test_suggest_reschedule(self, svc: SchedulingService):
        cancelled = _make_appt("prov-1", datetime(2026, 3, 4, 10, 0))
        slots = svc.generate_provider_slots("prov-1", date(2026, 3, 4))
        suggestions = svc.suggest_reschedule(cancelled, slots)
        assert len(suggestions) <= 5
        # First suggestion should be closest to 10:00
        assert suggestions[0].start_time.hour in (9, 10)


# ------------------------------------------------------- route optimizer

class TestRouteOptimizer:
    def test_optimize_reorders(self, optimizer: RouteOptimizer):
        appts = [
            _make_appt("prov-1", datetime(2026, 3, 2, 8, 0), location="123 Main St 90210"),
            _make_appt("prov-1", datetime(2026, 3, 2, 9, 0), location="456 Elm St 90220"),
            _make_appt("prov-1", datetime(2026, 3, 2, 10, 0), location="789 Oak St 90211"),
        ]
        optimized = optimizer.optimize_daily_route(appts, provider_home_address="Home 90210")
        # Starting from 90210 → 90210 first, then 90211 (adjacent), then 90220
        zips = [appt.time_slot.location.split()[-1] for appt in optimized]
        assert zips[0] == "90210"
        assert zips[1] == "90211"

    def test_travel_time_same_zip(self, optimizer: RouteOptimizer):
        assert optimizer.estimate_travel_time("A 90210", "B 90210") == 10

    def test_travel_time_adjacent(self, optimizer: RouteOptimizer):
        assert optimizer.estimate_travel_time("A 90210", "B 90212") == 20

    def test_travel_time_far(self, optimizer: RouteOptimizer):
        assert optimizer.estimate_travel_time("A 90210", "B 10001") == 30

    def test_flag_infeasible(self, optimizer: RouteOptimizer):
        appts = [
            _make_appt("prov-1", datetime(2026, 3, 2, 8, 0), location="A 90210"),
            _make_appt("prov-1", datetime(2026, 3, 2, 9, 0), location="B 10001"),
        ]
        flags = optimizer.flag_infeasible_schedule(appts, max_travel_minutes=25)
        assert len(flags) == 1

    def test_no_flag_when_feasible(self, optimizer: RouteOptimizer):
        appts = [
            _make_appt("prov-1", datetime(2026, 3, 2, 8, 0), location="A 90210"),
            _make_appt("prov-1", datetime(2026, 3, 2, 9, 0), location="B 90210"),
        ]
        flags = optimizer.flag_infeasible_schedule(appts, max_travel_minutes=45)
        assert len(flags) == 0


# ------------------------------------------------------- API endpoints

class TestSchedulingAPI:
    @pytest.fixture(autouse=True)
    def _clear_store(self):
        from rehab_os.api.routes.scheduling import _appointments
        _appointments.clear()
        yield
        _appointments.clear()

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from rehab_os.api.routes.scheduling import router
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return TestClient(app)

    def test_book_and_list(self, client):
        appt = _make_appt("prov-1", datetime(2026, 3, 2, 8, 0))
        resp = client.post("/api/v1/scheduling/appointments", json=appt.model_dump(mode="json"))
        assert resp.status_code == 201

        resp = client.get("/api/v1/scheduling/appointments", params={"patient_id": "pat-1"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_update_appointment(self, client):
        appt = _make_appt("prov-1", datetime(2026, 3, 2, 8, 0))
        client.post("/api/v1/scheduling/appointments", json=appt.model_dump(mode="json"))

        resp = client.put(
            f"/api/v1/scheduling/appointments/{appt.id}",
            json={"status": "cancelled"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_update_not_found(self, client):
        resp = client.put("/api/v1/scheduling/appointments/nope", json={"status": "cancelled"})
        assert resp.status_code == 404

    def test_provider_availability(self, client):
        resp = client.get(
            "/api/v1/scheduling/providers/prov-1/availability",
            params={"start_date": "2026-03-02", "end_date": "2026-03-02"},
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 12  # full day of 45-min slots

    def test_auto_schedule_endpoint(self, client):
        resp = client.post("/api/v1/scheduling/auto-schedule", json={
            "patient_id": "pat-1",
            "discipline": "PT",
            "frequency": "2x/week",
            "duration_weeks": 2,
            "provider_id": "prov-1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["appointments"]) == 4

    def test_optimize_route_endpoint(self, client):
        appts = [
            _make_appt("prov-1", datetime(2026, 3, 2, 8, 0), location="A 90210"),
            _make_appt("prov-1", datetime(2026, 3, 2, 9, 0), location="B 90220"),
        ]
        for a in appts:
            client.post("/api/v1/scheduling/appointments", json=a.model_dump(mode="json"))

        resp = client.post("/api/v1/scheduling/optimize-route", json={
            "appointment_ids": [a.id for a in appts],
        })
        assert resp.status_code == 200
        assert len(resp.json()) == 2
