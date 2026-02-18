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


# ------------------------------------------------ availability rules

class TestSchedulerWithAvailability:
    def test_availability_rules_respected(self, svc: SchedulingService):
        """Provider only works Mon/Wed/Fri 9-15, 30-min slots."""
        rules = [
            {"day_of_week": 0, "start_hour": 9, "end_hour": 15, "slot_duration_minutes": 30, "is_available": True},
            {"day_of_week": 1, "start_hour": 9, "end_hour": 15, "slot_duration_minutes": 30, "is_available": False},
            {"day_of_week": 2, "start_hour": 9, "end_hour": 15, "slot_duration_minutes": 30, "is_available": True},
            {"day_of_week": 3, "start_hour": 9, "end_hour": 15, "slot_duration_minutes": 30, "is_available": False},
            {"day_of_week": 4, "start_hour": 9, "end_hour": 15, "slot_duration_minutes": 30, "is_available": True},
        ]
        d = date(2026, 3, 2)  # Monday
        available = svc.find_available_slots("prov-1", (d, d + timedelta(days=4)), [], availability_rules=rules)
        weekdays = {s.start_time.weekday() for s in available}
        assert weekdays == {0, 2, 4}  # Mon, Wed, Fri only
        # 6h / 30min = 12 slots per day * 3 days = 36
        assert len(available) == 36

    def test_unavailable_days_skipped(self, svc: SchedulingService):
        """All days marked unavailable → no slots."""
        rules = [
            {"day_of_week": i, "is_available": False}
            for i in range(7)
        ]
        d = date(2026, 3, 2)
        available = svc.find_available_slots("prov-1", (d, d + timedelta(days=6)), [], availability_rules=rules)
        assert len(available) == 0

    def test_no_rules_uses_default(self, svc: SchedulingService):
        """When availability_rules=None, default weekday 8-17 behaviour."""
        d = date(2026, 3, 2)  # Monday
        available = svc.find_available_slots("prov-1", (d, d), [])
        assert len(available) == 12  # 9h / 45min = 12 slots

    def test_custom_hours_per_day(self, svc: SchedulingService):
        """Different hours per day are respected."""
        rules = [
            {"day_of_week": 0, "start_hour": 8, "end_hour": 12, "slot_duration_minutes": 45, "is_available": True},
            {"day_of_week": 1, "start_hour": 13, "end_hour": 17, "slot_duration_minutes": 45, "is_available": True},
        ]
        d = date(2026, 3, 2)  # Monday
        available = svc.find_available_slots("prov-1", (d, d + timedelta(days=1)), [], availability_rules=rules)
        mon_slots = [s for s in available if s.start_time.weekday() == 0]
        tue_slots = [s for s in available if s.start_time.weekday() == 1]
        assert all(s.start_time.hour >= 8 and s.start_time.hour < 12 for s in mon_slots)
        assert all(s.start_time.hour >= 13 and s.start_time.hour < 17 for s in tue_slots)
