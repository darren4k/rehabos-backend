"""Core scheduling engine for RehabOS."""

import re
import uuid
from datetime import date, datetime, timedelta
from typing import Optional

from rehab_os.scheduling.models import (
    Appointment,
    AppointmentStatus,
    PreferredTimeRange,
    ScheduleRequest,
    ScheduleResult,
    TimeSlot,
)

# Preferred weekday patterns keyed by visits-per-week.
_WEEKDAY_PATTERNS: dict[int, list[list[int]]] = {
    1: [[0], [2], [4]],                        # Mon, Wed, or Fri
    2: [[1, 3], [0, 2], [0, 3], [1, 4]],       # Tu/Th preferred, then M/W …
    3: [[0, 2, 4], [0, 1, 3], [1, 3, 4]],      # M/W/F preferred
    4: [[0, 1, 2, 3], [0, 1, 3, 4]],
    5: [[0, 1, 2, 3, 4]],
}


def _parse_frequency(freq: str) -> int:
    """Parse a frequency string like '3x/week' and return visits per week."""
    m = re.match(r"(\d+)\s*x?\s*/?\s*w", freq, re.IGNORECASE)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse frequency: {freq!r}")


class SchedulingService:
    """Generates slots and auto-schedules appointment series."""

    def __init__(self, default_slot_minutes: int = 45) -> None:
        self.default_slot_minutes = default_slot_minutes

    # ------------------------------------------------------------------
    # Slot generation
    # ------------------------------------------------------------------

    def generate_provider_slots(
        self,
        provider_id: str,
        day: date,
        start_hour: int = 8,
        end_hour: int = 17,
        slot_minutes: int | None = None,
    ) -> list[TimeSlot]:
        """Generate all possible time slots for a provider on a single day."""
        slot_minutes = slot_minutes or self.default_slot_minutes
        slots: list[TimeSlot] = []
        current = datetime(day.year, day.month, day.day, start_hour, 0)
        end = datetime(day.year, day.month, day.day, end_hour, 0)
        delta = timedelta(minutes=slot_minutes)

        while current + delta <= end:
            slots.append(
                TimeSlot(
                    start_time=current,
                    end_time=current + delta,
                    provider_id=provider_id,
                )
            )
            current += delta
        return slots

    # ------------------------------------------------------------------
    # Available-slot finder
    # ------------------------------------------------------------------

    def find_available_slots(
        self,
        provider_id: str,
        date_range: tuple[date, date],
        existing_appointments: list[Appointment],
        availability_rules: list[dict] | None = None,
    ) -> list[TimeSlot]:
        """Return open slots across *date_range* after removing booked ones.

        When *availability_rules* is provided, each dict should have keys:
        ``day_of_week`` (0=Mon..6=Sun), ``start_hour``, ``end_hour``,
        ``slot_duration_minutes``, ``is_available``.  Days without a rule or
        with ``is_available=False`` are skipped.  When ``None``, the default
        behaviour (weekdays 8-17, 45-min slots) is used.
        """
        start_date, end_date = date_range
        booked_intervals: set[tuple[datetime, datetime]] = set()
        for appt in existing_appointments:
            if (
                appt.provider_id == provider_id
                and appt.status
                not in (AppointmentStatus.CANCELLED, AppointmentStatus.NO_SHOW)
            ):
                booked_intervals.add(
                    (appt.time_slot.start_time, appt.time_slot.end_time)
                )

        # Index rules by day_of_week if provided
        rules_by_day: dict[int, dict] | None = None
        if availability_rules is not None:
            rules_by_day = {r["day_of_week"]: r for r in availability_rules}

        available: list[TimeSlot] = []
        current_date = start_date
        while current_date <= end_date:
            wd = current_date.weekday()
            if rules_by_day is not None:
                rule = rules_by_day.get(wd)
                if rule is None or not rule.get("is_available", True):
                    current_date += timedelta(days=1)
                    continue
                day_start = rule.get("start_hour", 8)
                day_end = rule.get("end_hour", 17)
                slot_min = rule.get("slot_duration_minutes", self.default_slot_minutes)
            else:
                if wd >= 5:  # skip weekends
                    current_date += timedelta(days=1)
                    continue
                day_start = 8
                day_end = 17
                slot_min = self.default_slot_minutes

            for slot in self.generate_provider_slots(
                provider_id, current_date,
                start_hour=day_start, end_hour=day_end, slot_minutes=slot_min,
            ):
                if (slot.start_time, slot.end_time) not in booked_intervals:
                    available.append(slot)
            current_date += timedelta(days=1)
        return available

    # ------------------------------------------------------------------
    # Auto-scheduler
    # ------------------------------------------------------------------

    def auto_schedule(
        self,
        request: ScheduleRequest,
        available_slots: list[TimeSlot],
    ) -> ScheduleResult:
        """Auto-schedule a series of appointments based on *request*.

        Picks optimally-spaced slots (M/W/F for 3×, Tu/Th for 2×, etc.),
        respects preferred times, and avoids back-to-back same-patient slots.
        """
        visits_per_week = _parse_frequency(request.frequency)
        total_visits = visits_per_week * request.duration_weeks

        # Group available slots by (iso_year, iso_week, weekday)
        slots_by_week_day: dict[tuple[int, int, int], list[TimeSlot]] = {}
        for s in available_slots:
            iso = s.start_time.isocalendar()
            key = (iso[0], iso[1], s.start_time.weekday())
            slots_by_week_day.setdefault(key, []).append(s)

        # Determine ordered weeks present in the available slots
        weeks_present: list[tuple[int, int]] = sorted(
            {(k[0], k[1]) for k in slots_by_week_day}
        )

        # Pick the best weekday pattern
        patterns = _WEEKDAY_PATTERNS.get(visits_per_week, [list(range(visits_per_week))])

        scheduled: list[Appointment] = []
        conflicts: list[str] = []
        optimization_notes: list[str] = []
        last_slot_end: datetime | None = None

        weeks_used = 0
        for year, week in weeks_present:
            if weeks_used >= request.duration_weeks:
                break

            best_slots_for_week: list[TimeSlot] = []
            for pattern in patterns:
                candidate: list[TimeSlot] = []
                for wd in pattern:
                    day_slots = slots_by_week_day.get((year, week, wd), [])
                    picked = self._pick_slot(day_slots, request.preferred_times, last_slot_end)
                    if picked:
                        candidate.append(picked)
                if len(candidate) > len(best_slots_for_week):
                    best_slots_for_week = candidate

            if len(best_slots_for_week) < visits_per_week:
                conflicts.append(
                    f"Week {year}-W{week:02d}: only {len(best_slots_for_week)}/{visits_per_week} slots available"
                )

            for slot in best_slots_for_week:
                if len(scheduled) >= total_visits:
                    break
                appt = Appointment(
                    id=str(uuid.uuid4()),
                    patient_id=request.patient_id,
                    provider_id=slot.provider_id,
                    time_slot=slot,
                    discipline=request.discipline,
                    encounter_type="treatment",
                    status=AppointmentStatus.SCHEDULED,
                )
                scheduled.append(appt)
                last_slot_end = slot.end_time

            weeks_used += 1

        if len(scheduled) < total_visits:
            conflicts.append(
                f"Could only schedule {len(scheduled)}/{total_visits} appointments"
            )

        if visits_per_week == 3:
            optimization_notes.append("Preferred M/W/F pattern for 3×/week")
        elif visits_per_week == 2:
            optimization_notes.append("Preferred Tu/Th pattern for 2×/week")

        # next_available: first unused slot after scheduled ones
        used_starts = {a.time_slot.start_time for a in scheduled}
        next_avail = next(
            (s.start_time for s in available_slots if s.start_time not in used_starts),
            None,
        )

        return ScheduleResult(
            appointments=scheduled,
            conflicts=conflicts,
            optimization_notes=optimization_notes,
            next_available=next_avail,
        )

    # ------------------------------------------------------------------
    # Reschedule suggestions
    # ------------------------------------------------------------------

    def suggest_reschedule(
        self,
        cancelled_appointment: Appointment,
        available_slots: list[TimeSlot],
    ) -> list[TimeSlot]:
        """Suggest alternative slots for a cancelled appointment.

        Returns up to 5 slots closest in time to the original.
        """
        original_start = cancelled_appointment.time_slot.start_time
        ranked = sorted(available_slots, key=lambda s: abs((s.start_time - original_start).total_seconds()))
        return ranked[:5]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_slot(
        day_slots: list[TimeSlot],
        preferred: Optional[list[PreferredTimeRange]],
        last_end: Optional[datetime],
    ) -> Optional[TimeSlot]:
        """Pick the best slot from a single day's candidates."""
        if not day_slots:
            return None

        candidates = day_slots

        # Filter by preferred time ranges if given
        if preferred:
            filtered = [
                s for s in candidates
                if any(p.start_hour <= s.start_time.hour < p.end_hour for p in preferred)
            ]
            if filtered:
                candidates = filtered

        # Avoid back-to-back with same patient (same end == next start)
        if last_end is not None:
            non_adjacent = [s for s in candidates if s.start_time != last_end]
            if non_adjacent:
                candidates = non_adjacent

        # Return the earliest remaining candidate
        return min(candidates, key=lambda s: s.start_time)
