"""Pydantic models for the scheduling service."""

from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AppointmentStatus(str, Enum):
    """Appointment lifecycle statuses."""

    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CHECKED_IN = "checked_in"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class TimeSlot(BaseModel):
    """A single schedulable time window."""

    start_time: datetime
    end_time: datetime
    provider_id: str
    location: Optional[str] = None


class PreferredTimeRange(BaseModel):
    """Patient-preferred time window (hour-of-day based)."""

    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=0, le=23)


class Appointment(BaseModel):
    """A booked appointment."""

    id: str
    patient_id: str
    provider_id: str
    time_slot: TimeSlot
    discipline: str
    encounter_type: str
    status: AppointmentStatus = AppointmentStatus.SCHEDULED
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScheduleRequest(BaseModel):
    """Request to auto-schedule a series of appointments."""

    patient_id: str
    discipline: str
    frequency: str = Field(description="e.g. '3x/week', '2x/week'")
    duration_weeks: int
    preferred_times: Optional[list[PreferredTimeRange]] = None
    provider_id: Optional[str] = None
    location: Optional[str] = None


class ScheduleResult(BaseModel):
    """Result of an auto-scheduling operation."""

    appointments: list[Appointment] = []
    conflicts: list[str] = []
    optimization_notes: list[str] = []
    next_available: Optional[datetime] = None


class ProviderSchedule(BaseModel):
    """A provider's schedule for a single day."""

    provider_id: str
    provider_name: str
    date: date
    slots: list[TimeSlot] = []
    booked: list[Appointment] = []
