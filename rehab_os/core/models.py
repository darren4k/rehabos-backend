"""SQLAlchemy 2.0 async models for the Patient-Core relational schema."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.types import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

import enum


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


class ProviderRole(str, enum.Enum):
    """Role within an organization."""
    owner = "owner"
    admin = "admin"
    therapist = "therapist"
    assistant = "assistant"


class Base(DeclarativeBase):
    pass


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str | None] = mapped_column(String(100), unique=True)
    phone: Mapped[str | None] = mapped_column(String(20))
    address: Mapped[str | None] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    providers: Mapped[list[Provider]] = relationship(back_populates="organization", lazy="selectin")
    patients: Mapped[list[Patient]] = relationship(back_populates="organization", lazy="selectin")


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    organization_id: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"))
    primary_therapist_id: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("providers.id", ondelete="SET NULL"))
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    dob: Mapped[date] = mapped_column(Date, nullable=False)
    sex: Mapped[str] = mapped_column(String(10), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20))
    email: Mapped[str | None] = mapped_column(String(255))
    address: Mapped[str | None] = mapped_column(Text)
    emergency_contact_name: Mapped[str | None] = mapped_column(String(200))
    emergency_contact_phone: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    organization: Mapped[Organization | None] = relationship(back_populates="patients")
    primary_therapist: Mapped[Provider | None] = relationship(foreign_keys=[primary_therapist_id])
    insurance_records: Mapped[list[Insurance]] = relationship(back_populates="patient", lazy="selectin")
    encounters: Mapped[list[Encounter]] = relationship(back_populates="patient", lazy="selectin")
    referrals: Mapped[list[Referral]] = relationship(back_populates="patient", lazy="selectin")
    clinical_notes: Mapped[list[ClinicalNote]] = relationship(back_populates="patient", lazy="selectin")
    appointments: Mapped[list[AppointmentDB]] = relationship(back_populates="patient", lazy="selectin")

    __table_args__ = (
        Index("ix_patients_last_name", "last_name"),
        Index("ix_patients_dob", "dob"),
        Index("ix_patients_active", "active"),
        Index("ix_patients_organization_id", "organization_id"),
        Index("ix_patients_primary_therapist_id", "primary_therapist_id"),
    )


class Insurance(Base):
    __tablename__ = "insurance"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    patient_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    payer_name: Mapped[str] = mapped_column(String(200), nullable=False)
    member_id: Mapped[str] = mapped_column(String(100), nullable=False)
    group_id: Mapped[str | None] = mapped_column(String(100))
    auth_number: Mapped[str | None] = mapped_column(String(100))
    authorized_visits: Mapped[int | None] = mapped_column(Integer)
    visits_used: Mapped[int] = mapped_column(Integer, default=0)
    frequency: Mapped[str | None] = mapped_column(String(50))
    duration_weeks: Mapped[int | None] = mapped_column(Integer)
    expiry_date: Mapped[date | None] = mapped_column(Date)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=True)

    patient: Mapped[Patient] = relationship(back_populates="insurance_records")

    __table_args__ = (
        Index("ix_insurance_patient_id", "patient_id"),
    )


class Provider(Base):
    __tablename__ = "providers"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    organization_id: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"))
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    credentials: Mapped[str | None] = mapped_column(String(50))
    npi: Mapped[str | None] = mapped_column(String(10), unique=True)
    discipline: Mapped[str] = mapped_column(String(10), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="therapist")
    email: Mapped[str | None] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False)

    organization: Mapped[Organization | None] = relationship(back_populates="providers")
    encounters: Mapped[list[Encounter]] = relationship(back_populates="provider", lazy="selectin")
    appointments: Mapped[list[AppointmentDB]] = relationship(back_populates="provider", lazy="selectin")
    availability_rules: Mapped[list[ProviderAvailability]] = relationship(back_populates="provider", lazy="selectin")

    __table_args__ = (
        Index("ix_providers_organization_id", "organization_id"),
        Index("ix_providers_email", "email", unique=True),
    )


class Encounter(Base):
    __tablename__ = "encounters"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    patient_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    provider_id: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("providers.id", ondelete="SET NULL"))
    encounter_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    discipline: Mapped[str] = mapped_column(String(10), nullable=False)
    setting: Mapped[str | None] = mapped_column(String(50))
    encounter_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="scheduled")
    soap_note_id: Mapped[str | None] = mapped_column(String(255))
    rehab_os_consultation_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    patient: Mapped[Patient] = relationship(back_populates="encounters")
    provider: Mapped[Provider | None] = relationship(back_populates="encounters")
    billing_records: Mapped[list[BillingRecord]] = relationship(back_populates="encounter", lazy="selectin")
    appointment: Mapped[AppointmentDB | None] = relationship(back_populates="encounter", uselist=False)

    __table_args__ = (
        Index("ix_encounters_patient_id", "patient_id"),
        Index("ix_encounters_date", "encounter_date"),
        Index("ix_encounters_status", "status"),
    )


class Referral(Base):
    __tablename__ = "referrals"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    patient_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    referring_provider_name: Mapped[str | None] = mapped_column(String(200))
    referring_provider_npi: Mapped[str | None] = mapped_column(String(10))
    referral_date: Mapped[date | None] = mapped_column(Date)
    received_date: Mapped[date | None] = mapped_column(Date)
    diagnosis_codes: Mapped[dict | None] = mapped_column(JSON)
    raw_document_path: Mapped[str | None] = mapped_column(Text)
    intake_result_json: Mapped[dict | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="pending")

    patient: Mapped[Patient] = relationship(back_populates="referrals")

    __table_args__ = (
        Index("ix_referrals_patient_id", "patient_id"),
        Index("ix_referrals_status", "status"),
    )


class BillingRecord(Base):
    __tablename__ = "billing_records"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    encounter_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("encounters.id", ondelete="CASCADE"), nullable=False)
    cpt_code: Mapped[str] = mapped_column(String(10), nullable=False)
    units: Mapped[int] = mapped_column(Integer, default=1)
    minutes: Mapped[int | None] = mapped_column(Integer)
    modifier: Mapped[str | None] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(20), default="unbilled")
    payer_response: Mapped[str | None] = mapped_column(Text)

    encounter: Mapped[Encounter] = relationship(back_populates="billing_records")

    __table_args__ = (
        Index("ix_billing_encounter_id", "encounter_id"),
        Index("ix_billing_status", "status"),
    )


class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    patient_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    therapist_id: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("providers.id", ondelete="SET NULL"))
    organization_id: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"))
    note_type: Mapped[str] = mapped_column(String(50), nullable=False)  # evaluation, daily_note, progress_note, recertification, discharge_summary
    note_date: Mapped[date] = mapped_column(Date, nullable=False)
    discipline: Mapped[str] = mapped_column(String(10), default="pt")
    therapist_name: Mapped[str | None] = mapped_column(String(200))

    # Content
    soap_subjective: Mapped[str | None] = mapped_column(Text)
    soap_objective: Mapped[str | None] = mapped_column(Text)
    soap_assessment: Mapped[str | None] = mapped_column(Text)
    soap_plan: Mapped[str | None] = mapped_column(Text)

    # Structured clinical data (JSON)
    structured_data: Mapped[dict | None] = mapped_column(JSON)  # ROM, MMT, tests, functional deficits, vitals, billing

    # Metadata
    transcript: Mapped[str | None] = mapped_column(Text)
    compliance_score: Mapped[int | None] = mapped_column(Integer)
    compliance_warnings: Mapped[list | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="final")  # draft, final

    # EMR sync
    emr_synced: Mapped[bool] = mapped_column(Boolean, default=False)
    emr_note_id: Mapped[str | None] = mapped_column(String(255))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    patient: Mapped[Patient] = relationship(back_populates="clinical_notes")
    therapist: Mapped[Provider | None] = relationship(foreign_keys=[therapist_id])

    __table_args__ = (
        Index("ix_clinical_notes_patient_id", "patient_id"),
        Index("ix_clinical_notes_note_date", "note_date"),
        Index("ix_clinical_notes_note_type", "note_type"),
        Index("ix_clinical_notes_therapist_id", "therapist_id"),
        Index("ix_clinical_notes_organization_id", "organization_id"),
    )


class AppointmentDB(Base):
    __tablename__ = "appointments"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    patient_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    provider_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("providers.id", ondelete="SET NULL"), nullable=False)
    organization_id: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"))
    encounter_id: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("encounters.id", ondelete="SET NULL"))
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    location: Mapped[str | None] = mapped_column(String(500))
    discipline: Mapped[str] = mapped_column(String(10), nullable=False)
    encounter_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="scheduled")
    cancel_reason: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    is_auto_scheduled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    patient: Mapped[Patient] = relationship(back_populates="appointments")
    provider: Mapped[Provider] = relationship(back_populates="appointments")
    encounter: Mapped[Encounter | None] = relationship(back_populates="appointment")

    __table_args__ = (
        Index("ix_appointments_patient_id", "patient_id"),
        Index("ix_appointments_provider_id", "provider_id"),
        Index("ix_appointments_start_time", "start_time"),
        Index("ix_appointments_status", "status"),
        Index("ix_appointments_provider_start", "provider_id", "start_time"),
        Index("ix_appointments_organization_id", "organization_id"),
    )


class ProviderAvailability(Base):
    __tablename__ = "provider_availability"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    provider_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("providers.id", ondelete="CASCADE"), nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)  # 0=Mon..6=Sun
    start_hour: Mapped[int] = mapped_column(Integer, nullable=False)
    end_hour: Mapped[int] = mapped_column(Integer, nullable=False)
    slot_duration_minutes: Mapped[int] = mapped_column(Integer, default=45)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    effective_date: Mapped[date | None] = mapped_column(Date)
    expiry_date: Mapped[date | None] = mapped_column(Date)

    provider: Mapped[Provider] = relationship(back_populates="availability_rules")

    __table_args__ = (
        Index("ix_provider_availability_provider", "provider_id"),
        Index("ix_provider_availability_day", "provider_id", "day_of_week"),
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    user_id: Mapped[str | None] = mapped_column(String(255))
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(255), nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    ip_address: Mapped[str | None] = mapped_column(String(45))

    __table_args__ = (
        Index("ix_audit_resource", "resource_type", "resource_id"),
        Index("ix_audit_timestamp", "timestamp"),
    )
