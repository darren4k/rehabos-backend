"""SQLAlchemy 2.0 async models for the Patient-Core relational schema."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=_new_uuid)
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

    insurance_records: Mapped[list[Insurance]] = relationship(back_populates="patient", lazy="selectin")
    encounters: Mapped[list[Encounter]] = relationship(back_populates="patient", lazy="selectin")
    referrals: Mapped[list[Referral]] = relationship(back_populates="patient", lazy="selectin")

    __table_args__ = (
        Index("ix_patients_last_name", "last_name"),
        Index("ix_patients_dob", "dob"),
        Index("ix_patients_active", "active"),
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
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    credentials: Mapped[str | None] = mapped_column(String(50))
    npi: Mapped[str | None] = mapped_column(String(10), unique=True)
    discipline: Mapped[str] = mapped_column(String(10), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    encounters: Mapped[list[Encounter]] = relationship(back_populates="provider", lazy="selectin")


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
