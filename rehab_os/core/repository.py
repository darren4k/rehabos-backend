"""CRUD repositories for Patient-Core models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import and_, delete, func, select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.core.models import (
    AppointmentDB,
    AuditLog,
    BillingRecord,
    ClinicalNote,
    Encounter,
    Insurance,
    Patient,
    Provider,
    ProviderAvailability,
    Referral,
)


class PatientRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Patient:
        patient = Patient(**kwargs)
        self.session.add(patient)
        await self.session.flush()
        return patient

    async def get_by_id(self, patient_id: uuid.UUID) -> Optional[Patient]:
        return await self.session.get(Patient, patient_id)

    async def list(self, offset: int = 0, limit: int = 50, active_only: bool = True) -> Sequence[Patient]:
        stmt = select(Patient)
        if active_only:
            stmt = stmt.where(Patient.active.is_(True))
        stmt = stmt.order_by(Patient.last_name, Patient.first_name).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update(self, patient_id: uuid.UUID, **kwargs) -> Optional[Patient]:
        patient = await self.get_by_id(patient_id)
        if not patient:
            return None
        for k, v in kwargs.items():
            if v is not None:
                setattr(patient, k, v)
        patient.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return patient

    async def search_by_name(self, query: str, limit: int = 20) -> Sequence[Patient]:
        pattern = f"%{query}%"
        stmt = (
            select(Patient)
            .where(
                Patient.active.is_(True),
                or_(
                    Patient.first_name.ilike(pattern),
                    Patient.last_name.ilike(pattern),
                ),
            )
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def deactivate(self, patient_id: uuid.UUID) -> Optional[Patient]:
        return await self.update(patient_id, active=False)


class EncounterRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Encounter:
        enc = Encounter(**kwargs)
        self.session.add(enc)
        await self.session.flush()
        return enc

    async def get_by_id(self, encounter_id: uuid.UUID) -> Optional[Encounter]:
        return await self.session.get(Encounter, encounter_id)

    async def list_by_patient(self, patient_id: uuid.UUID, limit: int = 50) -> Sequence[Encounter]:
        stmt = (
            select(Encounter)
            .where(Encounter.patient_id == patient_id)
            .order_by(Encounter.encounter_date.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_status(self, encounter_id: uuid.UUID, status: str) -> Optional[Encounter]:
        enc = await self.get_by_id(encounter_id)
        if enc:
            enc.status = status
            await self.session.flush()
        return enc

    async def get_recent(self, limit: int = 20) -> Sequence[Encounter]:
        stmt = select(Encounter).order_by(Encounter.encounter_date.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()


class ReferralRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Referral:
        ref = Referral(**kwargs)
        self.session.add(ref)
        await self.session.flush()
        return ref

    async def get_by_id(self, referral_id: uuid.UUID) -> Optional[Referral]:
        return await self.session.get(Referral, referral_id)

    async def list_pending(self, limit: int = 50) -> Sequence[Referral]:
        stmt = select(Referral).where(Referral.status == "pending").limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_status(self, referral_id: uuid.UUID, status: str) -> Optional[Referral]:
        ref = await self.get_by_id(referral_id)
        if ref:
            ref.status = status
            await self.session.flush()
        return ref


class InsuranceRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Insurance:
        ins = Insurance(**kwargs)
        self.session.add(ins)
        await self.session.flush()
        return ins

    async def get_by_patient(self, patient_id: uuid.UUID) -> Sequence[Insurance]:
        stmt = select(Insurance).where(Insurance.patient_id == patient_id)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_visits_used(self, insurance_id: uuid.UUID, visits_used: int) -> Optional[Insurance]:
        ins = await self.session.get(Insurance, insurance_id)
        if ins:
            ins.visits_used = visits_used
            await self.session.flush()
        return ins


class BillingRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> BillingRecord:
        rec = BillingRecord(**kwargs)
        self.session.add(rec)
        await self.session.flush()
        return rec

    async def get_by_encounter(self, encounter_id: uuid.UUID) -> Sequence[BillingRecord]:
        stmt = select(BillingRecord).where(BillingRecord.encounter_id == encounter_id)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_unbilled(self, limit: int = 100) -> Sequence[BillingRecord]:
        stmt = select(BillingRecord).where(BillingRecord.status == "unbilled").limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()


class ClinicalNoteRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> ClinicalNote:
        note = ClinicalNote(**kwargs)
        self.session.add(note)
        await self.session.flush()
        return note

    async def get_by_id(self, note_id: uuid.UUID) -> Optional[ClinicalNote]:
        return await self.session.get(ClinicalNote, note_id)

    async def update(self, note_id: uuid.UUID, **kwargs) -> Optional[ClinicalNote]:
        note = await self.get_by_id(note_id)
        if not note:
            return None
        for k, v in kwargs.items():
            if v is not None:
                setattr(note, k, v)
        note.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return note

    async def list_by_patient(
        self, patient_id: uuid.UUID, note_type: Optional[str] = None, limit: int = 50
    ) -> Sequence[ClinicalNote]:
        stmt = select(ClinicalNote).where(ClinicalNote.patient_id == patient_id)
        if note_type:
            stmt = stmt.where(ClinicalNote.note_type == note_type)
        stmt = stmt.order_by(ClinicalNote.note_date.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_latest_by_type(self, patient_id: uuid.UUID, note_type: str) -> Optional[ClinicalNote]:
        stmt = (
            select(ClinicalNote)
            .where(ClinicalNote.patient_id == patient_id, ClinicalNote.note_type == note_type)
            .order_by(ClinicalNote.note_date.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def search_notes(self, patient_id: uuid.UUID, query: str, limit: int = 20) -> Sequence[ClinicalNote]:
        pattern = f"%{query}%"
        stmt = (
            select(ClinicalNote)
            .where(
                ClinicalNote.patient_id == patient_id,
                or_(
                    ClinicalNote.soap_subjective.ilike(pattern),
                    ClinicalNote.soap_objective.ilike(pattern),
                    ClinicalNote.soap_assessment.ilike(pattern),
                    ClinicalNote.soap_plan.ilike(pattern),
                ),
            )
            .order_by(ClinicalNote.note_date.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()


class AuditRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def log_action(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: Optional[str] = None,
        details: Optional[dict] = None,
        ip_address: Optional[str] = None,
    ) -> AuditLog:
        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id),
            details=details,
            ip_address=ip_address,
        )
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def get_by_resource(self, resource_type: str, resource_id: str, limit: int = 50) -> Sequence[AuditLog]:
        stmt = (
            select(AuditLog)
            .where(AuditLog.resource_type == resource_type, AuditLog.resource_id == resource_id)
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()


class AppointmentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> AppointmentDB:
        appt = AppointmentDB(**kwargs)
        self.session.add(appt)
        await self.session.flush()
        return appt

    async def get_by_id(self, appointment_id: uuid.UUID) -> Optional[AppointmentDB]:
        return await self.session.get(AppointmentDB, appointment_id)

    async def list_by_provider_date_range(
        self,
        provider_id: uuid.UUID,
        start: datetime,
        end: datetime,
        exclude_statuses: Optional[list[str]] = None,
    ) -> Sequence[AppointmentDB]:
        stmt = select(AppointmentDB).where(
            AppointmentDB.provider_id == provider_id,
            AppointmentDB.start_time >= start,
            AppointmentDB.start_time <= end,
        )
        if exclude_statuses:
            stmt = stmt.where(AppointmentDB.status.notin_(exclude_statuses))
        stmt = stmt.order_by(AppointmentDB.start_time)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_by_patient(self, patient_id: uuid.UUID, limit: int = 50) -> Sequence[AppointmentDB]:
        stmt = (
            select(AppointmentDB)
            .where(AppointmentDB.patient_id == patient_id)
            .order_by(AppointmentDB.start_time.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_by_date(
        self, target_date: datetime, provider_id: Optional[uuid.UUID] = None
    ) -> Sequence[AppointmentDB]:
        day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        stmt = select(AppointmentDB).where(
            AppointmentDB.start_time >= day_start,
            AppointmentDB.start_time <= day_end,
        )
        if provider_id:
            stmt = stmt.where(AppointmentDB.provider_id == provider_id)
        stmt = stmt.order_by(AppointmentDB.start_time)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update(self, appointment_id: uuid.UUID, **kwargs) -> Optional[AppointmentDB]:
        appt = await self.get_by_id(appointment_id)
        if not appt:
            return None
        for k, v in kwargs.items():
            if v is not None:
                setattr(appt, k, v)
        appt.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return appt

    async def cancel(self, appointment_id: uuid.UUID, reason: Optional[str] = None) -> Optional[AppointmentDB]:
        appt = await self.get_by_id(appointment_id)
        if not appt:
            return None
        appt.status = "cancelled"
        appt.cancel_reason = reason
        appt.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return appt

    async def check_conflict(
        self,
        provider_id: uuid.UUID,
        start: datetime,
        end: datetime,
        exclude_id: Optional[uuid.UUID] = None,
    ) -> bool:
        stmt = select(func.count()).select_from(AppointmentDB).where(
            AppointmentDB.provider_id == provider_id,
            AppointmentDB.status.notin_(["cancelled", "no_show"]),
            AppointmentDB.start_time < end,
            AppointmentDB.end_time > start,
        )
        if exclude_id:
            stmt = stmt.where(AppointmentDB.id != exclude_id)
        result = await self.session.execute(stmt)
        return result.scalar_one() > 0

    async def count_by_patient_discipline(self, patient_id: uuid.UUID, discipline: str) -> int:
        stmt = select(func.count()).select_from(AppointmentDB).where(
            AppointmentDB.patient_id == patient_id,
            AppointmentDB.discipline == discipline,
            AppointmentDB.status.in_(["completed", "checked_in"]),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()


class ProviderAvailabilityRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> ProviderAvailability:
        avail = ProviderAvailability(**kwargs)
        self.session.add(avail)
        await self.session.flush()
        return avail

    async def get_by_provider(self, provider_id: uuid.UUID) -> Sequence[ProviderAvailability]:
        stmt = (
            select(ProviderAvailability)
            .where(ProviderAvailability.provider_id == provider_id)
            .order_by(ProviderAvailability.day_of_week)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_for_day(
        self, provider_id: uuid.UUID, day_of_week: int, target_date: Optional[datetime] = None
    ) -> Sequence[ProviderAvailability]:
        stmt = select(ProviderAvailability).where(
            ProviderAvailability.provider_id == provider_id,
            ProviderAvailability.day_of_week == day_of_week,
        )
        if target_date:
            d = target_date.date() if isinstance(target_date, datetime) else target_date
            stmt = stmt.where(
                or_(ProviderAvailability.effective_date.is_(None), ProviderAvailability.effective_date <= d),
                or_(ProviderAvailability.expiry_date.is_(None), ProviderAvailability.expiry_date >= d),
            )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def upsert(self, provider_id: uuid.UUID, day_of_week: int, **kwargs) -> ProviderAvailability:
        stmt = select(ProviderAvailability).where(
            ProviderAvailability.provider_id == provider_id,
            ProviderAvailability.day_of_week == day_of_week,
        )
        if "effective_date" in kwargs and kwargs["effective_date"] is not None:
            stmt = stmt.where(ProviderAvailability.effective_date == kwargs["effective_date"])
        else:
            stmt = stmt.where(ProviderAvailability.effective_date.is_(None))
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing:
            for k, v in kwargs.items():
                setattr(existing, k, v)
            await self.session.flush()
            return existing
        return await self.create(provider_id=provider_id, day_of_week=day_of_week, **kwargs)

    async def delete_by_id(self, avail_id: uuid.UUID) -> bool:
        avail = await self.session.get(ProviderAvailability, avail_id)
        if not avail:
            return False
        await self.session.delete(avail)
        await self.session.flush()
        return True
