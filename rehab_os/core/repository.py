"""CRUD repositories for Patient-Core models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.core.models import (
    AuditLog,
    BillingRecord,
    Encounter,
    Insurance,
    Patient,
    Provider,
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
