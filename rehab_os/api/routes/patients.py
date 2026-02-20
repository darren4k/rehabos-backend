"""Patient CRUD API routes."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.api.dependencies import get_current_user
from rehab_os.core.database import get_db
from rehab_os.core.models import Provider
from rehab_os.core.repository import (
    AuditRepository,
    EncounterRepository,
    InsuranceRepository,
    PatientRepository,
)
from rehab_os.core.schemas import (
    EncounterCreate,
    EncounterRead,
    InsuranceRead,
    PatientCreate,
    PatientRead,
    PatientUpdate,
)

router = APIRouter(prefix="/patients", tags=["patients"])


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.get("", response_model=list[PatientRead])
async def list_patients(
    search: Optional[str] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = PatientRepository(db)
    if search:
        return await repo.search_by_name(search, limit=limit)
    return await repo.list(offset=offset, limit=limit)


@router.get("/{patient_id}", response_model=PatientRead)
async def get_patient(patient_id: uuid.UUID, current_user: Provider = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    repo = PatientRepository(db)
    patient = await repo.get_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.post("", response_model=PatientRead, status_code=201)
async def create_patient(
    data: PatientCreate,
    request: Request,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = PatientRepository(db)
    patient = await repo.create(**data.model_dump())
    await AuditRepository(db).log_action(
        action="create",
        resource_type="patient",
        resource_id=str(patient.id),
        ip_address=_client_ip(request),
    )
    return patient


@router.put("/{patient_id}", response_model=PatientRead)
async def update_patient(
    patient_id: uuid.UUID,
    data: PatientUpdate,
    request: Request,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = PatientRepository(db)
    patient = await repo.update(patient_id, **data.model_dump(exclude_unset=True))
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    await AuditRepository(db).log_action(
        action="update",
        resource_type="patient",
        resource_id=str(patient_id),
        details=data.model_dump(exclude_unset=True),
        ip_address=_client_ip(request),
    )
    return patient


@router.post("/{patient_id}/encounters", response_model=EncounterRead, status_code=201)
async def create_encounter(
    patient_id: uuid.UUID,
    data: EncounterCreate,
    request: Request,
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Verify patient exists
    if not await PatientRepository(db).get_by_id(patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    enc = await EncounterRepository(db).create(patient_id=patient_id, **data.model_dump(exclude={"patient_id"}))
    await AuditRepository(db).log_action(
        action="create",
        resource_type="encounter",
        resource_id=str(enc.id),
        details={"patient_id": str(patient_id)},
        ip_address=_client_ip(request),
    )
    return enc


@router.get("/{patient_id}/encounters", response_model=list[EncounterRead])
async def list_encounters(
    patient_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=200),
    current_user: Provider = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await EncounterRepository(db).list_by_patient(patient_id, limit=limit)


@router.get("/{patient_id}/insurance", response_model=list[InsuranceRead])
async def get_insurance(patient_id: uuid.UUID, current_user: Provider = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    return await InsuranceRepository(db).get_by_patient(patient_id)
