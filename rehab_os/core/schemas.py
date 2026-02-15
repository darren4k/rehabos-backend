"""Pydantic schemas for Patient-Core API I/O."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# --- Patient ---

class PatientCreate(BaseModel):
    first_name: str
    last_name: str
    dob: date
    sex: str
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None


class PatientUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    active: Optional[bool] = None


class PatientRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    first_name: str
    last_name: str
    dob: date
    sex: str
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    active: bool


# --- Insurance ---

# --- Clinical Note ---

class ClinicalNoteCreate(BaseModel):
    patient_id: uuid.UUID
    note_type: str  # evaluation, daily_note, progress_note, recertification, discharge_summary
    note_date: date
    discipline: str = "pt"
    therapist_name: Optional[str] = None
    soap_subjective: Optional[str] = None
    soap_objective: Optional[str] = None
    soap_assessment: Optional[str] = None
    soap_plan: Optional[str] = None
    structured_data: Optional[dict] = None
    transcript: Optional[str] = None
    compliance_score: Optional[float] = None
    compliance_warnings: Optional[list[str]] = None
    status: str = "final"
    emr_synced: bool = False
    emr_note_id: Optional[str] = None


class ClinicalNoteUpdate(BaseModel):
    note_type: Optional[str] = None
    note_date: Optional[date] = None
    discipline: Optional[str] = None
    therapist_name: Optional[str] = None
    soap_subjective: Optional[str] = None
    soap_objective: Optional[str] = None
    soap_assessment: Optional[str] = None
    soap_plan: Optional[str] = None
    structured_data: Optional[dict] = None
    transcript: Optional[str] = None
    compliance_score: Optional[float] = None
    compliance_warnings: Optional[list[str]] = None
    status: Optional[str] = None
    emr_synced: Optional[bool] = None
    emr_note_id: Optional[str] = None


class ClinicalNoteRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    note_type: str
    note_date: date
    discipline: str
    therapist_name: Optional[str] = None
    soap_subjective: Optional[str] = None
    soap_objective: Optional[str] = None
    soap_assessment: Optional[str] = None
    soap_plan: Optional[str] = None
    structured_data: Optional[dict] = None
    transcript: Optional[str] = None
    compliance_score: Optional[float] = None
    compliance_warnings: Optional[list] = None
    status: str = "final"
    emr_synced: bool = False
    emr_note_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


# --- Insurance ---

class InsuranceCreate(BaseModel):
    patient_id: uuid.UUID
    payer_name: str
    member_id: str
    group_id: Optional[str] = None
    auth_number: Optional[str] = None
    authorized_visits: Optional[int] = None
    visits_used: int = 0
    frequency: Optional[str] = None
    duration_weeks: Optional[int] = None
    expiry_date: Optional[date] = None
    is_primary: bool = True


class InsuranceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    payer_name: str
    member_id: str
    group_id: Optional[str] = None
    auth_number: Optional[str] = None
    authorized_visits: Optional[int] = None
    visits_used: int
    frequency: Optional[str] = None
    duration_weeks: Optional[int] = None
    expiry_date: Optional[date] = None
    is_primary: bool


# --- Encounter ---

class EncounterCreate(BaseModel):
    patient_id: uuid.UUID
    provider_id: Optional[uuid.UUID] = None
    encounter_date: datetime
    discipline: str
    setting: Optional[str] = None
    encounter_type: str
    status: str = "scheduled"
    soap_note_id: Optional[str] = None
    rehab_os_consultation_id: Optional[str] = None


class EncounterUpdate(BaseModel):
    status: Optional[str] = None
    provider_id: Optional[uuid.UUID] = None
    soap_note_id: Optional[str] = None
    rehab_os_consultation_id: Optional[str] = None


class EncounterRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    provider_id: Optional[uuid.UUID] = None
    encounter_date: datetime
    discipline: str
    setting: Optional[str] = None
    encounter_type: str
    status: str
    soap_note_id: Optional[str] = None
    rehab_os_consultation_id: Optional[str] = None
    created_at: datetime


# --- Provider ---

class ProviderCreate(BaseModel):
    first_name: str
    last_name: str
    credentials: Optional[str] = None
    npi: Optional[str] = None
    discipline: str
    email: Optional[str] = None


class ProviderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    first_name: str
    last_name: str
    credentials: Optional[str] = None
    npi: Optional[str] = None
    discipline: str
    email: Optional[str] = None
    active: bool


# --- Referral ---

class ReferralCreate(BaseModel):
    patient_id: uuid.UUID
    referring_provider_name: Optional[str] = None
    referring_provider_npi: Optional[str] = None
    referral_date: Optional[date] = None
    received_date: Optional[date] = None
    diagnosis_codes: Optional[list[str]] = None
    raw_document_path: Optional[str] = None
    intake_result_json: Optional[dict] = None
    status: str = "pending"


class ReferralRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    referring_provider_name: Optional[str] = None
    referring_provider_npi: Optional[str] = None
    referral_date: Optional[date] = None
    received_date: Optional[date] = None
    diagnosis_codes: Optional[list[str]] = None
    raw_document_path: Optional[str] = None
    intake_result_json: Optional[dict] = None
    status: str


# --- Billing ---

class BillingCreate(BaseModel):
    encounter_id: uuid.UUID
    cpt_code: str
    units: int = 1
    minutes: Optional[int] = None
    modifier: Optional[str] = None
    status: str = "unbilled"
    payer_response: Optional[str] = None


class BillingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    encounter_id: uuid.UUID
    cpt_code: str
    units: int
    minutes: Optional[int] = None
    modifier: Optional[str] = None
    status: str
    payer_response: Optional[str] = None
