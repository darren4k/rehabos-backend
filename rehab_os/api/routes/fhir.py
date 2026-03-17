"""FHIR R4 API endpoints for RehabOS.

Exposes patient and encounter data as FHIR R4 JSON resources,
supports bundle import (ADT/referral), and publishes a
CapabilityStatement at /metadata.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.api.dependencies import get_current_user
from rehab_os.core.database import get_db
from rehab_os.core.models import Encounter, Patient, Provider
from rehab_os.fhir.exporter import FHIRExporter
from rehab_os.fhir.resources import FHIRResourceBuilder

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/fhir",
    tags=["FHIR"],
    dependencies=[Depends(get_current_user)],
)

_builder = FHIRResourceBuilder()
_exporter = FHIRExporter(_builder)


# ------------------------------------------------------------------
# GET /fhir/metadata — Capability Statement
# ------------------------------------------------------------------
@router.get("/metadata")
async def capability_statement() -> dict[str, Any]:
    """Return FHIR CapabilityStatement describing this server."""
    return {
        "resourceType": "CapabilityStatement",
        "status": "active",
        "date": "2026-03-17",
        "kind": "instance",
        "fhirVersion": "4.0.1",
        "format": ["json"],
        "implementation": {
            "description": "RehabOS FHIR R4 Server",
            "url": "/api/v1/fhir",
        },
        "rest": [
            {
                "mode": "server",
                "resource": [
                    {
                        "type": "Patient",
                        "profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient",
                        "interaction": [{"code": "read"}],
                        "operation": [
                            {
                                "name": "everything",
                                "definition": "http://hl7.org/fhir/OperationDefinition/Patient-everything",
                            }
                        ],
                    },
                    {
                        "type": "Encounter",
                        "profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-encounter",
                        "interaction": [{"code": "read"}],
                    },
                    {
                        "type": "Practitioner",
                        "profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-practitioner",
                        "interaction": [{"code": "read"}],
                    },
                    {
                        "type": "Observation",
                        "profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-observation-clinical-result",
                        "interaction": [{"code": "read"}],
                    },
                    {
                        "type": "DocumentReference",
                        "profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-documentreference",
                        "interaction": [{"code": "read"}],
                    },
                    {
                        "type": "Bundle",
                        "interaction": [{"code": "create"}],
                    },
                ],
            }
        ],
    }


# ------------------------------------------------------------------
# GET /fhir/Patient/{patient_id}
# ------------------------------------------------------------------
@router.get("/Patient/{patient_id}")
async def get_patient_fhir(
    patient_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return a single patient as a FHIR Patient resource."""
    result = await db.execute(
        select(Patient).where(Patient.id == patient_id)
    )
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return _builder.build_patient(patient)


# ------------------------------------------------------------------
# GET /fhir/Encounter/{encounter_id}
# ------------------------------------------------------------------
@router.get("/Encounter/{encounter_id}")
async def get_encounter_fhir(
    encounter_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return a single encounter as a FHIR Encounter resource."""
    result = await db.execute(
        select(Encounter).where(Encounter.id == encounter_id)
    )
    encounter = result.scalar_one_or_none()
    if not encounter:
        raise HTTPException(status_code=404, detail="Encounter not found")
    return _builder.build_encounter(encounter)


# ------------------------------------------------------------------
# GET /fhir/Patient/{patient_id}/$everything
# ------------------------------------------------------------------
@router.get("/Patient/{patient_id}/$everything")
async def patient_everything(
    patient_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Export the complete patient record as a FHIR Bundle.

    Includes Patient, Encounters, Practitioners, Observations
    (outcome scores), and DocumentReferences (clinical notes).
    """
    try:
        return await _exporter.export_patient_bundle(patient_id, db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ------------------------------------------------------------------
# POST /fhir/Bundle — import
# ------------------------------------------------------------------
@router.post("/Bundle", status_code=201)
async def import_bundle(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Import a FHIR Bundle (ADT message, referral, etc.).

    Currently supports Patient resources within the bundle.
    Other resource types are acknowledged but not persisted.
    """
    body = await request.json()

    if body.get("resourceType") != "Bundle":
        raise HTTPException(
            status_code=400, detail="Expected resourceType 'Bundle'"
        )

    # Validate
    errors = _exporter.validate_resource(body)
    if errors:
        raise HTTPException(status_code=422, detail={"validation_errors": errors})

    entries = body.get("entry", [])
    imported: list[str] = []
    skipped: list[str] = []

    for entry in entries:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType")

        if rtype == "Patient":
            # Upsert patient from FHIR resource
            names = resource.get("name", [])
            first_name = ""
            last_name = ""
            if names:
                last_name = names[0].get("family", "")
                given = names[0].get("given", [])
                first_name = given[0] if given else ""

            gender = resource.get("gender", "unknown")
            birth_date = resource.get("birthDate")
            if not (first_name and last_name and birth_date):
                skipped.append(f"Patient/{resource.get('id', '?')}: missing required fields")
                continue

            from datetime import date as date_type

            patient = Patient(
                first_name=first_name,
                last_name=last_name,
                dob=date_type.fromisoformat(birth_date),
                sex=gender,
                active=resource.get("active", True),
            )

            # Address
            addresses = resource.get("address", [])
            if addresses:
                patient.address = addresses[0].get("text", "")

            # Telecom
            for tc in resource.get("telecom", []):
                if tc.get("system") == "phone":
                    patient.phone = tc.get("value")
                elif tc.get("system") == "email":
                    patient.email = tc.get("value")

            db.add(patient)
            imported.append(f"Patient/{patient.id}")
        else:
            skipped.append(f"{rtype}/{resource.get('id', '?')}: import not supported")

    if imported:
        await db.commit()

    return {
        "resourceType": "OperationOutcome",
        "issue": [
            {
                "severity": "information",
                "code": "informational",
                "diagnostics": f"Imported {len(imported)} resource(s), skipped {len(skipped)}",
                "details": {
                    "imported": imported,
                    "skipped": skipped,
                },
            }
        ],
    }
