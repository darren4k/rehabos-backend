"""FHIR export service for RehabOS.

Orchestrates building complete FHIR Bundles from database records,
including patient-everything exports and basic resource validation.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.core.models import (
    BillingRecord,
    ClinicalNote,
    Encounter,
    Insurance,
    OutcomeScoreDB,
    Patient,
    Provider,
)
from rehab_os.fhir.mappings import MEASURE_LOINC
from rehab_os.fhir.resources import FHIRResourceBuilder

logger = logging.getLogger(__name__)

# Required top-level fields per resource type
_REQUIRED_FIELDS: dict[str, list[str]] = {
    "Patient": ["resourceType", "id", "name", "gender", "birthDate"],
    "Encounter": ["resourceType", "id", "status", "class", "subject"],
    "Practitioner": ["resourceType", "id", "name"],
    "Observation": ["resourceType", "id", "status", "code", "subject"],
    "CarePlan": ["resourceType", "id", "status", "intent", "subject"],
    "DocumentReference": ["resourceType", "id", "status", "type", "subject", "content"],
    "Claim": ["resourceType", "id", "status", "type", "patient", "provider", "item"],
    "Bundle": ["resourceType", "id", "type", "entry"],
}


class FHIRExporter:
    """High-level FHIR export service."""

    def __init__(self, builder: FHIRResourceBuilder | None = None) -> None:
        self.builder = builder or FHIRResourceBuilder()

    # ------------------------------------------------------------------
    # Patient $everything
    # ------------------------------------------------------------------
    async def export_patient_bundle(
        self, patient_id: str, db: AsyncSession
    ) -> dict[str, Any]:
        """Export a complete patient record as a FHIR Bundle.

        Includes: Patient, Encounters, Observations (outcome scores),
        DocumentReferences (clinical notes), Practitioner references.
        """
        resources: list[dict[str, Any]] = []

        # Patient
        result = await db.execute(
            select(Patient).where(Patient.id == patient_id)
        )
        patient = result.scalar_one_or_none()
        if not patient:
            raise ValueError(f"Patient {patient_id} not found")
        resources.append(self.builder.build_patient(patient))

        # Encounters
        enc_result = await db.execute(
            select(Encounter).where(Encounter.patient_id == patient_id)
        )
        encounters = enc_result.scalars().all()
        seen_providers: set[str] = set()
        for enc in encounters:
            resources.append(self.builder.build_encounter(enc, patient_id))
            if enc.provider_id:
                seen_providers.add(str(enc.provider_id))

        # Practitioners (deduplicated)
        for pid in seen_providers:
            prov_result = await db.execute(
                select(Provider).where(Provider.id == pid)
            )
            prov = prov_result.scalar_one_or_none()
            if prov:
                resources.append(self.builder.build_practitioner(prov))

        # Outcome scores -> Observations
        score_result = await db.execute(
            select(OutcomeScoreDB).where(
                OutcomeScoreDB.patient_id == patient_id
            )
        )
        scores = score_result.scalars().all()
        for score in scores:
            if score.measure_name in MEASURE_LOINC:
                resources.append(
                    self.builder.build_observation(
                        measure_name=score.measure_name,
                        score=score.score,
                        patient_id=patient_id,
                        observation_date=score.recorded_at,
                        encounter_id=score.encounter_id,
                        performer_id=score.recorded_by,
                    )
                )

        # Clinical notes -> DocumentReferences
        note_result = await db.execute(
            select(ClinicalNote).where(ClinicalNote.patient_id == patient_id)
        )
        notes = note_result.scalars().all()
        for note in notes:
            resources.append(
                self.builder.build_clinical_note(note, patient_id)
            )

        return self.builder.build_bundle(resources, bundle_type="collection")

    # ------------------------------------------------------------------
    # Single encounter export
    # ------------------------------------------------------------------
    async def export_encounter(
        self, encounter_id: str, db: AsyncSession
    ) -> dict[str, Any]:
        """Export a single encounter with related resources as a FHIR Bundle."""
        resources: list[dict[str, Any]] = []

        result = await db.execute(
            select(Encounter).where(Encounter.id == encounter_id)
        )
        encounter = result.scalar_one_or_none()
        if not encounter:
            raise ValueError(f"Encounter {encounter_id} not found")

        patient_id = str(encounter.patient_id)
        resources.append(
            self.builder.build_encounter(encounter, patient_id)
        )

        # Provider
        if encounter.provider_id:
            prov_result = await db.execute(
                select(Provider).where(Provider.id == encounter.provider_id)
            )
            prov = prov_result.scalar_one_or_none()
            if prov:
                resources.append(self.builder.build_practitioner(prov))

        # Billing records for this encounter
        bill_result = await db.execute(
            select(BillingRecord).where(
                BillingRecord.encounter_id == encounter_id
            )
        )
        billing = bill_result.scalars().all()
        if billing:
            claim_data = {
                "patient_id": patient_id,
                "provider_id": str(encounter.provider_id) if encounter.provider_id else "unknown",
                "encounter_id": encounter_id,
                "line_items": [
                    {
                        "cpt_code": b.cpt_code,
                        "units": b.units,
                        "modifier": b.modifier,
                    }
                    for b in billing
                ],
                "diagnosis_codes": [],
            }
            resources.append(self.builder.build_claim(claim_data))

        # Outcome scores recorded for this encounter
        score_result = await db.execute(
            select(OutcomeScoreDB).where(
                OutcomeScoreDB.encounter_id == encounter_id
            )
        )
        scores = score_result.scalars().all()
        for score in scores:
            if score.measure_name in MEASURE_LOINC:
                resources.append(
                    self.builder.build_observation(
                        measure_name=score.measure_name,
                        score=score.score,
                        patient_id=patient_id,
                        observation_date=score.recorded_at,
                        encounter_id=encounter_id,
                        performer_id=score.recorded_by,
                    )
                )

        return self.builder.build_bundle(resources, bundle_type="collection")

    # ------------------------------------------------------------------
    # C-CDA export (simplified)
    # ------------------------------------------------------------------
    async def export_ccda(
        self, patient_id: str, db: AsyncSession
    ) -> str:
        """Generate a simplified C-CDA XML document for interoperability.

        Covers: demographics, problems (from notes), and clinical notes.
        This is a minimal CDA document suitable for basic data exchange;
        production use should integrate a full CDA library.
        """
        result = await db.execute(
            select(Patient).where(Patient.id == patient_id)
        )
        patient = result.scalar_one_or_none()
        if not patient:
            raise ValueError(f"Patient {patient_id} not found")

        root = ET.Element("ClinicalDocument")
        root.set("xmlns", "urn:hl7-org:v3")
        root.set("xmlns:sdtc", "urn:hl7-org:sdtc")

        # Header
        type_id = ET.SubElement(root, "typeId")
        type_id.set("root", "2.16.840.1.113883.1.3")
        type_id.set("extension", "POCD_HD000040")

        template = ET.SubElement(root, "templateId")
        template.set("root", "2.16.840.1.113883.10.20.22.1.1")

        doc_id = ET.SubElement(root, "id")
        doc_id.set("root", str(patient_id))

        code = ET.SubElement(root, "code")
        code.set("code", "34133-9")
        code.set("codeSystem", "2.16.840.1.113883.6.1")
        code.set("displayName", "Summarization of Episode Note")

        effective = ET.SubElement(root, "effectiveTime")
        effective.set("value", datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))

        # Record target (patient demographics)
        record_target = ET.SubElement(root, "recordTarget")
        patient_role = ET.SubElement(record_target, "patientRole")

        pid_el = ET.SubElement(patient_role, "id")
        pid_el.set("root", str(patient_id))

        pat_el = ET.SubElement(patient_role, "patient")
        name_el = ET.SubElement(pat_el, "name")
        given = ET.SubElement(name_el, "given")
        given.text = patient.first_name
        family = ET.SubElement(name_el, "family")
        family.text = patient.last_name

        gender = ET.SubElement(pat_el, "administrativeGenderCode")
        gender.set("code", patient.sex[0].upper() if patient.sex else "UN")
        gender.set("codeSystem", "2.16.840.1.113883.5.1")

        birth = ET.SubElement(pat_el, "birthTime")
        birth.set("value", patient.dob.strftime("%Y%m%d"))

        # Structured body with notes section
        component = ET.SubElement(root, "component")
        structured_body = ET.SubElement(component, "structuredBody")

        note_result = await db.execute(
            select(ClinicalNote).where(ClinicalNote.patient_id == patient_id)
        )
        notes = note_result.scalars().all()

        if notes:
            section_comp = ET.SubElement(structured_body, "component")
            section = ET.SubElement(section_comp, "section")

            sec_code = ET.SubElement(section, "code")
            sec_code.set("code", "11506-3")
            sec_code.set("codeSystem", "2.16.840.1.113883.6.1")
            sec_code.set("displayName", "Progress note")

            title = ET.SubElement(section, "title")
            title.text = "Clinical Notes"

            for note in notes:
                text_parts = []
                for field in ["soap_subjective", "soap_objective", "soap_assessment", "soap_plan"]:
                    val = getattr(note, field, None)
                    if val:
                        text_parts.append(val)
                if text_parts:
                    text_el = ET.SubElement(section, "text")
                    text_el.text = " | ".join(text_parts)

        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_resource(self, resource: dict[str, Any]) -> list[str]:
        """Basic FHIR validation: required fields and reference integrity.

        Returns a list of error strings. Empty list means valid.
        This is NOT a full FHIR profile validator -- use the HL7 FHIR
        Validator for production conformance testing.
        """
        errors: list[str] = []
        rtype = resource.get("resourceType")

        if not rtype:
            errors.append("Missing 'resourceType'")
            return errors

        if not resource.get("id"):
            errors.append(f"{rtype}: Missing 'id'")

        # Check required fields
        required = _REQUIRED_FIELDS.get(rtype, [])
        for field in required:
            if field not in resource:
                errors.append(f"{rtype}: Missing required field '{field}'")

        # Validate references format
        self._check_references(resource, errors, path=rtype)

        # Bundle-specific: validate entries
        if rtype == "Bundle":
            for idx, entry in enumerate(resource.get("entry", [])):
                inner = entry.get("resource")
                if not inner:
                    errors.append(f"Bundle.entry[{idx}]: Missing 'resource'")
                else:
                    inner_errors = self.validate_resource(inner)
                    errors.extend(inner_errors)

        return errors

    def _check_references(
        self,
        obj: Any,
        errors: list[str],
        path: str,
    ) -> None:
        """Recursively check that reference fields have valid format."""
        if isinstance(obj, dict):
            ref = obj.get("reference")
            if ref and isinstance(ref, str):
                if "/" not in ref and not ref.startswith("urn:"):
                    errors.append(
                        f"{path}: Invalid reference format '{ref}' "
                        "(expected 'ResourceType/id' or 'urn:uuid:...')"
                    )
            for key, val in obj.items():
                self._check_references(val, errors, f"{path}.{key}")
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                self._check_references(item, errors, f"{path}[{idx}]")
