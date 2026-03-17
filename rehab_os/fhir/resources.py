"""FHIR R4 resource builders for RehabOS.

Transforms internal SQLAlchemy models into compliant FHIR R4 JSON
resources following HL7 structure conventions.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any, Optional

from rehab_os.fhir.mappings import (
    CPT_SYSTEM,
    DISCIPLINE_SNOMED,
    ENCOUNTER_STATUS_MAP,
    ENCOUNTER_TYPE_SNOMED,
    ICD10_SYSTEM,
    LOINC_SYSTEM,
    MEASURE_LOINC,
    NOTE_TYPE_LOINC,
    SETTING_CLASS_MAP,
    SEX_MAP,
    SNOMED_SYSTEM,
)


def _ts(dt: datetime | date | None) -> str | None:
    """Format a datetime/date as ISO-8601 string."""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return dt.isoformat()


def _meta(
    last_updated: datetime | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Build FHIR meta element."""
    m: dict[str, Any] = {
        "lastUpdated": _ts(last_updated or datetime.now(timezone.utc)),
    }
    if profile:
        m["profile"] = [profile]
    return m


class FHIRResourceBuilder:
    """Builds FHIR R4 JSON resources from RehabOS database models."""

    # ------------------------------------------------------------------
    # Patient
    # ------------------------------------------------------------------
    def build_patient(self, patient_db: Any) -> dict[str, Any]:
        """Build a FHIR Patient resource.

        Profile: http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient
        """
        resource: dict[str, Any] = {
            "resourceType": "Patient",
            "id": str(patient_db.id),
            "meta": _meta(
                last_updated=getattr(patient_db, "updated_at", None),
                profile="http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient",
            ),
            "active": getattr(patient_db, "active", True),
            "name": [
                {
                    "use": "official",
                    "family": patient_db.last_name,
                    "given": [patient_db.first_name],
                }
            ],
            "gender": SEX_MAP.get(
                (patient_db.sex or "").lower(), "unknown"
            ),
            "birthDate": _ts(patient_db.dob),
        }

        # Telecom
        telecom: list[dict[str, str]] = []
        if getattr(patient_db, "phone", None):
            telecom.append(
                {"system": "phone", "value": patient_db.phone, "use": "home"}
            )
        if getattr(patient_db, "email", None):
            telecom.append(
                {"system": "email", "value": patient_db.email, "use": "home"}
            )
        if telecom:
            resource["telecom"] = telecom

        # Address (stored as single text field)
        if getattr(patient_db, "address", None):
            resource["address"] = [
                {"use": "home", "text": patient_db.address}
            ]

        # Emergency contact
        if getattr(patient_db, "emergency_contact_name", None):
            contact: dict[str, Any] = {
                "relationship": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v2-0131",
                                "code": "C",
                                "display": "Emergency Contact",
                            }
                        ]
                    }
                ],
                "name": {"text": patient_db.emergency_contact_name},
            }
            if getattr(patient_db, "emergency_contact_phone", None):
                contact["telecom"] = [
                    {"system": "phone", "value": patient_db.emergency_contact_phone}
                ]
            resource["contact"] = [contact]

        return resource

    # ------------------------------------------------------------------
    # Encounter
    # ------------------------------------------------------------------
    def build_encounter(
        self,
        encounter_db: Any,
        patient_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a FHIR Encounter resource.

        Profile: http://hl7.org/fhir/us/core/StructureDefinition/us-core-encounter
        """
        pid = str(patient_id or encounter_db.patient_id)
        status = ENCOUNTER_STATUS_MAP.get(
            (encounter_db.status or "").lower(), "unknown"
        )

        # Encounter class from clinical setting
        setting_key = (getattr(encounter_db, "setting", None) or "outpatient").lower()
        enc_class = SETTING_CLASS_MAP.get(setting_key, SETTING_CLASS_MAP["outpatient"])

        # Encounter type from encounter_type field
        etype_key = (getattr(encounter_db, "encounter_type", None) or "treatment").lower()
        etype_snomed = ENCOUNTER_TYPE_SNOMED.get(
            etype_key, ENCOUNTER_TYPE_SNOMED["treatment"]
        )

        resource: dict[str, Any] = {
            "resourceType": "Encounter",
            "id": str(encounter_db.id),
            "meta": _meta(
                last_updated=getattr(encounter_db, "created_at", None),
                profile="http://hl7.org/fhir/us/core/StructureDefinition/us-core-encounter",
            ),
            "status": status,
            "class": {
                "system": enc_class["system"],
                "code": enc_class["code"],
                "display": enc_class["display"],
            },
            "type": [
                {
                    "coding": [
                        {
                            "system": SNOMED_SYSTEM,
                            "code": etype_snomed["code"],
                            "display": etype_snomed["display"],
                        }
                    ]
                }
            ],
            "subject": {"reference": f"Patient/{pid}"},
            "period": {
                "start": _ts(encounter_db.encounter_date),
            },
        }

        # Provider participant
        provider_id = getattr(encounter_db, "provider_id", None)
        if provider_id:
            resource["participant"] = [
                {
                    "type": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/v3-ParticipationType",
                                    "code": "PPRF",
                                    "display": "primary performer",
                                }
                            ]
                        }
                    ],
                    "individual": {
                        "reference": f"Practitioner/{provider_id}"
                    },
                }
            ]

        # Discipline as service type
        discipline = getattr(encounter_db, "discipline", None)
        if discipline and discipline.lower() in DISCIPLINE_SNOMED:
            snomed = DISCIPLINE_SNOMED[discipline.lower()]
            resource["serviceType"] = {
                "coding": [
                    {
                        "system": SNOMED_SYSTEM,
                        "code": snomed["code"],
                        "display": snomed["display"],
                    }
                ]
            }

        return resource

    # ------------------------------------------------------------------
    # Practitioner
    # ------------------------------------------------------------------
    def build_practitioner(self, provider_db: Any) -> dict[str, Any]:
        """Build a FHIR Practitioner resource.

        Profile: http://hl7.org/fhir/us/core/StructureDefinition/us-core-practitioner
        """
        resource: dict[str, Any] = {
            "resourceType": "Practitioner",
            "id": str(provider_db.id),
            "meta": _meta(
                profile="http://hl7.org/fhir/us/core/StructureDefinition/us-core-practitioner",
            ),
            "active": getattr(provider_db, "active", True),
            "name": [
                {
                    "use": "official",
                    "family": provider_db.last_name,
                    "given": [provider_db.first_name],
                    **(
                        {"suffix": [provider_db.credentials]}
                        if getattr(provider_db, "credentials", None)
                        else {}
                    ),
                }
            ],
        }

        # NPI identifier
        npi = getattr(provider_db, "npi", None)
        if npi:
            resource["identifier"] = [
                {
                    "system": "http://hl7.org/fhir/sid/us-npi",
                    "value": npi,
                }
            ]

        # Telecom
        email = getattr(provider_db, "email", None)
        if email:
            resource["telecom"] = [
                {"system": "email", "value": email, "use": "work"}
            ]

        # Qualification from discipline
        discipline = getattr(provider_db, "discipline", None)
        if discipline and discipline.lower() in DISCIPLINE_SNOMED:
            snomed = DISCIPLINE_SNOMED[discipline.lower()]
            resource["qualification"] = [
                {
                    "code": {
                        "coding": [
                            {
                                "system": SNOMED_SYSTEM,
                                "code": snomed["code"],
                                "display": snomed["display"],
                            }
                        ]
                    }
                }
            ]

        return resource

    # ------------------------------------------------------------------
    # Observation (outcome measures)
    # ------------------------------------------------------------------
    def build_observation(
        self,
        measure_name: str,
        score: float,
        patient_id: str,
        observation_date: datetime | date | None = None,
        encounter_id: str | None = None,
        performer_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a FHIR Observation for a rehab outcome measure.

        Profile: http://hl7.org/fhir/us/core/StructureDefinition/us-core-observation-clinical-result
        """
        loinc = MEASURE_LOINC.get(measure_name)
        if not loinc:
            raise ValueError(
                f"No LOINC mapping for measure '{measure_name}'. "
                f"Valid: {list(MEASURE_LOINC.keys())}"
            )

        effective = _ts(observation_date or datetime.now(timezone.utc))

        resource: dict[str, Any] = {
            "resourceType": "Observation",
            "id": str(uuid.uuid4()),
            "meta": _meta(
                profile="http://hl7.org/fhir/us/core/StructureDefinition/us-core-observation-clinical-result",
            ),
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "survey",
                            "display": "Survey",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": LOINC_SYSTEM,
                        "code": loinc["code"],
                        "display": loinc["display"],
                    }
                ],
                "text": measure_name,
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": effective,
            "valueQuantity": {
                "value": score,
                "unit": "{score}",
                "system": "http://unitsofmeasure.org",
                "code": "{score}",
            },
        }

        if encounter_id:
            resource["encounter"] = {
                "reference": f"Encounter/{encounter_id}"
            }

        if performer_id:
            resource["performer"] = [
                {"reference": f"Practitioner/{performer_id}"}
            ]

        return resource

    # ------------------------------------------------------------------
    # CarePlan
    # ------------------------------------------------------------------
    def build_care_plan(
        self,
        goals: list[dict[str, Any]],
        interventions: list[dict[str, Any]],
        patient_id: str,
        encounter_id: str | None = None,
        author_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a FHIR CarePlan resource.

        Profile: http://hl7.org/fhir/us/core/StructureDefinition/us-core-careplan
        """
        activities: list[dict[str, Any]] = []
        for intervention in interventions:
            detail: dict[str, Any] = {
                "status": "in-progress",
                "description": intervention.get("description", ""),
            }
            code = intervention.get("code")
            if code:
                detail["code"] = {
                    "coding": [
                        {
                            "system": SNOMED_SYSTEM,
                            "code": code,
                            "display": intervention.get("display", ""),
                        }
                    ]
                }
            activities.append({"detail": detail})

        resource: dict[str, Any] = {
            "resourceType": "CarePlan",
            "id": str(uuid.uuid4()),
            "meta": _meta(
                profile="http://hl7.org/fhir/us/core/StructureDefinition/us-core-careplan",
            ),
            "status": "active",
            "intent": "plan",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/us/core/CodeSystem/careplan-category",
                            "code": "assess-plan",
                            "display": "Assessment and Plan of Treatment",
                        }
                    ]
                }
            ],
            "subject": {"reference": f"Patient/{patient_id}"},
        }

        if goals:
            resource["goal"] = [
                {"reference": f"Goal/{g.get('id', str(uuid.uuid4()))}"}
                for g in goals
            ]
            resource["description"] = "; ".join(
                g.get("description", "") for g in goals if g.get("description")
            ) or "Rehabilitation plan of care"

        if activities:
            resource["activity"] = activities

        if encounter_id:
            resource["encounter"] = {"reference": f"Encounter/{encounter_id}"}

        if author_id:
            resource["author"] = {"reference": f"Practitioner/{author_id}"}

        return resource

    # ------------------------------------------------------------------
    # DocumentReference (clinical notes)
    # ------------------------------------------------------------------
    def build_clinical_note(
        self, note_db: Any, patient_id: str | None = None
    ) -> dict[str, Any]:
        """Build a FHIR DocumentReference for a clinical note.

        Profile: http://hl7.org/fhir/us/core/StructureDefinition/us-core-documentreference
        """
        pid = str(patient_id or note_db.patient_id)
        note_type_key = (getattr(note_db, "note_type", None) or "progress_note").lower()
        loinc = NOTE_TYPE_LOINC.get(note_type_key, NOTE_TYPE_LOINC["progress_note"])

        # Assemble plain-text content from SOAP sections
        sections = []
        for section in ["soap_subjective", "soap_objective", "soap_assessment", "soap_plan"]:
            text = getattr(note_db, section, None)
            if text:
                label = section.replace("soap_", "").upper()
                sections.append(f"{label}:\n{text}")
        content_text = "\n\n".join(sections) if sections else ""

        resource: dict[str, Any] = {
            "resourceType": "DocumentReference",
            "id": str(note_db.id),
            "meta": _meta(
                last_updated=getattr(note_db, "updated_at", None),
                profile="http://hl7.org/fhir/us/core/StructureDefinition/us-core-documentreference",
            ),
            "status": "current" if getattr(note_db, "status", "final") == "final" else "preliminary",
            "type": {
                "coding": [
                    {
                        "system": LOINC_SYSTEM,
                        "code": loinc["code"],
                        "display": loinc["display"],
                    }
                ]
            },
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category",
                            "code": "clinical-note",
                            "display": "Clinical Note",
                        }
                    ]
                }
            ],
            "subject": {"reference": f"Patient/{pid}"},
            "date": _ts(getattr(note_db, "note_date", None)),
            "content": [
                {
                    "attachment": {
                        "contentType": "text/plain",
                        "data": None,  # populated below if content exists
                    }
                }
            ],
        }

        if content_text:
            import base64

            resource["content"][0]["attachment"]["data"] = base64.b64encode(
                content_text.encode("utf-8")
            ).decode("ascii")

        # Author
        therapist_id = getattr(note_db, "therapist_id", None)
        if therapist_id:
            resource["author"] = [
                {"reference": f"Practitioner/{therapist_id}"}
            ]

        return resource

    # ------------------------------------------------------------------
    # Claim
    # ------------------------------------------------------------------
    def build_claim(self, claim_data: dict[str, Any]) -> dict[str, Any]:
        """Build a FHIR Claim resource from billing data.

        Args:
            claim_data: Dict with keys: patient_id, provider_id, encounter_id,
                        line_items (list of {cpt_code, units, modifier}),
                        diagnosis_codes (list of ICD-10 strings),
                        billing_form (CMS-1500 or UB-04).
        """
        patient_id = claim_data["patient_id"]
        line_items = claim_data.get("line_items", [])
        diagnosis_codes = claim_data.get("diagnosis_codes", [])

        use = "claim"
        claim_type_code = "professional"
        if claim_data.get("billing_form") == "UB-04":
            claim_type_code = "institutional"

        items: list[dict[str, Any]] = []
        for idx, li in enumerate(line_items, start=1):
            item: dict[str, Any] = {
                "sequence": idx,
                "productOrService": {
                    "coding": [
                        {
                            "system": CPT_SYSTEM,
                            "code": li["cpt_code"],
                        }
                    ]
                },
                "quantity": {"value": li.get("units", 1)},
            }
            modifier = li.get("modifier")
            if modifier:
                item["modifier"] = [
                    {
                        "coding": [
                            {
                                "system": CPT_SYSTEM,
                                "code": modifier,
                            }
                        ]
                    }
                ]
            items.append(item)

        resource: dict[str, Any] = {
            "resourceType": "Claim",
            "id": str(uuid.uuid4()),
            "meta": _meta(),
            "status": "active",
            "type": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/claim-type",
                        "code": claim_type_code,
                    }
                ]
            },
            "use": use,
            "patient": {"reference": f"Patient/{patient_id}"},
            "created": _ts(datetime.now(timezone.utc)),
            "provider": {
                "reference": f"Practitioner/{claim_data.get('provider_id', 'unknown')}"
            },
            "priority": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/processpriority",
                        "code": "normal",
                    }
                ]
            },
            "item": items,
        }

        if diagnosis_codes:
            resource["diagnosis"] = [
                {
                    "sequence": idx,
                    "diagnosisCodeableConcept": {
                        "coding": [
                            {
                                "system": ICD10_SYSTEM,
                                "code": code,
                            }
                        ]
                    },
                }
                for idx, code in enumerate(diagnosis_codes, start=1)
            ]

        encounter_id = claim_data.get("encounter_id")
        if encounter_id:
            resource["supportingInfo"] = [
                {
                    "sequence": 1,
                    "category": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/claiminformationcategory",
                                "code": "info",
                            }
                        ]
                    },
                    "valueReference": {
                        "reference": f"Encounter/{encounter_id}"
                    },
                }
            ]

        return resource

    # ------------------------------------------------------------------
    # Bundle
    # ------------------------------------------------------------------
    def build_bundle(
        self,
        resources: list[dict[str, Any]],
        bundle_type: str = "collection",
    ) -> dict[str, Any]:
        """Wrap multiple FHIR resources into a Bundle.

        Args:
            resources: List of FHIR resource dicts.
            bundle_type: One of collection, document, transaction, searchset.
        """
        entries: list[dict[str, Any]] = []
        for res in resources:
            rtype = res.get("resourceType", "Unknown")
            rid = res.get("id", "")
            entry: dict[str, Any] = {
                "fullUrl": f"urn:uuid:{rid}" if rid else f"urn:uuid:{uuid.uuid4()}",
                "resource": res,
            }
            if bundle_type == "transaction":
                entry["request"] = {
                    "method": "PUT",
                    "url": f"{rtype}/{rid}",
                }
            entries.append(entry)

        return {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "meta": _meta(),
            "type": bundle_type,
            "total": len(entries),
            "entry": entries,
        }
