"""Tests for FHIR R4 resource builders."""

import uuid
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from rehab_os.fhir.resources import FHIRResourceBuilder
from rehab_os.fhir.mappings import LOINC_SYSTEM, SNOMED_SYSTEM, CPT_SYSTEM, ICD10_SYSTEM


def _ns(**kwargs) -> SimpleNamespace:
    """Create a SimpleNamespace (mock DB model) with given attributes."""
    return SimpleNamespace(**kwargs)


@pytest.fixture
def builder():
    return FHIRResourceBuilder()


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------

class TestBuildPatient:
    def test_valid_structure(self, builder):
        patient = _ns(
            id=uuid.uuid4(),
            first_name="Jane",
            last_name="Doe",
            sex="female",
            dob=date(1958, 3, 15),
            active=True,
            updated_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
            phone="555-1234",
            email="jane@example.com",
            address="123 Main St",
            emergency_contact_name=None,
        )
        res = builder.build_patient(patient)
        assert res["resourceType"] == "Patient"
        assert res["id"] == str(patient.id)
        assert res["gender"] == "female"
        assert res["birthDate"] == "1958-03-15"
        assert res["name"][0]["family"] == "Doe"
        assert res["name"][0]["given"] == ["Jane"]
        assert res["active"] is True

    def test_has_meta(self, builder):
        patient = _ns(id=uuid.uuid4(), first_name="A", last_name="B", sex="male",
                      dob=date(1990, 1, 1), active=True, updated_at=None)
        res = builder.build_patient(patient)
        assert "meta" in res
        assert "lastUpdated" in res["meta"]

    def test_telecom_phone_and_email(self, builder):
        patient = _ns(id=uuid.uuid4(), first_name="A", last_name="B", sex="male",
                      dob=date(1990, 1, 1), active=True, updated_at=None,
                      phone="555-0000", email="a@b.com", address=None,
                      emergency_contact_name=None)
        res = builder.build_patient(patient)
        assert len(res["telecom"]) == 2
        systems = {t["system"] for t in res["telecom"]}
        assert "phone" in systems
        assert "email" in systems

    def test_unknown_sex_mapped(self, builder):
        patient = _ns(id=uuid.uuid4(), first_name="A", last_name="B", sex="xyz",
                      dob=date(1990, 1, 1), active=True, updated_at=None)
        res = builder.build_patient(patient)
        assert res["gender"] == "unknown"

    def test_emergency_contact(self, builder):
        patient = _ns(id=uuid.uuid4(), first_name="A", last_name="B", sex="female",
                      dob=date(1990, 1, 1), active=True, updated_at=None,
                      emergency_contact_name="Bob", emergency_contact_phone="555-9999")
        res = builder.build_patient(patient)
        assert "contact" in res
        assert res["contact"][0]["name"]["text"] == "Bob"


# ---------------------------------------------------------------------------
# Encounter
# ---------------------------------------------------------------------------

class TestBuildEncounter:
    def test_valid_structure(self, builder):
        enc = _ns(
            id=uuid.uuid4(),
            patient_id=uuid.uuid4(),
            status="completed",
            encounter_date=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
            setting="outpatient",
            encounter_type="treatment",
            provider_id=uuid.uuid4(),
            discipline="pt",
            created_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        )
        res = builder.build_encounter(enc)
        assert res["resourceType"] == "Encounter"
        assert res["status"] == "finished"  # completed -> finished
        assert "class" in res
        assert res["class"]["code"] == "AMB"
        assert res["subject"]["reference"].startswith("Patient/")
        assert "period" in res
        assert "start" in res["period"]

    def test_subject_reference(self, builder):
        pid = uuid.uuid4()
        enc = _ns(id=uuid.uuid4(), patient_id=pid, status="scheduled",
                  encounter_date=datetime.now(timezone.utc), setting="outpatient",
                  encounter_type="treatment", provider_id=None, discipline=None,
                  created_at=None)
        res = builder.build_encounter(enc, patient_id=str(pid))
        assert res["subject"]["reference"] == f"Patient/{pid}"

    def test_homecare_class(self, builder):
        enc = _ns(id=uuid.uuid4(), patient_id=uuid.uuid4(), status="in_progress",
                  encounter_date=datetime.now(timezone.utc), setting="homecare",
                  encounter_type="treatment", provider_id=None, discipline=None,
                  created_at=None)
        res = builder.build_encounter(enc)
        assert res["class"]["code"] == "HH"

    def test_participant_provider(self, builder):
        prov_id = uuid.uuid4()
        enc = _ns(id=uuid.uuid4(), patient_id=uuid.uuid4(), status="completed",
                  encounter_date=datetime.now(timezone.utc), setting="outpatient",
                  encounter_type="treatment", provider_id=prov_id, discipline=None,
                  created_at=None)
        res = builder.build_encounter(enc)
        assert "participant" in res
        assert f"Practitioner/{prov_id}" in res["participant"][0]["individual"]["reference"]


# ---------------------------------------------------------------------------
# Practitioner
# ---------------------------------------------------------------------------

class TestBuildPractitioner:
    def test_valid_structure(self, builder):
        prov = _ns(
            id=uuid.uuid4(),
            first_name="John",
            last_name="Smith",
            npi="1234567890",
            active=True,
            credentials="DPT",
            email="john@clinic.com",
            discipline="pt",
        )
        res = builder.build_practitioner(prov)
        assert res["resourceType"] == "Practitioner"
        assert res["name"][0]["family"] == "Smith"
        assert res["name"][0]["given"] == ["John"]

    def test_has_npi_identifier(self, builder):
        prov = _ns(id=uuid.uuid4(), first_name="A", last_name="B", npi="9876543210",
                   active=True, credentials=None, email=None, discipline=None)
        res = builder.build_practitioner(prov)
        assert "identifier" in res
        npi_id = res["identifier"][0]
        assert npi_id["system"] == "http://hl7.org/fhir/sid/us-npi"
        assert npi_id["value"] == "9876543210"

    def test_no_npi_no_identifier(self, builder):
        prov = _ns(id=uuid.uuid4(), first_name="A", last_name="B", npi=None,
                   active=True, credentials=None, email=None, discipline=None)
        res = builder.build_practitioner(prov)
        assert "identifier" not in res

    def test_qualification_from_discipline(self, builder):
        prov = _ns(id=uuid.uuid4(), first_name="A", last_name="B", npi="1111111111",
                   active=True, credentials=None, email=None, discipline="ot")
        res = builder.build_practitioner(prov)
        assert "qualification" in res
        assert res["qualification"][0]["code"]["coding"][0]["system"] == SNOMED_SYSTEM


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TestBuildObservation:
    def test_valid_structure(self, builder):
        res = builder.build_observation(
            measure_name="LEFS",
            score=72.0,
            patient_id="PAT1",
        )
        assert res["resourceType"] == "Observation"
        assert res["status"] == "final"
        assert res["code"]["coding"][0]["system"] == LOINC_SYSTEM
        assert res["code"]["coding"][0]["code"] == "72100-1"
        assert res["subject"]["reference"] == "Patient/PAT1"
        assert res["valueQuantity"]["value"] == 72.0

    def test_with_encounter(self, builder):
        res = builder.build_observation("NPRS", 7.0, "PAT1", encounter_id="ENC1")
        assert res["encounter"]["reference"] == "Encounter/ENC1"

    def test_with_performer(self, builder):
        res = builder.build_observation("Berg", 45.0, "PAT1", performer_id="PROV1")
        assert res["performer"][0]["reference"] == "Practitioner/PROV1"

    def test_invalid_measure_raises(self, builder):
        with pytest.raises(ValueError, match="No LOINC mapping"):
            builder.build_observation("INVALID_MEASURE", 0.0, "PAT1")

    def test_category_survey(self, builder):
        res = builder.build_observation("ODI", 30.0, "PAT1")
        cat = res["category"][0]["coding"][0]
        assert cat["code"] == "survey"

    def test_effectivedatetime_set(self, builder):
        dt = datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc)
        res = builder.build_observation("TUG", 12.5, "PAT1", observation_date=dt)
        assert "2026-03-15" in res["effectiveDateTime"]


# ---------------------------------------------------------------------------
# CarePlan
# ---------------------------------------------------------------------------

class TestBuildCarePlan:
    def test_valid_structure(self, builder):
        goals = [{"id": "G1", "description": "Improve balance"}]
        interventions = [{"description": "Balance training", "code": "228557008",
                          "display": "Balance exercise"}]
        res = builder.build_care_plan(goals, interventions, patient_id="PAT1")
        assert res["resourceType"] == "CarePlan"
        assert res["status"] == "active"
        assert res["intent"] == "plan"
        assert res["subject"]["reference"] == "Patient/PAT1"

    def test_has_goals(self, builder):
        goals = [{"id": "G1", "description": "Walk independently"}]
        res = builder.build_care_plan(goals, [], "PAT1")
        assert "goal" in res
        assert len(res["goal"]) == 1
        assert "Goal/" in res["goal"][0]["reference"]

    def test_has_activities(self, builder):
        interventions = [
            {"description": "Gait training"},
            {"description": "Strengthening"},
        ]
        res = builder.build_care_plan([], interventions, "PAT1")
        assert "activity" in res
        assert len(res["activity"]) == 2
        assert res["activity"][0]["detail"]["description"] == "Gait training"

    def test_with_encounter_and_author(self, builder):
        res = builder.build_care_plan([], [], "PAT1", encounter_id="ENC1", author_id="PROV1")
        assert res["encounter"]["reference"] == "Encounter/ENC1"
        assert res["author"]["reference"] == "Practitioner/PROV1"


# ---------------------------------------------------------------------------
# Clinical note (DocumentReference)
# ---------------------------------------------------------------------------

class TestBuildClinicalNote:
    def test_valid_structure(self, builder):
        note = _ns(
            id=uuid.uuid4(),
            patient_id=uuid.uuid4(),
            note_type="progress_note",
            status="final",
            note_date=datetime(2026, 3, 15, tzinfo=timezone.utc),
            soap_subjective="Patient reports less pain.",
            soap_objective="ROM improved.",
            soap_assessment="Progressing well.",
            soap_plan="Continue current POC.",
            therapist_id=uuid.uuid4(),
            updated_at=None,
        )
        res = builder.build_clinical_note(note)
        assert res["resourceType"] == "DocumentReference"
        assert res["status"] == "current"
        assert res["subject"]["reference"].startswith("Patient/")
        assert "content" in res
        assert len(res["content"]) == 1

    def test_has_base64_content(self, builder):
        import base64
        note = _ns(
            id=uuid.uuid4(), patient_id=uuid.uuid4(), note_type="progress_note",
            status="final", note_date=datetime.now(timezone.utc),
            soap_subjective="S text", soap_objective="O text",
            soap_assessment="A text", soap_plan="P text",
            therapist_id=None, updated_at=None,
        )
        res = builder.build_clinical_note(note)
        data = res["content"][0]["attachment"]["data"]
        assert data is not None
        decoded = base64.b64decode(data).decode("utf-8")
        assert "SUBJECTIVE" in decoded
        assert "S text" in decoded

    def test_loinc_type_code(self, builder):
        note = _ns(
            id=uuid.uuid4(), patient_id=uuid.uuid4(), note_type="discharge_summary",
            status="final", note_date=datetime.now(timezone.utc),
            soap_subjective=None, soap_objective=None, soap_assessment=None,
            soap_plan=None, therapist_id=None, updated_at=None,
        )
        res = builder.build_clinical_note(note)
        code = res["type"]["coding"][0]["code"]
        assert code == "18842-5"  # Discharge summary LOINC

    def test_author_reference(self, builder):
        tid = uuid.uuid4()
        note = _ns(
            id=uuid.uuid4(), patient_id=uuid.uuid4(), note_type="daily_note",
            status="final", note_date=datetime.now(timezone.utc),
            soap_subjective=None, soap_objective=None, soap_assessment=None,
            soap_plan=None, therapist_id=tid, updated_at=None,
        )
        res = builder.build_clinical_note(note)
        assert res["author"][0]["reference"] == f"Practitioner/{tid}"


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------

class TestBuildClaim:
    def test_valid_structure(self, builder):
        data = {
            "patient_id": "PAT1",
            "provider_id": "PROV1",
            "encounter_id": "ENC1",
            "line_items": [
                {"cpt_code": "97110", "units": 2, "modifier": "GP"},
                {"cpt_code": "97140", "units": 1},
            ],
            "diagnosis_codes": ["M54.5", "M79.3"],
        }
        res = builder.build_claim(data)
        assert res["resourceType"] == "Claim"
        assert res["status"] == "active"
        assert res["use"] == "claim"
        assert res["patient"]["reference"] == "Patient/PAT1"
        assert res["provider"]["reference"] == "Practitioner/PROV1"

    def test_has_cpt_lines(self, builder):
        data = {
            "patient_id": "PAT1",
            "line_items": [
                {"cpt_code": "97110", "units": 2, "modifier": "GP"},
                {"cpt_code": "97530", "units": 1},
            ],
            "diagnosis_codes": [],
        }
        res = builder.build_claim(data)
        assert len(res["item"]) == 2
        assert res["item"][0]["productOrService"]["coding"][0]["code"] == "97110"
        assert res["item"][0]["quantity"]["value"] == 2
        assert "modifier" in res["item"][0]

    def test_has_diagnosis(self, builder):
        data = {
            "patient_id": "PAT1",
            "line_items": [{"cpt_code": "97110", "units": 1}],
            "diagnosis_codes": ["M54.5"],
        }
        res = builder.build_claim(data)
        assert "diagnosis" in res
        assert res["diagnosis"][0]["diagnosisCodeableConcept"]["coding"][0]["code"] == "M54.5"
        assert res["diagnosis"][0]["diagnosisCodeableConcept"]["coding"][0]["system"] == ICD10_SYSTEM

    def test_institutional_type(self, builder):
        data = {
            "patient_id": "PAT1",
            "billing_form": "UB-04",
            "line_items": [{"cpt_code": "97110", "units": 1}],
            "diagnosis_codes": [],
        }
        res = builder.build_claim(data)
        assert res["type"]["coding"][0]["code"] == "institutional"


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

class TestBuildBundle:
    def test_wraps_resources(self, builder):
        resources = [
            {"resourceType": "Patient", "id": "P1"},
            {"resourceType": "Encounter", "id": "E1"},
        ]
        bundle = builder.build_bundle(resources)
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert bundle["total"] == 2
        assert len(bundle["entry"]) == 2
        assert bundle["entry"][0]["resource"]["resourceType"] == "Patient"
        assert bundle["entry"][1]["resource"]["resourceType"] == "Encounter"

    def test_fullurl_set(self, builder):
        resources = [{"resourceType": "Patient", "id": "P1"}]
        bundle = builder.build_bundle(resources)
        assert "urn:uuid:P1" == bundle["entry"][0]["fullUrl"]

    def test_transaction_bundle_has_request(self, builder):
        resources = [{"resourceType": "Patient", "id": "P1"}]
        bundle = builder.build_bundle(resources, bundle_type="transaction")
        assert bundle["type"] == "transaction"
        assert "request" in bundle["entry"][0]
        assert bundle["entry"][0]["request"]["method"] == "PUT"
        assert bundle["entry"][0]["request"]["url"] == "Patient/P1"

    def test_empty_bundle(self, builder):
        bundle = builder.build_bundle([])
        assert bundle["total"] == 0
        assert bundle["entry"] == []
