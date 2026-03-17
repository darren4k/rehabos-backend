"""Tests for FHIR export service validation."""

import uuid
from datetime import date, datetime, timezone

import pytest

from rehab_os.fhir.exporter import FHIRExporter
from rehab_os.fhir.resources import FHIRResourceBuilder


@pytest.fixture
def exporter():
    return FHIRExporter()


# ---------------------------------------------------------------------------
# Validate resource
# ---------------------------------------------------------------------------

class TestValidateResource:
    def test_valid_patient(self, exporter):
        resource = {
            "resourceType": "Patient",
            "id": str(uuid.uuid4()),
            "name": [{"family": "Doe", "given": ["Jane"]}],
            "gender": "female",
            "birthDate": "1958-03-15",
        }
        errors = exporter.validate_resource(resource)
        assert errors == []

    def test_valid_encounter(self, exporter):
        resource = {
            "resourceType": "Encounter",
            "id": str(uuid.uuid4()),
            "status": "finished",
            "class": {"code": "AMB"},
            "subject": {"reference": "Patient/123"},
        }
        errors = exporter.validate_resource(resource)
        assert errors == []

    def test_valid_observation(self, exporter):
        resource = {
            "resourceType": "Observation",
            "id": str(uuid.uuid4()),
            "status": "final",
            "code": {"coding": [{"code": "72100-1"}]},
            "subject": {"reference": "Patient/123"},
        }
        errors = exporter.validate_resource(resource)
        assert errors == []

    def test_valid_bundle(self, exporter):
        resource = {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "P1",
                        "name": [{"family": "Test"}],
                        "gender": "male",
                        "birthDate": "2000-01-01",
                    }
                }
            ],
        }
        errors = exporter.validate_resource(resource)
        assert errors == []

    def test_missing_resource_type(self, exporter):
        resource = {"id": "123", "name": []}
        errors = exporter.validate_resource(resource)
        assert any("resourceType" in e for e in errors)

    def test_missing_id(self, exporter):
        resource = {"resourceType": "Patient", "name": [], "gender": "male",
                    "birthDate": "2000-01-01"}
        errors = exporter.validate_resource(resource)
        assert any("id" in e for e in errors)

    def test_missing_required_fields_patient(self, exporter):
        resource = {"resourceType": "Patient", "id": "P1"}
        errors = exporter.validate_resource(resource)
        assert any("name" in e for e in errors)
        assert any("gender" in e for e in errors)
        assert any("birthDate" in e for e in errors)

    def test_missing_required_fields_encounter(self, exporter):
        resource = {"resourceType": "Encounter", "id": "E1"}
        errors = exporter.validate_resource(resource)
        assert any("status" in e for e in errors)
        assert any("class" in e for e in errors)
        assert any("subject" in e for e in errors)

    def test_invalid_reference_format(self, exporter):
        resource = {
            "resourceType": "Observation",
            "id": "O1",
            "status": "final",
            "code": {"coding": []},
            "subject": {"reference": "bad-reference-no-slash"},
        }
        errors = exporter.validate_resource(resource)
        assert any("Invalid reference" in e for e in errors)

    def test_valid_urn_reference(self, exporter):
        resource = {
            "resourceType": "Observation",
            "id": "O1",
            "status": "final",
            "code": {"coding": []},
            "subject": {"reference": "urn:uuid:12345678-1234-1234-1234-123456789012"},
        }
        errors = exporter.validate_resource(resource)
        ref_errors = [e for e in errors if "reference" in e.lower()]
        assert len(ref_errors) == 0

    def test_bundle_validates_inner_resources(self, exporter):
        resource = {
            "resourceType": "Bundle",
            "id": "B1",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "P1"}},  # missing name, gender, birthDate
            ],
        }
        errors = exporter.validate_resource(resource)
        assert any("name" in e for e in errors)

    def test_bundle_entry_missing_resource(self, exporter):
        resource = {
            "resourceType": "Bundle",
            "id": "B1",
            "type": "collection",
            "entry": [{}],
        }
        errors = exporter.validate_resource(resource)
        assert any("Missing 'resource'" in e for e in errors)


# ---------------------------------------------------------------------------
# Integration: build + validate round-trip
# ---------------------------------------------------------------------------

class TestBuildAndValidate:
    def test_built_patient_passes_validation(self):
        from types import SimpleNamespace
        builder = FHIRResourceBuilder()
        exporter = FHIRExporter(builder)
        patient = SimpleNamespace(
            id=uuid.uuid4(), first_name="Jane", last_name="Doe",
            sex="female", dob=date(1958, 3, 15), active=True, updated_at=None,
        )
        resource = builder.build_patient(patient)
        errors = exporter.validate_resource(resource)
        assert errors == []

    def test_built_observation_passes_validation(self):
        builder = FHIRResourceBuilder()
        exporter = FHIRExporter(builder)
        resource = builder.build_observation("LEFS", 72.0, "PAT1")
        errors = exporter.validate_resource(resource)
        assert errors == []

    def test_built_claim_passes_validation(self):
        builder = FHIRResourceBuilder()
        exporter = FHIRExporter(builder)
        resource = builder.build_claim({
            "patient_id": "PAT1",
            "provider_id": "PROV1",
            "line_items": [{"cpt_code": "97110", "units": 1}],
            "diagnosis_codes": ["M54.5"],
        })
        errors = exporter.validate_resource(resource)
        assert errors == []
