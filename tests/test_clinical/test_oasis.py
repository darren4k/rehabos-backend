"""Tests for rehab_os.clinical.oasis — OASIS-E assessment module."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from rehab_os.clinical.oasis import (
    OASIS_ITEMS,
    OASISAssessment,
    OASISAssessmentType,
    OASISService,
    OASISValidationError,
    _DISCHARGE_SKIP_ITEMS,
    _SOC_ROC_ONLY_ITEMS,
)


# ---------------------------------------------------------------------------
# OASISAssessment unit tests
# ---------------------------------------------------------------------------

class TestCreateAssessment:
    def test_create_assessment_soc(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        assert a.assessment_type == OASISAssessmentType.SOC
        assert a.patient_id == "pt-001"
        assert a.assessment_id  # non-empty UUID
        assert a.submitted is False
        assert a.responses == {}

    def test_create_assessment_custom_id(self):
        a = OASISAssessment(OASISAssessmentType.DISCHARGE, "pt-002", assessment_id="custom-id")
        assert a.assessment_id == "custom-id"


class TestSetResponse:
    def test_set_valid_response(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        a.set_response("M0069", 1)  # Male
        assert a.get_response("M0069") == 1

    def test_set_text_response(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        a.set_response("M0040", "John Doe")
        assert a.get_response("M0040") == "John Doe"

    def test_set_invalid_item_raises(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        with pytest.raises(ValueError, match="Unknown OASIS item"):
            a.set_response("ZZZZZ", 1)

    def test_set_invalid_categorical_raises(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        with pytest.raises(ValueError, match="Invalid value"):
            a.set_response("M0069", 99)

    def test_updated_at_changes(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        original = a.updated_at
        a.set_response("M0040", "Jane")
        assert a.updated_at >= original


class TestValidateMissing:
    def test_validate_missing_required(self):
        """A fresh SOC assessment should have many missing-required errors."""
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        errors = a.validate()
        assert len(errors) > 0
        assert all(e["severity"] == "error" for e in errors)
        # All errors should reference missing items
        assert all("missing" in e["message"].lower() for e in errors)

    def test_validate_all_required_filled(self):
        """Fill every required item for a SOC — should get zero errors."""
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        required = a.get_required_items()
        for item in required:
            if item.options and item.response_type == "categorical":
                a.set_response(item.item_id, item.options[0]["value"])
            elif item.response_type == "numeric":
                a.set_response(item.item_id, 0)
            elif item.response_type == "date":
                a.set_response(item.item_id, "2026-01-15")
            else:
                a.set_response(item.item_id, "test-value")
        errors = [e for e in a.validate() if e["severity"] == "error"]
        assert errors == []


class TestCompletionPercentage:
    def test_completion_zero(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        assert a.get_completion_pct() == 0.0

    def test_completion_partial(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        required = a.get_required_items()
        # Fill half
        half = required[: len(required) // 2]
        for item in half:
            if item.options and item.response_type == "categorical":
                a.set_response(item.item_id, item.options[0]["value"])
            else:
                a.set_response(item.item_id, "x")
        pct = a.get_completion_pct()
        assert 0.0 < pct < 100.0

    def test_completion_full(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        required = a.get_required_items()
        for item in required:
            if item.options and item.response_type == "categorical":
                a.set_response(item.item_id, item.options[0]["value"])
            elif item.response_type == "numeric":
                a.set_response(item.item_id, 0)
            elif item.response_type == "date":
                a.set_response(item.item_id, "2026-01-15")
            else:
                a.set_response(item.item_id, "x")
        assert a.get_completion_pct() == 100.0


class TestGetRequiredItemsByType:
    def test_soc_includes_soc_only_items(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        req_ids = {i.item_id for i in a.get_required_items()}
        for soc_item in _SOC_ROC_ONLY_ITEMS:
            assert soc_item in req_ids, f"{soc_item} should be required for SOC"

    def test_discharge_excludes_soc_only_and_clinical_items(self):
        a = OASISAssessment(OASISAssessmentType.DISCHARGE, "pt-001")
        req_ids = {i.item_id for i in a.get_required_items()}
        for item_id in _SOC_ROC_ONLY_ITEMS:
            assert item_id not in req_ids, f"{item_id} should NOT be required for DISCHARGE"
        for item_id in _DISCHARGE_SKIP_ITEMS:
            assert item_id not in req_ids, f"{item_id} should be skipped for DISCHARGE"

    def test_recert_includes_soc_only_items(self):
        """RECERT is NOT in SOC/ROC-only set, so M0030/M0110 should be excluded."""
        a = OASISAssessment(OASISAssessmentType.RECERT, "pt-001")
        req_ids = {i.item_id for i in a.get_required_items()}
        for item_id in _SOC_ROC_ONLY_ITEMS:
            assert item_id not in req_ids


class TestSkipLogic:
    def test_m1324_skipped_when_m1322_zero(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        a.set_response("M1322", 0)
        req_ids = {i.item_id for i in a.get_required_items()}
        assert "M1324" not in req_ids

    def test_m1324_not_required_even_when_m1322_positive(self):
        """M1324 is marked required=False, so it never enters the required list.
        The skip_logic field is informational ('Only if M1322 > 0')."""
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        a.set_response("M1322", 2)
        req_ids = {i.item_id for i in a.get_required_items()}
        # M1324 has required=False, so it is not in the required set
        assert "M1324" not in req_ids


class TestSubmissionFormat:
    def test_to_submission_format_structure(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        a.set_response("M0040", "John Doe")
        a.set_response("M0069", 1)
        fmt = a.to_submission_format()
        assert fmt["patient_id"] == "pt-001"
        assert fmt["assessment_type"] == "start_of_care"
        assert "M0040" in fmt["items"]
        assert "M0069" in fmt["items"]
        assert fmt["submitted_at"] is None

    def test_to_submission_format_submitted(self):
        a = OASISAssessment(OASISAssessmentType.SOC, "pt-001")
        a.submitted = True
        fmt = a.to_submission_format()
        assert fmt["submitted_at"] is not None


# ---------------------------------------------------------------------------
# OASISService tests
# ---------------------------------------------------------------------------

class TestOASISService:
    def test_create_and_save(self):
        svc = OASISService()
        a = svc.create_assessment(OASISAssessmentType.SOC, "pt-100")
        assert svc.get_assessment(a.assessment_id) is a
        svc.save_progress(a.assessment_id, {"M0040": "Jane", "M1021": "M54.5"})
        assert a.get_response("M0040") == "Jane"
        assert a.get_response("M1021") == "M54.5"

    def test_save_submitted_raises(self):
        svc = OASISService()
        a = svc.create_assessment(OASISAssessmentType.SOC, "pt-100")
        a.submitted = True
        with pytest.raises(ValueError, match="already submitted"):
            svc.save_progress(a.assessment_id, {"M0040": "X"})

    def test_validate_for_submission(self):
        svc = OASISService()
        a = svc.create_assessment(OASISAssessmentType.SOC, "pt-100")
        errors = svc.validate_for_submission(a.assessment_id)
        assert len(errors) > 0

    def test_list_assessments_filter(self):
        svc = OASISService()
        svc.create_assessment(OASISAssessmentType.SOC, "pt-A")
        svc.create_assessment(OASISAssessmentType.DISCHARGE, "pt-B")
        all_list = svc.list_assessments()
        assert len(all_list) == 2
        a_only = svc.list_assessments(patient_id="pt-A")
        assert len(a_only) == 1


class TestRecertDueTracking:
    def test_recert_due_within_window(self):
        svc = OASISService()
        a = svc.create_assessment(OASISAssessmentType.SOC, "pt-due")
        # Backdate created_at to 50 days ago (within 14-day warning window)
        a.created_at = (datetime.now(timezone.utc) - timedelta(days=50)).isoformat()
        due = svc.get_recert_due_patients()
        assert len(due) == 1
        assert due[0]["patient_id"] == "pt-due"
        assert due[0]["days_until_due"] <= 14

    def test_recert_not_due_yet(self):
        svc = OASISService()
        a = svc.create_assessment(OASISAssessmentType.SOC, "pt-ok")
        # Created today — 60 days until due, well outside window
        due = svc.get_recert_due_patients()
        assert len(due) == 0

    def test_recert_overdue(self):
        svc = OASISService()
        a = svc.create_assessment(OASISAssessmentType.SOC, "pt-late")
        a.created_at = (datetime.now(timezone.utc) - timedelta(days=70)).isoformat()
        due = svc.get_recert_due_patients()
        assert len(due) == 1
        assert due[0]["overdue"] is True
