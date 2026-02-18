"""Tests for the clinical note PDF generator."""

import pytest
from rehab_os.export.pdf_generator import generate_note_pdf


def _sample_soap():
    return {
        "subjective": "Patient reports reduced pain since last visit. 4/10 at rest.",
        "objective": "AROM R knee flexion 110 degrees. Gait training x 100ft with RW, CGA.",
        "assessment": "Patient progressing toward functional goals. Skilled PT continues to be medically necessary.",
        "plan": "Continue POC 3x/week. Progress amb distance and reduce assist level.",
    }


class TestGenerateNotePDF:
    def test_returns_bytes(self):
        result = generate_note_pdf(
            note_type="daily_note",
            note_date="2026-02-18",
            discipline="pt",
            soap=_sample_soap(),
        )
        assert isinstance(result, bytes)
        assert len(result) > 100

    def test_pdf_header(self):
        result = generate_note_pdf(
            note_type="daily_note",
            note_date="2026-02-18",
            discipline="pt",
            soap=_sample_soap(),
        )
        # PDF magic bytes
        assert result[:5] == b"%PDF-"

    def test_with_therapist_and_structured_data(self):
        result = generate_note_pdf(
            note_type="evaluation",
            note_date="2026-02-18",
            discipline="ot",
            therapist_name="Jane Smith, OTR/L",
            soap=_sample_soap(),
            structured_data={
                "rom": [{"joint": "Shoulder", "motion": "flexion", "value": 95, "side": "right"}],
                "mmt": [{"muscle_group": "Quadriceps", "grade": "4/5", "side": "bilateral"}],
                "standardized_tests": [{"name": "Berg", "score": "42", "max_score": 56, "interpretation": "Low fall risk"}],
                "billing_codes": [{"code": "97110", "description": "Therapeutic Exercise", "units": 2}],
            },
            compliance_score=85.0,
            compliance_warnings=["Consider adding prior level of function"],
        )
        assert isinstance(result, bytes)
        assert len(result) > 500

    def test_empty_soap_sections(self):
        result = generate_note_pdf(
            note_type="progress_note",
            note_date="2026-02-18",
            discipline="slp",
            soap={"subjective": "", "objective": "", "assessment": "", "plan": ""},
        )
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_custom_clinic_name(self):
        result = generate_note_pdf(
            note_type="daily_note",
            note_date="2026-02-18",
            discipline="pt",
            soap=_sample_soap(),
            clinic_name="My Custom Clinic",
        )
        assert isinstance(result, bytes)

    def test_no_structured_data(self):
        result = generate_note_pdf(
            note_type="discharge_summary",
            note_date="2026-02-18",
            discipline="pt",
            soap=_sample_soap(),
            structured_data=None,
        )
        assert isinstance(result, bytes)
