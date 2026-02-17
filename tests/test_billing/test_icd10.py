"""Tests for ICD-10 code suggestion."""

import pytest

from rehab_os.billing.icd10_suggester import suggest_icd10_codes


class TestDiagnosisMatching:
    def test_right_knee_oa(self):
        result = suggest_icd10_codes(["right knee osteoarthritis"])
        codes = [s.code for s in result]
        assert "M17.11" in codes

    def test_left_knee_oa(self):
        result = suggest_icd10_codes(["left knee OA"])
        codes = [s.code for s in result]
        assert "M17.12" in codes

    def test_low_back_pain(self):
        result = suggest_icd10_codes(["low back pain"])
        codes = [s.code for s in result]
        assert "M54.5" in codes

    def test_cva(self):
        result = suggest_icd10_codes(["CVA"])
        codes = [s.code for s in result]
        assert "I63.9" in codes

    def test_total_knee_replacement(self):
        result = suggest_icd10_codes(["right TKA"])
        codes = [s.code for s in result]
        assert "Z96.651" in codes

    def test_rotator_cuff(self):
        result = suggest_icd10_codes(["right rotator cuff tear"])
        codes = [s.code for s in result]
        assert "M75.111" in codes

    def test_parkinsons(self):
        result = suggest_icd10_codes(["Parkinson's disease"])
        codes = [s.code for s in result]
        assert "G20" in codes

    def test_fall_risk(self):
        result = suggest_icd10_codes(["history of falls"])
        codes = [s.code for s in result]
        assert "R29.6" in codes


class TestChiefComplaintMatching:
    def test_knee_pain_from_complaint(self):
        result = suggest_icd10_codes([], chief_complaint="right knee pain")
        codes = [s.code for s in result]
        assert "M25.561" in codes

    def test_gait_deficit(self):
        result = suggest_icd10_codes([], chief_complaint="difficulty walking")
        codes = [s.code for s in result]
        assert "R26.2" in codes

    def test_balance_impairment(self):
        result = suggest_icd10_codes([], chief_complaint="unsteady on feet")
        codes = [s.code for s in result]
        assert "R26.81" in codes


class TestPainLocationMatching:
    def test_pain_location_hip(self):
        result = suggest_icd10_codes([], pain_location="right hip pain")
        codes = [s.code for s in result]
        assert "M25.551" in codes

    def test_pain_location_shoulder(self):
        result = suggest_icd10_codes([], pain_location="left shoulder pain")
        codes = [s.code for s in result]
        assert "M25.512" in codes


class TestConfidenceOrdering:
    def test_diagnosis_higher_confidence_than_complaint(self):
        result = suggest_icd10_codes(
            ["right knee osteoarthritis"],
            chief_complaint="right knee pain",
        )
        # Diagnosis match should be first (higher confidence)
        assert result[0].confidence > result[-1].confidence

    def test_empty_inputs_return_empty(self):
        result = suggest_icd10_codes([])
        assert result == []


class TestDirectICD10CodeRef:
    def test_code_in_diagnosis(self):
        result = suggest_icd10_codes(["M17.11 R knee OA"])
        codes = [s.code for s in result]
        assert "M17.11" in codes


class TestBillingEngine:
    def test_full_pipeline(self):
        from rehab_os.billing.engine import generate_billing

        result = generate_billing(
            interventions=[
                {"name": "therapeutic exercise", "duration_minutes": 15},
                {"name": "gait training", "duration_minutes": 10},
                {"name": "manual therapy", "duration_minutes": 10},
            ],
            diagnosis_list=["M17.11 R knee OA"],
            chief_complaint="right knee pain",
            pain_location="right knee",
            note_type="daily_note",
        )
        # Should have CPT codes
        assert len(result.cpt_lines) > 0
        cpt_codes = {l.code for l in result.cpt_lines}
        assert "97110" in cpt_codes
        assert "97116" in cpt_codes

        # Should have ICD-10 codes
        assert len(result.icd10_codes) > 0

        # Should have valid units
        assert result.total_units > 0
        assert result.unit_validation.total_timed_minutes == 35

    def test_empty_interventions(self):
        from rehab_os.billing.engine import generate_billing

        result = generate_billing(
            interventions=[],
            diagnosis_list=[],
        )
        assert result.total_units == 0
        assert len(result.warnings) > 0  # Should warn about missing ICD-10
