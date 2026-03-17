"""Tests for rehab_os.clinical.prompts — SOAP prompt generation and QA."""
from __future__ import annotations

import pytest

from rehab_os.clinical.prompts import (
    DIAGNOSIS_TEMPLATES,
    INTERVENTION_TEMPLATES,
    PROHIBITED_PHRASES,
    SETTING_ADDENDUM,
    SKILLED_REPLACEMENTS,
    check_prohibited_phrases,
    get_diagnosis_template,
    get_discipline_codes,
    get_intervention_template,
    get_skilled_replacement,
    get_soap_system_prompt,
)


# ---------------------------------------------------------------------------
# System prompt generation
# ---------------------------------------------------------------------------

class TestGetSoapSystemPrompt:
    def test_pt_medicare(self):
        prompt = get_soap_system_prompt(discipline="PT", payer="Medicare", setting="outpatient")
        assert "PT" in prompt
        assert "Medicare" in prompt
        assert "PROHIBITED PHRASES" in prompt
        assert "97110" in prompt  # PT CPT code

    def test_homecare_includes_oasis_addendum(self):
        prompt = get_soap_system_prompt(discipline="PT", payer="Medicare", setting="homecare")
        assert "OASIS" in prompt
        assert "homebound" in prompt.lower()
        assert "60-day" in prompt

    def test_snf_includes_fim_addendum(self):
        prompt = get_soap_system_prompt(discipline="PT", payer="Medicare", setting="snf")
        assert "FIM" in prompt
        assert "PDPM" in prompt

    def test_ot_codes(self):
        prompt = get_soap_system_prompt(discipline="OT", payer="Medicare")
        assert "97535" in prompt  # OT-specific code

    def test_slp_codes(self):
        prompt = get_soap_system_prompt(discipline="SLP", payer="Medicare")
        assert "92507" in prompt  # SLP-specific code

    def test_hmo_payer(self):
        prompt = get_soap_system_prompt(payer="HMO")
        assert "authorization" in prompt.lower()


# ---------------------------------------------------------------------------
# Prohibited phrase checking
# ---------------------------------------------------------------------------

class TestCheckProhibitedPhrases:
    def test_detects_violation(self):
        text = "The patient tolerated treatment well and continues to benefit from therapy."
        found = check_prohibited_phrases(text)
        assert "tolerated treatment well" in found
        assert "continues to benefit" in found

    def test_clean_text(self):
        text = (
            "Patient demonstrated improved weight shift to R LE with tactile "
            "cueing during sit-to-stand transfers. ROM flex improved 80->95 degrees."
        )
        found = check_prohibited_phrases(text)
        assert found == []

    def test_case_insensitive(self):
        text = "Patient TOLERATED TREATMENT WELL today."
        found = check_prohibited_phrases(text)
        assert len(found) >= 1

    def test_multiple_violations(self):
        text = "Good session. Patient doing well. Stable. No complaints."
        found = check_prohibited_phrases(text)
        assert len(found) >= 3


# ---------------------------------------------------------------------------
# Skilled replacements
# ---------------------------------------------------------------------------

class TestGetSkilledReplacement:
    def test_known_phrase(self):
        replacement = get_skilled_replacement("tolerated treatment well")
        assert "specific response" in replacement.lower() or "cueing" in replacement.lower()

    def test_unknown_phrase(self):
        replacement = get_skilled_replacement("some random phrase")
        assert "WHAT/WHY/HOW" in replacement


# ---------------------------------------------------------------------------
# Diagnosis templates
# ---------------------------------------------------------------------------

class TestGetDiagnosisTemplate:
    def test_stroke_by_keyword(self):
        template = get_diagnosis_template("CVA - left MCA stroke")
        assert template is not None
        assert template["name"] == "CVA/Stroke"

    def test_parkinsons_by_keyword(self):
        template = get_diagnosis_template("Parkinson's disease")
        assert template is not None
        assert "Bradykinesia" in template["common_deficits"]

    def test_by_icd10_pattern(self):
        template = get_diagnosis_template("I63.9")
        assert template is not None
        assert template["name"] == "CVA/Stroke"

    def test_no_match(self):
        template = get_diagnosis_template("earwax impaction")
        assert template is None

    def test_fall_risk(self):
        template = get_diagnosis_template("History of falls, unsteady gait")
        assert template is not None
        assert "Fall" in template["name"]


# ---------------------------------------------------------------------------
# Intervention templates
# ---------------------------------------------------------------------------

class TestGetInterventionTemplate:
    def test_therapeutic_exercise(self):
        template = get_intervention_template("97110")
        assert template is not None
        assert template["name"] == "Therapeutic Exercise"
        assert len(template["skilled_documentation"]) > 0

    def test_gait_training(self):
        template = get_intervention_template("97116")
        assert template is not None
        assert template["name"] == "Gait Training"

    def test_unknown_code(self):
        template = get_intervention_template("99999")
        assert template is None

    def test_manual_therapy(self):
        template = get_intervention_template("97140")
        assert template is not None
        assert "Manual Therapy" in template["name"]
