"""Billing engine — orchestrates CPT mapping, 8-minute rule, and ICD-10 suggestion.

Called from /encounter/generate to produce validated billing output.

Includes payer-specific validation merged from DocPilot: max units per day,
prior authorization flags, modifier validation (GP/GO/GN), and medical
necessity language checks.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from rehab_os.billing.cpt_mapper import CPTBillingLine, map_interventions_to_cpt
from rehab_os.billing.eight_minute_rule import UnitValidation, validate_units
from rehab_os.billing.icd10_suggester import ICD10Suggestion, suggest_icd10_codes

logger = logging.getLogger(__name__)


# ============================================================================
# Payer-Specific Rules (merged from DocPilot)
# ============================================================================

PAYER_RULES: dict[str, dict[str, Any]] = {
    "Medicare": {
        "max_units_per_day": 8,
        "requires_medical_necessity": True,
        "progress_note_frequency": 10,  # every 10 visits
        "therapy_cap_applies": True,
    },
    "HMO": {
        "max_units_per_day": 6,
        "requires_authorization": True,
        "requires_medical_necessity": True,
    },
    "Commercial": {
        "max_units_per_day": 8,
        "requires_medical_necessity": True,
    },
    "Medicaid": {
        "max_units_per_day": 6,
        "requires_prior_auth": True,
        "requires_medical_necessity": True,
    },
}

# Discipline-specific modifiers per CMS
DISCIPLINE_MODIFIERS: dict[str, str] = {
    "PT": "GP",   # Physical Therapy
    "OT": "GO",   # Occupational Therapy
    "SLP": "GN",  # Speech-Language Pathology
}

# Medical necessity keywords that should appear in assessment
MEDICAL_NECESSITY_KEYWORDS = [
    "medically necessary",
    "skilled",
    "requires therapist",
    "safety risk",
    "fall risk",
    "functional deficit",
    "unable to perform",
    "clinical judgment",
]


# ============================================================================
# Data Models
# ============================================================================

class BillingResult(BaseModel):
    """Complete billing output for an encounter."""

    cpt_lines: list[CPTBillingLine] = Field(default_factory=list)
    icd10_codes: list[ICD10Suggestion] = Field(default_factory=list)
    unit_validation: UnitValidation = Field(default_factory=UnitValidation)
    total_units: int = 0
    warnings: list[str] = Field(default_factory=list)


class PayerValidationResult(BaseModel):
    """Result of payer-specific billing validation."""

    is_valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    modifier: Optional[str] = None
    requires_prior_auth: bool = False


# ============================================================================
# Core Billing Generation (unchanged interface)
# ============================================================================

def generate_billing(
    interventions: list[dict],
    diagnosis_list: list[str],
    chief_complaint: str | None = None,
    pain_location: str | None = None,
    note_type: str = "daily_note",
) -> BillingResult:
    """Generate complete billing from encounter data.

    Args:
        interventions: List of intervention dicts from EncounterState.
        diagnosis_list: Known diagnoses from patient history.
        chief_complaint: Current chief complaint.
        pain_location: Documented pain location.
        note_type: Type of clinical note.

    Returns:
        BillingResult with CPT codes, ICD-10 suggestions, and validation.
    """
    # 1. Map interventions to CPT codes
    cpt_lines = map_interventions_to_cpt(interventions, note_type)

    # 2. Validate units with 8-minute rule
    unit_validation = validate_units(cpt_lines)

    # 3. Suggest ICD-10 codes
    icd10_codes = suggest_icd10_codes(
        diagnosis_list=diagnosis_list,
        chief_complaint=chief_complaint,
        pain_location=pain_location,
    )

    # Collect all warnings
    warnings = list(unit_validation.warnings)

    # Warn if no ICD-10 codes found
    if not icd10_codes:
        warnings.append(
            "No ICD-10 codes could be auto-suggested. "
            "Ensure diagnosis is documented in patient history."
        )

    # Warn if CPT codes without matching ICD-10
    if cpt_lines and not icd10_codes:
        warnings.append(
            "CPT codes generated but no ICD-10 codes to pair. "
            "Claims require at least one ICD-10 code."
        )

    return BillingResult(
        cpt_lines=unit_validation.billing_lines,
        icd10_codes=icd10_codes,
        unit_validation=unit_validation,
        total_units=unit_validation.total_units,
        warnings=warnings,
    )


# ============================================================================
# Payer-Specific Validation (merged from DocPilot)
# ============================================================================

def validate_billing(
    cpt_codes: list[dict[str, Any]],
    total_timed_minutes: int,
    payer_type: str = "Medicare",
    discipline: str = "PT",
    assessment_text: Optional[str] = None,
    setting: str = "outpatient",
) -> PayerValidationResult:
    """Validate billing against payer-specific rules.

    Checks beyond the 8-minute rule: payer max units, modifier
    requirements, prior auth flags, medical necessity language.

    Args:
        cpt_codes: List of dicts with keys: cpt, units, minutes.
        total_timed_minutes: Total timed minutes for the encounter.
        payer_type: Medicare, HMO, Commercial, or Medicaid.
        discipline: PT, OT, or SLP.
        assessment_text: Optional assessment section text for
            medical necessity language check.
        setting: outpatient, homecare, or snf.

    Returns:
        PayerValidationResult with errors, warnings, and suggestions.
    """
    errors: list[str] = []
    warnings: list[str] = []
    suggestions: list[str] = []

    total_units = sum(c.get("units", 0) for c in cpt_codes)
    payer_rules = PAYER_RULES.get(payer_type, {})

    # 1. Payer max units per day
    max_units = payer_rules.get("max_units_per_day")
    if max_units and total_units > max_units:
        errors.append(
            f"{payer_type} typically allows max {max_units} units/day. "
            f"Billed: {total_units} units."
        )

    # 2. Prior authorization flags
    requires_prior_auth = bool(
        payer_rules.get("requires_authorization")
        or payer_rules.get("requires_prior_auth")
    )
    if requires_prior_auth:
        suggestions.append(
            f"{payer_type} may require prior authorization. Verify coverage."
        )

    # 3. Discipline modifier
    modifier = DISCIPLINE_MODIFIERS.get(discipline.upper())
    if modifier:
        suggestions.append(
            f"Append modifier -{modifier} to therapy CPT codes for "
            f"{discipline} billing."
        )

    # 4. High manual therapy flag (audit risk)
    manual_codes = [c for c in cpt_codes if c.get("cpt") == "97140"]
    manual_units = sum(c.get("units", 0) for c in manual_codes)
    if manual_units > 4:
        warnings.append(
            "High manual therapy units (>4) may require additional "
            "documentation of medical necessity."
        )

    # 5. Medical necessity language check
    if assessment_text and payer_rules.get("requires_medical_necessity"):
        assessment_lower = assessment_text.lower()
        has_necessity = any(
            kw in assessment_lower for kw in MEDICAL_NECESSITY_KEYWORDS
        )
        if not has_necessity:
            warnings.append(
                "Assessment section may lack medical necessity language. "
                "Consider adding: 'Skilled therapy is medically necessary "
                "to address [deficit] and reduce [risk].'"
            )

    # 6. SNF-specific: check minutes documentation for PDPM
    if setting == "snf":
        suggestions.append(
            "SNF setting: ensure minutes are documented per discipline "
            "for PDPM compliance."
        )

    # 7. Homecare: remind about homebound status
    if setting == "homecare":
        suggestions.append(
            "Home health: ensure homebound status is documented and "
            "physician order is on file."
        )

    return PayerValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        suggestions=suggestions,
        modifier=modifier,
        requires_prior_auth=requires_prior_auth,
    )
