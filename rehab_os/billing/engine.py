"""Billing engine — orchestrates CPT mapping, 8-minute rule, and ICD-10 suggestion.

Called from /encounter/generate to produce validated billing output.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from rehab_os.billing.cpt_mapper import CPTBillingLine, map_interventions_to_cpt
from rehab_os.billing.eight_minute_rule import UnitValidation, validate_units
from rehab_os.billing.icd10_suggester import ICD10Suggestion, suggest_icd10_codes


class BillingResult(BaseModel):
    """Complete billing output for an encounter."""

    cpt_lines: list[CPTBillingLine] = Field(default_factory=list)
    icd10_codes: list[ICD10Suggestion] = Field(default_factory=list)
    unit_validation: UnitValidation = Field(default_factory=UnitValidation)
    total_units: int = 0
    warnings: list[str] = Field(default_factory=list)


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
