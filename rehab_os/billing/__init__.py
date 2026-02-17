"""Billing module — CPT mapping, 8-minute rule, ICD-10 suggestion."""

from rehab_os.billing.cpt_mapper import map_interventions_to_cpt, CPTCode
from rehab_os.billing.eight_minute_rule import validate_units, UnitValidation
from rehab_os.billing.icd10_suggester import suggest_icd10_codes, ICD10Suggestion
from rehab_os.billing.engine import generate_billing, BillingResult

__all__ = [
    "map_interventions_to_cpt",
    "CPTCode",
    "validate_units",
    "UnitValidation",
    "suggest_icd10_codes",
    "ICD10Suggestion",
    "generate_billing",
    "BillingResult",
]
