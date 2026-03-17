"""FHIR R4 code mappings for RehabOS clinical concepts.

Maps internal RehabOS enums and codes to standard FHIR coding systems
(SNOMED CT, LOINC, HL7 value sets).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sex -> FHIR AdministrativeGender (http://hl7.org/fhir/administrative-gender)
# ---------------------------------------------------------------------------
SEX_MAP: dict[str, str] = {
    "male": "male",
    "female": "female",
    "other": "other",
    "unknown": "unknown",
}

# ---------------------------------------------------------------------------
# ClinicalSetting -> FHIR Encounter.class (http://terminology.hl7.org/CodeSystem/v3-ActCode)
# ---------------------------------------------------------------------------
SETTING_CLASS_MAP: dict[str, dict[str, str]] = {
    "outpatient": {
        "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
        "code": "AMB",
        "display": "ambulatory",
    },
    "homecare": {
        "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
        "code": "HH",
        "display": "home health",
    },
    "snf": {
        "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
        "code": "IMP",
        "display": "inpatient encounter",
    },
    "irf": {
        "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
        "code": "IMP",
        "display": "inpatient encounter",
    },
    "alf": {
        "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
        "code": "AMB",
        "display": "ambulatory",
    },
    "school": {
        "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
        "code": "AMB",
        "display": "ambulatory",
    },
    "telehealth": {
        "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
        "code": "VR",
        "display": "virtual",
    },
}

# ---------------------------------------------------------------------------
# Discipline -> SNOMED CT specialty codes
# ---------------------------------------------------------------------------
DISCIPLINE_SNOMED: dict[str, dict[str, str]] = {
    "pt": {"code": "36682004", "display": "Physiotherapy"},
    "ot": {"code": "80546007", "display": "Occupational therapy"},
    "slp": {"code": "159026005", "display": "Speech and language therapy"},
}

# ---------------------------------------------------------------------------
# Encounter status -> FHIR EncounterStatus
# (http://hl7.org/fhir/encounter-status)
# ---------------------------------------------------------------------------
ENCOUNTER_STATUS_MAP: dict[str, str] = {
    "scheduled": "planned",
    "in_progress": "in-progress",
    "completed": "finished",
    "cancelled": "cancelled",
    "no_show": "cancelled",
    "checked_in": "arrived",
}

# ---------------------------------------------------------------------------
# Outcome measures -> LOINC codes
# Real LOINC codes for rehab outcome instruments
# ---------------------------------------------------------------------------
MEASURE_LOINC: dict[str, dict[str, str]] = {
    # PT - Orthopedic
    "LEFS": {"code": "72100-1", "display": "LEFS total score"},
    "ODI": {"code": "89195-5", "display": "Oswestry Disability Index total score"},
    "NDI": {"code": "72101-9", "display": "Neck Disability Index total score"},
    "DASH": {"code": "72102-7", "display": "DASH total score"},
    "SPADI": {"code": "72103-5", "display": "SPADI total score"},
    "KOOS": {"code": "72104-3", "display": "KOOS total score"},
    # PT - Neuro/Balance
    "Berg": {"code": "52737-0", "display": "Berg Balance Scale total score"},
    "TUG": {"code": "54821-4", "display": "Timed Up and Go"},
    "6MWT": {"code": "64098-7", "display": "Six minute walk test distance"},
    "DGI": {"code": "52738-8", "display": "Dynamic Gait Index total score"},
    "FGA": {"code": "52739-6", "display": "Functional Gait Assessment total score"},
    "ABC": {"code": "52740-4", "display": "Activities-specific Balance Confidence Scale"},
    # PT - Pain
    "NPRS": {"code": "72514-3", "display": "Pain severity - Numeric rating scale"},
    "PSFS": {"code": "72105-0", "display": "Patient-Specific Functional Scale"},
    # OT
    "QuickDASH": {"code": "72106-8", "display": "Quick DASH total score"},
    "FIM": {"code": "54614-3", "display": "FIM instrument total score"},
    "Barthel": {"code": "96761-2", "display": "Barthel Index total score"},
    "COPM": {"code": "72107-6", "display": "COPM performance score"},
    # SLP
    "FOIS": {"code": "72108-4", "display": "Functional Oral Intake Scale level"},
    "ASHA_NOMS": {"code": "72109-2", "display": "ASHA NOMS FCM level"},
    "VHI": {"code": "72110-0", "display": "Voice Handicap Index total score"},
    # Global
    "GRC": {"code": "72111-8", "display": "Global Rating of Change score"},
}

# ---------------------------------------------------------------------------
# Clinical note type -> LOINC document codes
# ---------------------------------------------------------------------------
NOTE_TYPE_LOINC: dict[str, dict[str, str]] = {
    "evaluation": {"code": "34117-2", "display": "History and physical note"},
    "initial_eval": {"code": "34117-2", "display": "History and physical note"},
    "soc_eval": {"code": "34117-2", "display": "History and physical note"},
    "daily_note": {"code": "34108-1", "display": "Outpatient note"},
    "progress_note": {"code": "11506-3", "display": "Progress note"},
    "recertification": {"code": "11506-3", "display": "Progress note"},
    "recert": {"code": "11506-3", "display": "Progress note"},
    "discharge_summary": {"code": "18842-5", "display": "Discharge summary"},
    "discharge": {"code": "18842-5", "display": "Discharge summary"},
}

# ---------------------------------------------------------------------------
# Encounter type -> SNOMED procedure codes
# ---------------------------------------------------------------------------
ENCOUNTER_TYPE_SNOMED: dict[str, dict[str, str]] = {
    "evaluation": {"code": "410620009", "display": "Assessment (procedure)"},
    "treatment": {"code": "277132007", "display": "Therapeutic procedure"},
    "re_evaluation": {"code": "410620009", "display": "Assessment (procedure)"},
    "discharge": {"code": "58000006", "display": "Patient discharge"},
    "screening": {"code": "171207006", "display": "Screening procedure"},
}

# ---------------------------------------------------------------------------
# CPT system URI
# ---------------------------------------------------------------------------
CPT_SYSTEM = "http://www.ama-assn.org/go/cpt"
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"
SNOMED_SYSTEM = "http://snomed.info/sct"
LOINC_SYSTEM = "http://loinc.org"
