"""Setting-aware clinical configuration for RehabOS.

Each ClinicalSetting defines note types, billing forms, documentation
standards, and regulatory requirements specific to that care environment.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class ClinicalSetting(str, Enum):
    OUTPATIENT = "outpatient"
    HOMECARE = "homecare"
    SNF = "snf"           # Skilled Nursing Facility
    IRF = "irf"           # Inpatient Rehab Facility
    ALF = "alf"           # Assisted Living Facility
    SCHOOL = "school"     # Pediatric school-based
    TELEHEALTH = "telehealth"


# Per-setting configuration
SETTING_CONFIG: dict[ClinicalSetting, dict[str, Any]] = {
    ClinicalSetting.OUTPATIENT: {
        "note_types": [
            "initial_eval", "progress_note", "daily_note",
            "discharge", "recert",
        ],
        "billing_form": "CMS-1500",
        "requires_oasis": False,
        "requires_fim": False,
        "documentation_standard": "APTA",
        "visit_frequency_unit": "visits_per_week",
        "max_episode_days": 90,
        "scheduling_type": "clinic_slots",
    },
    ClinicalSetting.HOMECARE: {
        "note_types": [
            "soc_eval", "recert_eval", "roc_eval",
            "progress_note", "discharge", "supervisory_visit",
        ],
        "billing_form": "CMS-1500",
        "requires_oasis": True,
        "requires_fim": False,
        "documentation_standard": "CMS_HH",
        "visit_frequency_unit": "visits_per_60day_cert",
        "max_episode_days": 60,
        "scheduling_type": "route_based",
        "requires_physician_order": True,
        "cert_period_days": 60,
    },
    ClinicalSetting.SNF: {
        "note_types": [
            "initial_eval", "progress_note", "daily_note",
            "discharge", "weekly_summary",
        ],
        "billing_form": "UB-04",
        "requires_oasis": False,
        "requires_fim": True,
        "documentation_standard": "CMS_SNF",
        "visit_frequency_unit": "minutes_per_week",
        "max_episode_days": 100,  # Medicare Part A
        "scheduling_type": "facility_schedule",
    },
    ClinicalSetting.IRF: {
        "note_types": [
            "initial_eval", "progress_note", "daily_note",
            "discharge", "team_conference",
        ],
        "billing_form": "UB-04",
        "requires_oasis": False,
        "requires_fim": True,
        "documentation_standard": "CMS_IRF",
        "visit_frequency_unit": "hours_per_day",
        "max_episode_days": 60,
        "scheduling_type": "facility_schedule",
        "min_therapy_hours_per_week": 15,  # 3 hrs/day, 5 days
        "requires_physician_order": True,
        "requires_preadmission_screen": True,
    },
    ClinicalSetting.ALF: {
        "note_types": [
            "initial_eval", "progress_note", "daily_note",
            "discharge", "monthly_summary",
        ],
        "billing_form": "CMS-1500",
        "requires_oasis": False,
        "requires_fim": False,
        "documentation_standard": "APTA",
        "visit_frequency_unit": "visits_per_week",
        "max_episode_days": 90,
        "scheduling_type": "facility_schedule",
    },
    ClinicalSetting.SCHOOL: {
        "note_types": [
            "initial_eval", "progress_note", "daily_note",
            "discharge", "iep_report",
        ],
        "billing_form": "CMS-1500",
        "requires_oasis": False,
        "requires_fim": False,
        "documentation_standard": "IDEA",
        "visit_frequency_unit": "sessions_per_iep_period",
        "max_episode_days": 365,  # IEP annual cycle
        "scheduling_type": "school_schedule",
        "requires_iep": True,
    },
    ClinicalSetting.TELEHEALTH: {
        "note_types": [
            "initial_eval", "progress_note", "discharge",
        ],
        "billing_form": "CMS-1500",
        "requires_oasis": False,
        "requires_fim": False,
        "documentation_standard": "APTA",
        "visit_frequency_unit": "visits_per_week",
        "max_episode_days": 90,
        "scheduling_type": "virtual_slots",
        "modifier": "95",  # Synchronous telehealth modifier
        "requires_consent": True,
    },
}


def get_setting_config(setting: ClinicalSetting) -> dict[str, Any]:
    """Return configuration for a clinical setting, defaulting to outpatient."""
    return SETTING_CONFIG.get(setting, SETTING_CONFIG[ClinicalSetting.OUTPATIENT])


def get_valid_note_types(setting: ClinicalSetting) -> list[str]:
    """Return valid note types for a given setting."""
    config = get_setting_config(setting)
    return config["note_types"]


def requires_instrument(setting: ClinicalSetting, instrument: str) -> bool:
    """Check if a setting requires a specific instrument (e.g., OASIS, FIM)."""
    config = get_setting_config(setting)
    mapping = {
        "oasis": "requires_oasis",
        "fim": "requires_fim",
        "iep": "requires_iep",
        "preadmission_screen": "requires_preadmission_screen",
    }
    key = mapping.get(instrument.lower())
    if key:
        return config.get(key, False)
    return False
