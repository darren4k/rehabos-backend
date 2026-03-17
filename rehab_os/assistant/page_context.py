"""Page context definitions for the AI assistant."""

from __future__ import annotations

from typing import TypedDict

PAGE_CONTEXTS: dict[str, dict] = {
    "dashboard": {
        "label": "Dashboard",
        "greeting": "What can I help you with?",
        "commands": ["show_schedule", "pending_notes", "overdue_evals"],
    },
    "clinic_mode": {
        "label": "Clinic Mode",
        "greeting": "I can help improve your note or suggest interventions.",
        "commands": [
            "improve_note",
            "suggest_interventions",
            "suggest_cpt",
            "suggest_goals",
        ],
    },
    "patients": {
        "label": "Patient List",
        "greeting": "Looking for a patient or need a summary?",
        "commands": ["show_patients", "overdue_evals", "pending_notes"],
    },
    "patient_detail": {
        "label": "Patient Detail",
        "greeting": "I have context on this patient.",
        "commands": [
            "patient_summary",
            "patient_history",
            "suggest_goals",
            "draft_care_plan",
        ],
    },
    "scheduling": {
        "label": "Scheduling",
        "greeting": "Need help with the schedule?",
        "commands": ["show_schedule", "check_authorization"],
    },
    "skilled_notes": {
        "label": "Skilled Notes",
        "greeting": "I can help draft or improve your note.",
        "commands": [
            "improve_note",
            "skilled_justification",
            "documentation_tips",
            "suggest_cpt",
        ],
    },
    "rehab_program": {
        "label": "Rehab Program",
        "greeting": "Need help designing a program?",
        "commands": ["suggest_hep", "suggest_interventions", "evidence_search"],
    },
    "intake": {
        "label": "Intake",
        "greeting": "I can help process this referral.",
        "commands": ["red_flag_check", "suggest_goals", "draft_care_plan"],
    },
}


def get_page_context(page: str) -> dict:
    """Get context for a page, falling back to dashboard."""
    return PAGE_CONTEXTS.get(page, PAGE_CONTEXTS["dashboard"])
