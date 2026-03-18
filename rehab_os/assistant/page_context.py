"""Page context definitions for the AI assistant."""

from __future__ import annotations

PAGE_CONTEXTS: dict[str, dict] = {
    "dashboard": {
        "label": "Dashboard",
        "greeting": "What can I help you with?",
        "commands": ["show_schedule", "pending_notes", "overdue_evals", "run_compliance_check"],
    },
    "clinic_mode": {
        "label": "Clinic Mode",
        "greeting": "I can help improve your note or suggest interventions.",
        "commands": [
            "improve_note",
            "suggest_interventions",
            "suggest_cpt",
            "suggest_goals",
            "dictate_note",
            "complete_visit",
        ],
    },
    "patients": {
        "label": "Patient List",
        "greeting": "Looking for a patient or need a summary?",
        "commands": ["show_patients", "set_patient", "overdue_evals", "pending_notes", "onboard_patient"],
    },
    "patient_detail": {
        "label": "Patient Detail",
        "greeting": "I have context on this patient.",
        "commands": [
            "patient_summary",
            "patient_history",
            "patient_timeline",
            "suggest_goals",
            "draft_care_plan",
            "add_diagnosis",
            "add_goal",
            "send_to_md",
        ],
    },
    "scheduling": {
        "label": "Scheduling",
        "greeting": "Need help with the schedule?",
        "commands": ["show_schedule", "schedule_visit", "cancel_visit", "check_authorization"],
    },
    "skilled_notes": {
        "label": "Skilled Notes",
        "greeting": "I can help draft or improve your note.",
        "commands": [
            "improve_note",
            "skilled_justification",
            "documentation_tips",
            "suggest_cpt",
            "sign_note",
            "dictate_note",
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
        "commands": ["red_flag_check", "suggest_goals", "draft_care_plan", "onboard_patient"],
    },
    "billing": {
        "label": "Billing",
        "greeting": "Need help with billing codes or units?",
        "commands": ["suggest_cpt", "eight_min_rule", "check_authorization"],
    },
    "reports": {
        "label": "Reports",
        "greeting": "What report or analytics do you need?",
        "commands": ["run_compliance_check", "peer_comparison", "overdue_evals"],
    },
    "messages": {
        "label": "Messages",
        "greeting": "Need to send a message or fax?",
        "commands": ["send_message", "send_fax", "send_to_md"],
    },
}


ROLE_COMMANDS: dict[str, list[dict]] = {
    "owner": [
        {"text": "Revenue Report", "cmd": "show revenue dashboard"},
        {"text": "Team Stats", "cmd": "show team performance"},
        {"text": "Compliance Check", "cmd": "run compliance check"},
        {"text": "All Unsigned Notes", "cmd": "show all unsigned notes"},
    ],
    "admin": [
        {"text": "Compliance Check", "cmd": "run compliance check"},
        {"text": "Overdue Evals", "cmd": "show overdue evals"},
        {"text": "Pending Notes", "cmd": "show pending notes"},
        {"text": "Onboard Patient", "cmd": "onboard patient"},
    ],
    "therapist": [
        {"text": "My Schedule", "cmd": "my schedule today"},
        {"text": "Unsigned Notes", "cmd": "show unsigned notes"},
        {"text": "Start Session", "cmd": "start session with"},
        {"text": "Dictate Note", "cmd": "dictate note"},
    ],
    "assistant": [
        {"text": "Schedule Visit", "cmd": "schedule visit"},
        {"text": "Check Auth", "cmd": "check authorization"},
        {"text": "Onboard Patient", "cmd": "onboard patient"},
        {"text": "Send Fax", "cmd": "send fax to"},
    ],
}


def get_page_context(page: str) -> dict:
    """Get context for a page, falling back to dashboard."""
    return PAGE_CONTEXTS.get(page, PAGE_CONTEXTS["dashboard"])


def get_role_commands(role: str) -> list[dict]:
    """Get role-specific quick commands."""
    return ROLE_COMMANDS.get(role, ROLE_COMMANDS.get("therapist", []))
