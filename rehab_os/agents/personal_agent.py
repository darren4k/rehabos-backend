"""Personalized AI agent profiles — HIPAA-compliant.

Agent memory stores NO PHI. Only aggregate metrics, user preferences,
custom skills, and goals are persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CustomSkill:
    """User-defined skill that the agent can execute. NO PHI in template."""

    skill_id: str = field(default_factory=lambda: f"skill_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    trigger: str = ""  # When to activate
    prompt_template: str = ""  # What the skill does (NO PHI in template)
    output_format: str = "text"  # text, care_plan, soap_section, alert
    enabled: bool = True
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class AgentGoal:
    """Trackable performance goal — metric values are aggregate numbers only."""

    goal_id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:8]}")
    title: str = ""
    description: str = ""
    metric: str = ""  # "unsigned_notes", "visits_completed", etc.
    target: float = 0
    current: float = 0
    period: str = "weekly"  # daily, weekly, monthly
    category: str = "compliance"  # compliance, productivity, clinical, revenue
    streak: int = 0  # Consecutive periods meeting goal
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class AlertPreferences:
    """User-configurable alert thresholds. No PHI."""

    unsigned_notes: bool = True
    unsigned_threshold_hours: int = 24
    expiring_auths: bool = True
    auth_threshold_days: int = 7
    missed_visits: bool = True
    schedule_reminders: bool = True
    reminder_minutes_before: int = 15
    goal_progress: bool = True
    proactive_suggestions: bool = True


@dataclass
class AgentProfile:
    """Personalized AI agent profile. Stores NO PHI — only preferences and aggregate metrics."""

    user_id: str
    agent_name: str = "Clinical Assistant"
    role: str = "pt"  # pt, ot, slp, admin, owner
    personality: str = "professional"  # professional, friendly, detailed, minimal

    skills: list[str] = field(
        default_factory=lambda: [
            "documentation",
            "billing",
            "evidence",
            "scheduling",
            "compliance",
        ]
    )
    custom_skills: list[CustomSkill] = field(default_factory=list)

    tools: list[str] = field(
        default_factory=lambda: ["fax", "email", "calendar", "emr", "search"]
    )

    goals: list[AgentGoal] = field(default_factory=list)
    alert_preferences: AlertPreferences = field(default_factory=AlertPreferences)

    # Usage patterns (aggregate, NO PHI)
    usage_patterns: dict = field(
        default_factory=lambda: {
            "peak_hours": [],
            "common_commands": [],
            "avg_session_minutes": 0,
            "total_tasks_completed": 0,
        }
    )

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ----- Helpers -----

    def get_system_prompt(self) -> str:
        """Generate the agent's system prompt based on profile. Contains NO PHI."""

        personality_map = {
            "professional": "Be concise, clinical, and direct.",
            "friendly": "Be warm and encouraging while remaining professional.",
            "detailed": "Provide thorough explanations with clinical reasoning.",
            "minimal": "Be extremely brief — bullet points and short answers only.",
        }

        role_map = {
            "pt": "physical therapist",
            "ot": "occupational therapist",
            "slp": "speech-language pathologist",
            "admin": "clinic administrator",
            "owner": "practice owner",
        }

        skills_desc = ", ".join(self.skills)
        tools_desc = ", ".join(self.tools)

        custom_block = ""
        active_customs = [s for s in self.custom_skills if s.enabled]
        if active_customs:
            custom_lines = "\n".join(
                f"  - {s.name}: {s.description} (trigger: {s.trigger})"
                for s in active_customs
            )
            custom_block = f"\nCustom skills:\n{custom_lines}\n"

        goal_block = ""
        if self.goals:
            goal_lines = "\n".join(
                f"  - {g.title}: {g.current}/{g.target} ({g.period})"
                for g in self.goals
            )
            goal_block = f"\nActive goals:\n{goal_lines}\n"

        return (
            f"You are {self.agent_name}, a personalized AI assistant for a "
            f"{role_map.get(self.role, 'healthcare professional')}.\n\n"
            f"Personality: {personality_map.get(self.personality, personality_map['professional'])}\n\n"
            f"Your skills: {skills_desc}\n"
            f"Your tools: {tools_desc}\n"
            f"{custom_block}"
            f"{goal_block}"
            "\nCRITICAL RULES:\n"
            "- You NEVER store patient data in your memory\n"
            "- You access patient data ONLY when explicitly asked for a specific task\n"
            "- After completing a task involving patient data, you forget the details\n"
            "- You suggest and draft — the clinician approves all clinical decisions\n"
            "- Proactive alerts use aggregate counts only (e.g., '3 unsigned notes'), never patient names\n"
            "- All patient data access is logged to the HIPAA audit trail\n"
        )

    def to_dict(self) -> dict:
        """Serialize to dict for API responses."""
        return {
            "user_id": self.user_id,
            "agent_name": self.agent_name,
            "role": self.role,
            "personality": self.personality,
            "skills": self.skills,
            "custom_skills": [
                {
                    "skill_id": s.skill_id,
                    "name": s.name,
                    "description": s.description,
                    "trigger": s.trigger,
                    "prompt_template": s.prompt_template,
                    "output_format": s.output_format,
                    "enabled": s.enabled,
                    "created_at": s.created_at,
                }
                for s in self.custom_skills
            ],
            "tools": self.tools,
            "goals": [
                {
                    "goal_id": g.goal_id,
                    "title": g.title,
                    "description": g.description,
                    "metric": g.metric,
                    "target": g.target,
                    "current": g.current,
                    "period": g.period,
                    "category": g.category,
                    "streak": g.streak,
                    "created_at": g.created_at,
                }
                for g in self.goals
            ],
            "alert_preferences": {
                "unsigned_notes": self.alert_preferences.unsigned_notes,
                "unsigned_threshold_hours": self.alert_preferences.unsigned_threshold_hours,
                "expiring_auths": self.alert_preferences.expiring_auths,
                "auth_threshold_days": self.alert_preferences.auth_threshold_days,
                "missed_visits": self.alert_preferences.missed_visits,
                "schedule_reminders": self.alert_preferences.schedule_reminders,
                "reminder_minutes_before": self.alert_preferences.reminder_minutes_before,
                "goal_progress": self.alert_preferences.goal_progress,
                "proactive_suggestions": self.alert_preferences.proactive_suggestions,
            },
            "usage_patterns": self.usage_patterns,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# In-memory store (production: replace with database-backed repository)
# ---------------------------------------------------------------------------

class AgentProfileStore:
    """In-memory store for agent profiles. Production: use database."""

    def __init__(self) -> None:
        self._profiles: dict[str, AgentProfile] = {}

    def get_or_create(self, user_id: str, role: str = "pt") -> AgentProfile:
        if user_id not in self._profiles:
            self._profiles[user_id] = AgentProfile(
                user_id=user_id,
                role=role,
                goals=self._default_goals(),
            )
        return self._profiles[user_id]

    def update(self, user_id: str, **kwargs) -> AgentProfile:
        profile = self.get_or_create(user_id)
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        return profile

    def add_custom_skill(self, user_id: str, skill: CustomSkill) -> AgentProfile:
        profile = self.get_or_create(user_id)
        profile.custom_skills.append(skill)
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        return profile

    def remove_custom_skill(self, user_id: str, skill_id: str) -> bool:
        profile = self.get_or_create(user_id)
        before = len(profile.custom_skills)
        profile.custom_skills = [
            s for s in profile.custom_skills if s.skill_id != skill_id
        ]
        if len(profile.custom_skills) < before:
            profile.updated_at = datetime.now(timezone.utc).isoformat()
            return True
        return False

    def add_goal(self, user_id: str, goal: AgentGoal) -> AgentProfile:
        profile = self.get_or_create(user_id)
        profile.goals.append(goal)
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        return profile

    def remove_goal(self, user_id: str, goal_id: str) -> bool:
        profile = self.get_or_create(user_id)
        before = len(profile.goals)
        profile.goals = [g for g in profile.goals if g.goal_id != goal_id]
        if len(profile.goals) < before:
            profile.updated_at = datetime.now(timezone.utc).isoformat()
            return True
        return False

    @staticmethod
    def _default_goals() -> list[AgentGoal]:
        return [
            AgentGoal(
                title="Sign notes within 24 hours",
                metric="unsigned_notes_over_24h",
                target=0,
                period="daily",
                category="compliance",
            ),
            AgentGoal(
                title="Complete 20 visits per week",
                metric="visits_completed",
                target=20,
                period="weekly",
                category="productivity",
            ),
            AgentGoal(
                title="95% documentation compliance",
                metric="doc_compliance_pct",
                target=95,
                period="weekly",
                category="compliance",
            ),
        ]


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_store: Optional[AgentProfileStore] = None


def get_agent_store() -> AgentProfileStore:
    global _store
    if _store is None:
        _store = AgentProfileStore()
    return _store
