"""Proactive alert engine — generates alerts using AGGREGATE queries only.

No PHI in alerts. All messages use counts ("3 unsigned notes"),
never patient names or identifiers.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from rehab_os.agents.personal_agent import AgentGoal, AgentProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert model
# ---------------------------------------------------------------------------

@dataclass
class ProactiveAlert:
    """A single proactive alert. Message must be aggregate only — NO PHI."""

    alert_id: str = field(default_factory=lambda: f"alert_{uuid.uuid4().hex[:8]}")
    priority: str = "info"  # critical, high, medium, low, info
    category: str = "suggestion"  # compliance, schedule, auth, goal, suggestion
    message: str = ""  # NO PHI — aggregate only ("3 unsigned notes")
    action_label: str = ""  # "Review & Sign"
    action_href: str = ""  # "/skilled-notes?status=unsigned"
    icon: str = "alert"  # alert, calendar, file, target, sparkle
    dismissed: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "priority": self.priority,
            "category": self.category,
            "message": self.message,
            "action_label": self.action_label,
            "action_href": self.action_href,
            "icon": self.icon,
            "dismissed": self.dismissed,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ProactiveEngine:
    """Generates proactive alerts for a user.

    Uses AGGREGATE queries — no PHI in results.
    """

    def __init__(self) -> None:
        self._dismissed: dict[str, set[str]] = {}  # user_id -> set of dismissed alert_ids

    async def check_all(
        self, user_id: str, profile: AgentProfile
    ) -> list[ProactiveAlert]:
        """Run all enabled checks and return sorted alerts."""
        alerts: list[ProactiveAlert] = []
        prefs = profile.alert_preferences

        if prefs.unsigned_notes:
            alerts += self._check_unsigned_notes(
                user_id, prefs.unsigned_threshold_hours
            )
        if prefs.expiring_auths:
            alerts += self._check_expiring_auths(user_id, prefs.auth_threshold_days)
        if prefs.schedule_reminders:
            alerts += self._check_upcoming_visits(
                user_id, prefs.reminder_minutes_before
            )
        if prefs.goal_progress:
            alerts += self._check_goal_progress(user_id, profile.goals)
        if prefs.proactive_suggestions:
            alerts += self._generate_suggestions(user_id)

        # Filter dismissed
        dismissed = self._dismissed.get(user_id, set())
        alerts = [a for a in alerts if a.alert_id not in dismissed]

        # Sort by priority
        alerts.sort(key=lambda a: PRIORITY_ORDER.get(a.priority, 5))

        return alerts

    def dismiss(self, user_id: str, alert_id: str) -> bool:
        """Dismiss an alert for a user."""
        if user_id not in self._dismissed:
            self._dismissed[user_id] = set()
        self._dismissed[user_id].add(alert_id)
        return True

    # ----- Check implementations (aggregate queries, NO PHI) -----

    def _check_unsigned_notes(
        self, user_id: str, threshold_hours: int
    ) -> list[ProactiveAlert]:
        """Count unsigned notes — returns count only, no patient names.

        Production: ``SELECT COUNT(*) FROM clinical_notes
                      WHERE provider_id = :uid AND status = 'draft'
                      AND created_at < NOW() - INTERVAL ':hours hours'``
        """
        # Mock — production replaces with real SQL COUNT(*)
        return [
            ProactiveAlert(
                alert_id=f"unsigned_{user_id}",
                priority="high",
                category="compliance",
                message=f"3 notes unsigned for more than {threshold_hours} hours",
                action_label="Review & Sign",
                action_href="/skilled-notes?status=unsigned",
                icon="file",
            )
        ]

    def _check_expiring_auths(
        self, user_id: str, threshold_days: int
    ) -> list[ProactiveAlert]:
        """Count authorizations expiring soon — aggregate only."""
        return [
            ProactiveAlert(
                alert_id=f"auth_{user_id}",
                priority="medium",
                category="auth",
                message=f"1 authorization expiring within {threshold_days} days",
                action_label="Review Authorizations",
                action_href="/revenue?tab=auth-expiring",
                icon="alert",
            )
        ]

    def _check_upcoming_visits(
        self, user_id: str, minutes_before: int
    ) -> list[ProactiveAlert]:
        """Count today's scheduled visits — aggregate only."""
        return [
            ProactiveAlert(
                alert_id=f"schedule_{user_id}",
                priority="info",
                category="schedule",
                message="3 visits scheduled today",
                action_label="View Schedule",
                action_href="/scheduling",
                icon="calendar",
            )
        ]

    def _check_goal_progress(
        self, user_id: str, goals: list[AgentGoal]
    ) -> list[ProactiveAlert]:
        """Generate alerts for goals under 80% of target."""
        alerts: list[ProactiveAlert] = []
        for goal in goals:
            pct = (goal.current / goal.target * 100) if goal.target > 0 else 0
            if pct < 80:
                alerts.append(
                    ProactiveAlert(
                        alert_id=f"goal_{goal.goal_id}",
                        priority="low",
                        category="goal",
                        message=f"Goal '{goal.title}': {pct:.0f}% of target",
                        action_label="View Goals",
                        action_href="/goals",
                        icon="target",
                    )
                )
        return alerts

    def _generate_suggestions(self, user_id: str) -> list[ProactiveAlert]:
        """AI-driven suggestions based on aggregate context. No PHI."""
        return [
            ProactiveAlert(
                alert_id=f"suggest_{user_id}",
                priority="info",
                category="suggestion",
                message="You have evaluations today — consider reviewing your eval templates",
                action_label="Open Templates",
                action_href="/rehab-program",
                icon="sparkle",
            )
        ]


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_engine: Optional[ProactiveEngine] = None


def get_proactive_engine() -> ProactiveEngine:
    global _engine
    if _engine is None:
        _engine = ProactiveEngine()
    return _engine
