"""Tests for the proactive alert engine — HIPAA compliance focus."""

import pytest

from rehab_os.agents.personal_agent import AgentGoal, AgentProfile, AlertPreferences
from rehab_os.agents.proactive_engine import ProactiveEngine, ProactiveAlert


@pytest.fixture
def engine():
    return ProactiveEngine()


@pytest.fixture
def profile():
    return AgentProfile(
        user_id="user_1",
        goals=[
            AgentGoal(
                title="Sign notes within 24 hours",
                metric="unsigned_notes_over_24h",
                target=0,
                current=3,
                period="daily",
            ),
            AgentGoal(
                title="Complete 20 visits",
                metric="visits_completed",
                target=20,
                current=12,
                period="weekly",
            ),
        ],
    )


class TestProactiveEngine:
    @pytest.mark.asyncio
    async def test_check_unsigned_notes_returns_count_not_names(self, engine, profile):
        alerts = await engine.check_all("user_1", profile)

        unsigned = [a for a in alerts if a.category == "compliance"]
        assert len(unsigned) >= 1

        for alert in unsigned:
            # Must use aggregate counts, never patient names
            assert "patient" not in alert.message.lower()
            assert "john" not in alert.message.lower()
            assert "doe" not in alert.message.lower()
            # Should contain a count
            assert any(c.isdigit() for c in alert.message)

    @pytest.mark.asyncio
    async def test_check_expiring_auths(self, engine, profile):
        alerts = await engine.check_all("user_1", profile)

        auth_alerts = [a for a in alerts if a.category == "auth"]
        assert len(auth_alerts) >= 1
        assert "expiring" in auth_alerts[0].message.lower()
        assert auth_alerts[0].action_href.startswith("/")

    @pytest.mark.asyncio
    async def test_check_goal_progress_under_target(self, engine, profile):
        alerts = await engine.check_all("user_1", profile)

        goal_alerts = [a for a in alerts if a.category == "goal"]
        assert len(goal_alerts) >= 1
        assert goal_alerts[0].priority == "low"
        assert "%" in goal_alerts[0].message

    @pytest.mark.asyncio
    async def test_alerts_sorted_by_priority(self, engine, profile):
        alerts = await engine.check_all("user_1", profile)

        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        priorities = [priority_order.get(a.priority, 5) for a in alerts]
        assert priorities == sorted(priorities)

    @pytest.mark.asyncio
    async def test_generate_suggestions_no_phi(self, engine, profile):
        alerts = await engine.check_all("user_1", profile)

        suggestions = [a for a in alerts if a.category == "suggestion"]
        assert len(suggestions) >= 1
        for s in suggestions:
            assert "patient" not in s.message.lower()
            assert s.icon == "sparkle"

    @pytest.mark.asyncio
    async def test_dismiss_alert(self, engine, profile):
        alerts_before = await engine.check_all("user_1", profile)
        assert len(alerts_before) > 0

        # Dismiss the first alert
        first_id = alerts_before[0].alert_id
        engine.dismiss("user_1", first_id)

        alerts_after = await engine.check_all("user_1", profile)
        ids_after = [a.alert_id for a in alerts_after]
        assert first_id not in ids_after

    @pytest.mark.asyncio
    async def test_disabled_prefs_suppress_alerts(self, engine):
        """Disabling all preferences should produce no alerts."""
        profile = AgentProfile(
            user_id="user_2",
            alert_preferences=AlertPreferences(
                unsigned_notes=False,
                expiring_auths=False,
                missed_visits=False,
                schedule_reminders=False,
                goal_progress=False,
                proactive_suggestions=False,
            ),
        )
        alerts = await engine.check_all("user_2", profile)
        assert len(alerts) == 0

    def test_alert_to_dict(self):
        alert = ProactiveAlert(
            alert_id="test_1",
            priority="high",
            category="compliance",
            message="3 unsigned notes",
        )
        data = alert.to_dict()
        assert data["alert_id"] == "test_1"
        assert data["priority"] == "high"
        assert isinstance(data["created_at"], str)
