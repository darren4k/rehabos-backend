"""Tests for personalized AI agent profiles — HIPAA compliance focus."""

import pytest

from rehab_os.agents.personal_agent import (
    AgentGoal,
    AgentProfile,
    AgentProfileStore,
    AlertPreferences,
    CustomSkill,
    get_agent_store,
)


class TestAgentProfile:
    def test_create_default_profile(self):
        store = AgentProfileStore()
        profile = store.get_or_create("user_1")

        assert profile.user_id == "user_1"
        assert profile.agent_name == "Clinical Assistant"
        assert profile.role == "pt"
        assert profile.personality == "professional"
        assert "documentation" in profile.skills
        assert "fax" in profile.tools
        assert profile.created_at is not None
        assert profile.updated_at is not None

    def test_get_system_prompt_no_phi(self):
        profile = AgentProfile(user_id="user_1", role="pt")

        prompt = profile.get_system_prompt()

        assert "NEVER store patient data" in prompt
        assert "aggregate counts only" in prompt
        assert "HIPAA audit trail" in prompt
        assert "physical therapist" in prompt
        # Must not contain any placeholder that could hold PHI
        assert "patient_id" not in prompt.lower()
        assert "ssn" not in prompt.lower()
        assert "mrn" not in prompt.lower()

    def test_add_custom_skill(self):
        store = AgentProfileStore()
        skill = CustomSkill(
            name="Quick SOAP",
            description="Generate SOAP note template",
            trigger="start_note",
            prompt_template="Create a SOAP note structure for the current encounter.",
        )
        profile = store.add_custom_skill("user_1", skill)

        assert len(profile.custom_skills) == 1
        assert profile.custom_skills[0].name == "Quick SOAP"
        assert profile.custom_skills[0].skill_id.startswith("skill_")

    def test_remove_custom_skill(self):
        store = AgentProfileStore()
        skill = CustomSkill(name="Temp Skill")
        store.add_custom_skill("user_1", skill)

        removed = store.remove_custom_skill("user_1", skill.skill_id)
        assert removed is True

        profile = store.get_or_create("user_1")
        assert len(profile.custom_skills) == 0

    def test_remove_nonexistent_skill(self):
        store = AgentProfileStore()
        store.get_or_create("user_1")
        removed = store.remove_custom_skill("user_1", "skill_nonexistent")
        assert removed is False

    def test_add_goal(self):
        store = AgentProfileStore()
        goal = AgentGoal(
            title="Reduce turnaround",
            metric="note_turnaround_hours",
            target=8,
            period="weekly",
            category="productivity",
        )
        profile = store.add_goal("user_1", goal)

        # 3 defaults + 1 new
        assert len(profile.goals) == 4
        assert profile.goals[-1].title == "Reduce turnaround"

    def test_default_goals_created(self):
        store = AgentProfileStore()
        profile = store.get_or_create("user_1")

        assert len(profile.goals) == 3
        metrics = [g.metric for g in profile.goals]
        assert "unsigned_notes_over_24h" in metrics
        assert "visits_completed" in metrics
        assert "doc_compliance_pct" in metrics

    def test_alert_preferences_defaults(self):
        prefs = AlertPreferences()

        assert prefs.unsigned_notes is True
        assert prefs.unsigned_threshold_hours == 24
        assert prefs.expiring_auths is True
        assert prefs.auth_threshold_days == 7
        assert prefs.missed_visits is True
        assert prefs.schedule_reminders is True
        assert prefs.reminder_minutes_before == 15
        assert prefs.goal_progress is True
        assert prefs.proactive_suggestions is True

    def test_usage_patterns_no_phi(self):
        profile = AgentProfile(user_id="user_1")

        patterns = profile.usage_patterns
        assert "peak_hours" in patterns
        assert "common_commands" in patterns
        assert "avg_session_minutes" in patterns
        assert "total_tasks_completed" in patterns
        # No PHI fields
        assert "patient" not in str(patterns).lower()
        assert "name" not in str(patterns).lower()

    def test_update_profile(self):
        store = AgentProfileStore()
        store.get_or_create("user_1")
        profile = store.update(
            "user_1", agent_name="My Assistant", personality="friendly"
        )

        assert profile.agent_name == "My Assistant"
        assert profile.personality == "friendly"

    def test_to_dict_serialization(self):
        store = AgentProfileStore()
        profile = store.get_or_create("user_1")
        data = profile.to_dict()

        assert data["user_id"] == "user_1"
        assert isinstance(data["goals"], list)
        assert isinstance(data["alert_preferences"], dict)
        assert isinstance(data["usage_patterns"], dict)

    def test_system_prompt_includes_custom_skills(self):
        profile = AgentProfile(user_id="user_1")
        profile.custom_skills.append(
            CustomSkill(name="Quick SOAP", description="Template gen", trigger="note")
        )

        prompt = profile.get_system_prompt()
        assert "Quick SOAP" in prompt

    def test_get_agent_store_singleton(self):
        """Module-level store is a singleton."""
        s1 = get_agent_store()
        s2 = get_agent_store()
        assert s1 is s2
