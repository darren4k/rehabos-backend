"""API routes for personalized AI agent profiles.

All endpoints are auth-protected. Agent profiles store NO PHI —
only preferences, aggregate metrics, and custom skills/goals.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from rehab_os.agents.goal_tracker import list_metrics, update_goal_progress
from rehab_os.agents.personal_agent import (
    AgentGoal,
    AlertPreferences,
    CustomSkill,
    get_agent_store,
)
from rehab_os.agents.proactive_engine import get_proactive_engine
from rehab_os.api.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    dependencies=[Depends(get_current_user)],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ProfileUpdateRequest(BaseModel):
    agent_name: Optional[str] = None
    personality: Optional[str] = None
    role: Optional[str] = None
    skills: Optional[list[str]] = None
    tools: Optional[list[str]] = None
    alert_preferences: Optional[dict] = None


class CustomSkillRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    trigger: str = Field(default="", max_length=200)
    prompt_template: str = Field(default="", max_length=2000)
    output_format: str = Field(default="text")
    enabled: bool = True


class GoalRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    metric: str = Field(..., min_length=1)
    target: float = Field(..., gt=0)
    period: str = Field(default="weekly")
    category: str = Field(default="compliance")


# ---------------------------------------------------------------------------
# Profile endpoints
# ---------------------------------------------------------------------------

@router.get("/profile")
async def get_profile(user=Depends(get_current_user)):
    """Get current user's agent profile."""
    store = get_agent_store()
    profile = store.get_or_create(str(user.id))
    return profile.to_dict()


@router.put("/profile")
async def update_profile(
    body: ProfileUpdateRequest,
    user=Depends(get_current_user),
):
    """Update agent profile settings."""
    store = get_agent_store()
    updates = body.model_dump(exclude_none=True)

    # Handle nested alert_preferences
    if "alert_preferences" in updates:
        profile = store.get_or_create(str(user.id))
        prefs = profile.alert_preferences
        for key, value in updates.pop("alert_preferences").items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)

    if updates:
        store.update(str(user.id), **updates)

    profile = store.get_or_create(str(user.id))
    return profile.to_dict()


# ---------------------------------------------------------------------------
# Custom skills endpoints
# ---------------------------------------------------------------------------

@router.post("/skills")
async def create_custom_skill(
    body: CustomSkillRequest,
    user=Depends(get_current_user),
):
    """Create a custom skill. Prompt templates must NOT contain PHI."""
    store = get_agent_store()
    skill = CustomSkill(
        name=body.name,
        description=body.description,
        trigger=body.trigger,
        prompt_template=body.prompt_template,
        output_format=body.output_format,
        enabled=body.enabled,
    )
    profile = store.add_custom_skill(str(user.id), skill)
    return {
        "skill_id": skill.skill_id,
        "profile": profile.to_dict(),
    }


@router.delete("/skills/{skill_id}")
async def delete_custom_skill(
    skill_id: str,
    user=Depends(get_current_user),
):
    """Delete a custom skill."""
    store = get_agent_store()
    removed = store.remove_custom_skill(str(user.id), skill_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"deleted": skill_id}


# ---------------------------------------------------------------------------
# Goals endpoints
# ---------------------------------------------------------------------------

@router.get("/goals")
async def get_goals(user=Depends(get_current_user)):
    """Get goals with current progress (aggregate metrics only)."""
    store = get_agent_store()
    profile = store.get_or_create(str(user.id))

    # Refresh current values from live metrics
    await update_goal_progress(str(user.id), profile.goals)

    return {
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
            for g in profile.goals
        ],
        "available_metrics": list_metrics(),
    }


@router.post("/goals")
async def create_goal(
    body: GoalRequest,
    user=Depends(get_current_user),
):
    """Create a new performance goal."""
    store = get_agent_store()
    goal = AgentGoal(
        title=body.title,
        description=body.description,
        metric=body.metric,
        target=body.target,
        period=body.period,
        category=body.category,
    )
    profile = store.add_goal(str(user.id), goal)
    return {
        "goal_id": goal.goal_id,
        "profile": profile.to_dict(),
    }


@router.delete("/goals/{goal_id}")
async def delete_goal(
    goal_id: str,
    user=Depends(get_current_user),
):
    """Delete a goal."""
    store = get_agent_store()
    removed = store.remove_goal(str(user.id), goal_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Goal not found")
    return {"deleted": goal_id}


# ---------------------------------------------------------------------------
# Alerts endpoints
# ---------------------------------------------------------------------------

@router.get("/alerts")
async def get_alerts(user=Depends(get_current_user)):
    """Get current proactive alerts. Aggregate counts only — NO PHI."""
    store = get_agent_store()
    engine = get_proactive_engine()
    profile = store.get_or_create(str(user.id))
    alerts = await engine.check_all(str(user.id), profile)
    return {"alerts": [a.to_dict() for a in alerts]}


@router.post("/alerts/{alert_id}/dismiss")
async def dismiss_alert(
    alert_id: str,
    user=Depends(get_current_user),
):
    """Dismiss a proactive alert."""
    engine = get_proactive_engine()
    engine.dismiss(str(user.id), alert_id)
    return {"dismissed": alert_id}
