"""Feedback annotation API for reviewing agent outputs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from rehab_os.observability import PromptAnalytics

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackAnnotation(BaseModel):
    """Feedback annotation for an agent event."""

    event_id: str = Field(..., description="Request ID of the event to annotate")
    agent_name: str = Field(..., description="Name of the agent")
    status: str = Field(
        ..., description="Feedback status: accepted, rejected, needs_review"
    )
    notes: Optional[str] = Field(None, description="Reviewer notes")
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    suggested_improvement: Optional[str] = Field(
        None, description="Suggested prompt improvement"
    )


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""

    success: bool
    message: str
    event_id: str


class AgentEventSummary(BaseModel):
    """Summary of an agent event for review."""

    request_id: str
    agent_name: str
    timestamp: str
    input_summary: Optional[str]
    output_summary: Optional[str]
    confidence_score: Optional[float]
    duration_ms: Optional[float]
    feedback_status: Optional[str]
    feedback_notes: Optional[str]


class ReviewQueueResponse(BaseModel):
    """Response containing events to review."""

    total: int
    events: list[AgentEventSummary]


class PromptTuningCandidate(BaseModel):
    """Agent identified as needing prompt tuning."""

    agent: str
    reasons: list[str]
    metrics: dict[str, Any]


class EffectivenessReportResponse(BaseModel):
    """Response containing effectiveness report."""

    time_range: dict[str, str]
    summary: dict[str, Any]
    agents: dict[str, Any]
    attention_needed: dict[str, list[str]]


def _get_log_dir() -> Path:
    """Get log directory path."""
    return Path("data/logs")


def _load_agent_events(
    limit: int = 100,
    agent_name: Optional[str] = None,
    needs_review: bool = False,
) -> list[dict[str, Any]]:
    """Load agent events from log file."""
    log_file = _get_log_dir() / "agent_runs.jsonl"
    if not log_file.exists():
        return []

    events = []
    with open(log_file) as f:
        for line in f:
            try:
                event = json.loads(line)

                # Filter by agent name
                if agent_name and event.get("agent_name") != agent_name:
                    continue

                # Filter for events needing review
                if needs_review:
                    status = event.get("feedback_status")
                    if status in ("accepted", "rejected"):
                        continue

                events.append(event)
            except json.JSONDecodeError:
                continue

    # Return most recent first
    events.reverse()
    return events[:limit]


def _save_feedback(
    event_id: str,
    agent_name: str,
    feedback: FeedbackAnnotation,
) -> bool:
    """Save feedback annotation to a separate feedback log."""
    feedback_file = _get_log_dir() / "feedback.jsonl"

    feedback_entry = {
        "event_id": event_id,
        "agent_name": agent_name,
        "status": feedback.status,
        "notes": feedback.notes,
        "tags": feedback.tags,
        "suggested_improvement": feedback.suggested_improvement,
        "timestamp": datetime.utcnow().isoformat(),
    }

    with open(feedback_file, "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")

    return True


@router.get("/queue", response_model=ReviewQueueResponse)
async def get_review_queue(
    agent_name: Optional[str] = Query(None, description="Filter by agent name"),
    limit: int = Query(50, ge=1, le=200, description="Maximum events to return"),
    needs_review_only: bool = Query(True, description="Only show events needing review"),
):
    """Get queue of agent events for review.

    Returns agent outputs that haven't been reviewed yet,
    prioritized by those with low confidence or errors.
    """
    events = _load_agent_events(
        limit=limit,
        agent_name=agent_name,
        needs_review=needs_review_only,
    )

    summaries = [
        AgentEventSummary(
            request_id=e.get("request_id", "unknown"),
            agent_name=e.get("agent_name", "unknown"),
            timestamp=e.get("timestamp", ""),
            input_summary=e.get("input_summary"),
            output_summary=e.get("output_summary"),
            confidence_score=e.get("confidence_score"),
            duration_ms=e.get("duration_ms"),
            feedback_status=e.get("feedback_status"),
            feedback_notes=e.get("feedback_notes"),
        )
        for e in events
    ]

    # Sort by confidence (low first) to prioritize uncertain outputs
    summaries.sort(
        key=lambda x: x.confidence_score if x.confidence_score is not None else 0
    )

    return ReviewQueueResponse(total=len(summaries), events=summaries)


@router.post("/annotate", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackAnnotation):
    """Submit feedback annotation for an agent event.

    Marks the event as accepted, rejected, or needs_review,
    with optional notes and improvement suggestions.
    """
    if feedback.status not in ("accepted", "rejected", "needs_review"):
        raise HTTPException(
            status_code=400,
            detail="Status must be: accepted, rejected, or needs_review",
        )

    success = _save_feedback(
        event_id=feedback.event_id,
        agent_name=feedback.agent_name,
        feedback=feedback,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save feedback")

    return FeedbackResponse(
        success=True,
        message=f"Feedback saved for {feedback.agent_name}",
        event_id=feedback.event_id,
    )


@router.get("/tuning-candidates", response_model=list[PromptTuningCandidate])
async def get_tuning_candidates(
    days: int = Query(7, ge=1, le=30, description="Days of history to analyze"),
):
    """Get agents that need prompt tuning based on effectiveness metrics.

    Analyzes logs to identify agents with:
    - High QA correction rate
    - High latency
    - Low confidence scores
    - Low acceptance rate
    """
    from datetime import timedelta

    analytics = PromptAnalytics(_get_log_dir())
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    candidates = analytics.get_prompt_tuning_candidates(start_time, end_time)

    return [
        PromptTuningCandidate(
            agent=c["agent"],
            reasons=c["reasons"],
            metrics=c["metrics"],
        )
        for c in candidates
    ]


@router.get("/effectiveness", response_model=EffectivenessReportResponse)
async def get_effectiveness_report(
    days: int = Query(7, ge=1, le=30, description="Days of history to analyze"),
):
    """Get comprehensive prompt effectiveness report.

    Returns metrics on:
    - Agent performance (success rate, latency, confidence)
    - LLM usage (tokens, cost estimates, fallback rate)
    - Model tier efficiency
    - Agents needing attention
    """
    from datetime import timedelta

    analytics = PromptAnalytics(_get_log_dir())
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    report = analytics.generate_report(start_time, end_time)
    report_dict = report.to_dict()

    return EffectivenessReportResponse(
        time_range=report_dict["time_range"],
        summary=report_dict["summary"],
        agents=report_dict["agents"],
        attention_needed=report_dict["attention_needed"],
    )


@router.get("/feedback-stats")
async def get_feedback_stats():
    """Get statistics on feedback annotations."""
    feedback_file = _get_log_dir() / "feedback.jsonl"
    if not feedback_file.exists():
        return {
            "total_annotations": 0,
            "by_status": {},
            "by_agent": {},
            "common_tags": [],
        }

    feedback_entries = []
    with open(feedback_file) as f:
        for line in f:
            try:
                feedback_entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Aggregate stats
    by_status: dict[str, int] = {}
    by_agent: dict[str, dict[str, int]] = {}
    all_tags: list[str] = []

    for entry in feedback_entries:
        status = entry.get("status", "unknown")
        agent = entry.get("agent_name", "unknown")
        tags = entry.get("tags", [])

        by_status[status] = by_status.get(status, 0) + 1

        if agent not in by_agent:
            by_agent[agent] = {"accepted": 0, "rejected": 0, "needs_review": 0}
        by_agent[agent][status] = by_agent[agent].get(status, 0) + 1

        all_tags.extend(tags)

    # Count tag frequency
    tag_counts: dict[str, int] = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    common_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_annotations": len(feedback_entries),
        "by_status": by_status,
        "by_agent": by_agent,
        "common_tags": [{"tag": t[0], "count": t[1]} for t in common_tags],
    }


@router.get("/improvements")
async def get_improvement_suggestions():
    """Get suggested prompt improvements from feedback.

    Returns improvement suggestions grouped by agent.
    """
    feedback_file = _get_log_dir() / "feedback.jsonl"
    if not feedback_file.exists():
        return {"improvements": []}

    improvements: dict[str, list[str]] = {}

    with open(feedback_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                suggestion = entry.get("suggested_improvement")
                agent = entry.get("agent_name", "unknown")

                if suggestion:
                    if agent not in improvements:
                        improvements[agent] = []
                    improvements[agent].append(suggestion)
            except json.JSONDecodeError:
                continue

    return {
        "improvements": [
            {"agent": agent, "suggestions": suggestions}
            for agent, suggestions in improvements.items()
        ]
    }
