"""Session management for frontend applications."""

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/sessions", tags=["sessions"])


# In-memory session store (use Redis in production)
_sessions: dict[str, dict[str, Any]] = {}


class SessionCreate(BaseModel):
    """Request to create a new session."""

    user_id: Optional[str] = None
    discipline: str = "PT"
    care_setting: str = "outpatient"
    chief_complaint: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Session information."""

    session_id: str
    user_id: Optional[str] = None
    discipline: str = "PT"
    care_setting: str = "outpatient"
    created_at: str
    updated_at: Optional[str] = None
    last_activity: str
    consult_count: int = 0
    status: str = "pending"  # pending, in_progress, completed, error
    chief_complaint: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsultHistoryItem(BaseModel):
    """A consultation in session history."""

    consult_id: str
    timestamp: str
    query_summary: str
    diagnosis: Optional[str] = None
    has_red_flags: bool = False
    qa_score: Optional[float] = None


@router.get("", response_model=list[SessionResponse])
async def list_sessions():
    """List all active sessions."""
    return [SessionResponse(**s) for s in _sessions.values()]


@router.post("/create", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    """Create a new consultation session.

    Sessions track user context across multiple consultations.
    """
    session_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()

    session = {
        "session_id": session_id,
        "user_id": request.user_id,
        "discipline": request.discipline,
        "care_setting": request.care_setting,
        "created_at": now,
        "updated_at": now,
        "last_activity": now,
        "consult_count": 0,
        "status": "pending",
        "chief_complaint": request.chief_complaint,
        "consults": [],
        "metadata": request.metadata,
    }

    _sessions[session_id] = session

    return SessionResponse(**session)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session information."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(**_sessions[session_id])


@router.post("/{session_id}/consult")
async def add_consult_to_session(
    session_id: str,
    consult_id: str,
    query_summary: str,
    diagnosis: Optional[str] = None,
    has_red_flags: bool = False,
    qa_score: Optional[float] = None,
):
    """Record a consultation in the session history."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    session["last_activity"] = datetime.now(timezone.utc).isoformat()
    session["consult_count"] += 1
    session["consults"].append({
        "consult_id": consult_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_summary": query_summary[:100],
        "diagnosis": diagnosis,
        "has_red_flags": has_red_flags,
        "qa_score": qa_score,
    })

    return {"status": "recorded", "consult_count": session["consult_count"]}


@router.get("/{session_id}/history", response_model=list[ConsultHistoryItem])
async def get_session_history(
    session_id: str,
    limit: int = Query(20, ge=1, le=100),
):
    """Get consultation history for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    consults = _sessions[session_id].get("consults", [])
    return [ConsultHistoryItem(**c) for c in consults[-limit:]]


@router.delete("/{session_id}")
async def end_session(session_id: str):
    """End and clean up a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del _sessions[session_id]
    return {"status": "ended", "session_id": session_id}


@router.get("/{session_id}/logs")
async def get_session_logs(
    session_id: str,
    log_type: str = Query("orchestrator", description="Log type: orchestrator, agent_runs, llm_calls"),
    limit: int = Query(50, ge=1, le=200),
):
    """Get observability logs for a session.

    Useful for debugging and tracing consultation flow.
    """
    logs_dir = Path("data/logs")
    log_file = logs_dir / f"{log_type}.jsonl"

    if not log_file.exists():
        return {"logs": [], "total": 0}

    session_logs = []
    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("session_id") == session_id:
                    session_logs.append(entry)
            except json.JSONDecodeError:
                continue

    return {
        "logs": session_logs[-limit:],
        "total": len(session_logs),
        "session_id": session_id,
    }
