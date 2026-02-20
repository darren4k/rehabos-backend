"""Session management for frontend applications."""

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from rehab_os.api.dependencies import get_current_user
from rehab_os.core.models import Provider

router = APIRouter(prefix="/sessions", tags=["sessions"])

# File-backed session store with in-memory cache
_SESSIONS_DIR = Path("data/sessions")
_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
_sessions: dict[str, dict[str, Any]] = {}


def _session_file(session_id: str) -> Path:
    return _SESSIONS_DIR / f"{session_id}.json"


def _save_session(session: dict[str, Any]) -> None:
    """Persist session to disk and update cache."""
    sid = session["session_id"]
    _sessions[sid] = session
    with open(_session_file(sid), "w") as f:
        json.dump(session, f)


def _load_session(session_id: str) -> Optional[dict[str, Any]]:
    """Load session from cache or disk."""
    if session_id in _sessions:
        return _sessions[session_id]
    fp = _session_file(session_id)
    if fp.exists():
        with open(fp) as f:
            data = json.load(f)
        _sessions[session_id] = data
        return data
    return None


def _delete_session(session_id: str) -> None:
    """Remove session from cache and disk."""
    _sessions.pop(session_id, None)
    fp = _session_file(session_id)
    if fp.exists():
        fp.unlink()


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
async def list_sessions(current_user: Provider = Depends(get_current_user)):
    """List all active sessions."""
    # Load any sessions from disk not yet in cache
    for fp in _SESSIONS_DIR.glob("*.json"):
        sid = fp.stem
        if sid not in _sessions:
            try:
                with open(fp) as f:
                    _sessions[sid] = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return [SessionResponse(**s) for s in _sessions.values()]


@router.post("/create", response_model=SessionResponse)
async def create_session(request: SessionCreate, current_user: Provider = Depends(get_current_user)):
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

    _save_session(session)

    return SessionResponse(**session)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, current_user: Provider = Depends(get_current_user)):
    """Get session information."""
    session = _load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(**session)


@router.post("/{session_id}/consult")
async def add_consult_to_session(
    session_id: str,
    consult_id: str,
    query_summary: str,
    diagnosis: Optional[str] = None,
    has_red_flags: bool = False,
    qa_score: Optional[float] = None,
    current_user: Provider = Depends(get_current_user),
):
    """Record a consultation in the session history."""
    session = _load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

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
    _save_session(session)

    return {"status": "recorded", "consult_count": session["consult_count"]}


@router.get("/{session_id}/history", response_model=list[ConsultHistoryItem])
async def get_session_history(
    session_id: str,
    limit: int = Query(20, ge=1, le=100),
    current_user: Provider = Depends(get_current_user),
):
    """Get consultation history for a session."""
    session = _load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    consults = session.get("consults", [])
    return [ConsultHistoryItem(**c) for c in consults[-limit:]]


@router.delete("/{session_id}")
async def end_session(session_id: str, current_user: Provider = Depends(get_current_user)):
    """End and clean up a session."""
    session = _load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    _delete_session(session_id)
    return {"status": "ended", "session_id": session_id}


@router.get("/{session_id}/logs")
async def get_session_logs(
    session_id: str,
    log_type: str = Query("orchestrator", description="Log type: orchestrator, agent_runs, llm_calls"),
    limit: int = Query(50, ge=1, le=200),
    current_user: Provider = Depends(get_current_user),
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
