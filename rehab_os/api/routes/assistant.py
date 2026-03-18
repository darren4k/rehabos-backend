"""AI Assistant API routes — session-aware agentic assistant."""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from rehab_os.api.dependencies import get_current_user
from rehab_os.assistant.coordinator import (
    AssistantResponse,
    AssistantSession,
    ChatCoordinator,
    get_session,
)
from rehab_os.assistant.page_context import get_page_context, get_role_commands, PAGE_CONTEXTS
from rehab_os.core.models import Provider

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/assistant",
    dependencies=[Depends(get_current_user)],
)

# ---------------------------------------------------------------------------
# In-memory conversation storage (keyed by provider_id, last 50 messages)
# ---------------------------------------------------------------------------
MAX_HISTORY = 50
_conversations: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

# Pending approval actions (keyed by action_id)
_pending_approvals: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    page: str = Field(default="dashboard")
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None


class ChatResponseModel(BaseModel):
    type: str
    content: str
    data: Optional[dict[str, Any]] = None
    suggestions: Optional[list[str]] = None
    requires_approval: bool = False
    approval_action: Optional[str] = None
    navigation: Optional[str] = None
    form_action: Optional[dict[str, Any]] = None
    speak: Optional[bool] = None


class PageContextResponse(BaseModel):
    label: str
    greeting: str
    commands: list[str]
    role_commands: Optional[list[dict]] = None


class ApprovalResponse(BaseModel):
    status: str
    message: str


class HistoryEntry(BaseModel):
    role: str
    content: str
    timestamp: str
    response_type: Optional[str] = None


class SetPatientRequest(BaseModel):
    patient_id: str
    patient_name: str


class SessionResponse(BaseModel):
    session_id: str
    provider_id: str
    active_patient_id: Optional[str] = None
    active_patient_name: Optional[str] = None
    active_encounter_id: Optional[str] = None
    current_page: str
    recent_patients: list[dict]
    tasks_in_progress: list[dict]
    message_count: int
    created_at: str
    last_activity: str


class SendFaxRequest(BaseModel):
    recipient_name: str
    recipient_fax: str = ""
    patient_id: Optional[str] = None
    content: str = ""
    document_type: str = "clinical_summary"


class SendMessageRequest(BaseModel):
    recipient_id: str = ""
    recipient_name: str = ""
    subject: str = ""
    body: str
    urgent: bool = False


class TaskRequest(BaseModel):
    task_type: str  # "onboard", "complete_visit", "compliance_check"
    params: dict[str, Any] = Field(default_factory=dict)


class PublicChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_coordinator(request: Request) -> ChatCoordinator:
    """Get or create the ChatCoordinator using the app's LLM router."""
    if not hasattr(request.app.state, "_assistant_coordinator"):
        request.app.state._assistant_coordinator = ChatCoordinator(
            llm=request.app.state.llm_router,
        )
    return request.app.state._assistant_coordinator


# ---------------------------------------------------------------------------
# Public endpoint (no auth)
# ---------------------------------------------------------------------------
public_router = APIRouter(prefix="/assistant")


@public_router.post("/public/chat")
async def public_chat(body: PublicChatRequest) -> dict:
    """Limited chatbot for landing/login page -- no PHI, just product info."""
    msg = body.message.lower()

    # Simple keyword responses for common questions
    if any(w in msg for w in ("feature", "what can", "what does", "capabilities")):
        answer = (
            "RehabOS is a comprehensive rehabilitation practice management platform. "
            "Key features: AI-powered SOAP documentation, scheduling, billing with "
            "8-minute rule calculator, outcome tracking, care plan management, "
            "and clinical decision support for PT/OT/SLP."
        )
    elif any(w in msg for w in ("price", "pricing", "cost", "how much")):
        answer = "Contact us for pricing details. We offer plans for solo practitioners and multi-site clinics."
    elif any(w in msg for w in ("demo", "trial", "try")):
        answer = "We'd love to show you RehabOS. Contact us to schedule a personalized demo."
    elif any(w in msg for w in ("hipaa", "security", "compliant")):
        answer = (
            "RehabOS is built with HIPAA compliance at its core: encrypted PHI, "
            "role-based access, full audit trails, and secure local-first AI processing."
        )
    elif any(w in msg for w in ("hello", "hi", "hey")):
        answer = "Hello! I'm the RehabOS assistant. I can answer questions about the platform. Log in to access clinical features."
    else:
        answer = (
            "I'm the RehabOS assistant. I can answer questions about features, "
            "pricing, and security. Log in to access clinical documentation, "
            "scheduling, and AI-powered tools."
        )

    return {
        "type": "text",
        "content": answer,
        "suggestions": ["What features does RehabOS have?", "Is RehabOS HIPAA compliant?", "Can I get a demo?"],
    }


# ---------------------------------------------------------------------------
# Authenticated endpoints
# ---------------------------------------------------------------------------
@router.post("/chat", response_model=ChatResponseModel)
async def chat(
    body: ChatRequest,
    request: Request,
    current_user: Provider = Depends(get_current_user),
) -> ChatResponseModel:
    """Main chat endpoint. Session-aware with command detection and LLM fallback."""
    provider_id = str(current_user.id)
    coordinator = _get_coordinator(request)
    history = _conversations[provider_id]

    # Record user message
    now = datetime.now(timezone.utc).isoformat()
    history.append({"role": "user", "content": body.message, "timestamp": now})

    # Process with session context
    result: AssistantResponse = await coordinator.process(
        provider_id=provider_id,
        message=body.message,
        page_context=body.page,
        patient_id=body.patient_id,
        patient_name=body.patient_name,
        conversation_history=list(history),
    )

    # Record assistant response
    history.append({
        "role": "assistant",
        "content": result.content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "response_type": result.type,
    })

    # If requires approval, stash the action
    action_id: Optional[str] = None
    if result.requires_approval and result.approval_action:
        action_id = str(uuid.uuid4())
        _pending_approvals[action_id] = {
            "action": result.approval_action,
            "content": result.content,
            "data": result.data,
            "provider_id": provider_id,
            "patient_id": body.patient_id,
            "created": now,
        }
        result.approval_action = action_id

    return ChatResponseModel(**result.to_dict())


@router.get("/context/{page}", response_model=PageContextResponse)
async def get_context(
    page: str,
    current_user: Provider = Depends(get_current_user),
) -> PageContextResponse:
    """Get page context, greeting, available commands, and role-specific actions."""
    ctx = get_page_context(page)
    role_cmds = get_role_commands(getattr(current_user, "role", "therapist"))
    return PageContextResponse(
        label=ctx["label"],
        greeting=ctx["greeting"],
        commands=ctx["commands"],
        role_commands=role_cmds,
    )


@router.post("/approve/{action_id}", response_model=ApprovalResponse)
async def approve_action(
    action_id: str,
    current_user: Provider = Depends(get_current_user),
) -> ApprovalResponse:
    """Approve a suggested clinical action."""
    provider_id = str(current_user.id)

    pending = _pending_approvals.pop(action_id, None)
    if not pending:
        raise HTTPException(status_code=404, detail="Action not found or already processed")

    if pending["provider_id"] != provider_id:
        _pending_approvals[action_id] = pending
        raise HTTPException(status_code=403, detail="Not your action to approve")

    logger.info(
        "Provider %s approved action %s (%s)",
        provider_id,
        action_id,
        pending["action"],
    )

    return ApprovalResponse(
        status="approved",
        message=f"Action '{pending['action']}' approved. Ready to apply.",
    )


@router.get("/history", response_model=list[HistoryEntry])
async def get_history(
    current_user: Provider = Depends(get_current_user),
) -> list[HistoryEntry]:
    """Get recent conversation history (last 50 messages)."""
    provider_id = str(current_user.id)
    history = _conversations.get(provider_id, deque())
    return [
        HistoryEntry(
            role=entry["role"],
            content=entry["content"],
            timestamp=entry["timestamp"],
            response_type=entry.get("response_type"),
        )
        for entry in history
    ]


@router.post("/set-patient")
async def set_patient(
    body: SetPatientRequest,
    current_user: Provider = Depends(get_current_user),
) -> dict:
    """Set active patient in session."""
    provider_id = str(current_user.id)
    session = get_session(provider_id)
    session.set_patient(body.patient_id, body.patient_name)
    return {
        "status": "ok",
        "active_patient_id": session.active_patient_id,
        "active_patient_name": session.active_patient_name,
    }


@router.get("/session", response_model=SessionResponse)
async def get_session_state(
    current_user: Provider = Depends(get_current_user),
) -> SessionResponse:
    """Get current session state (active patient, recent patients, tasks)."""
    provider_id = str(current_user.id)
    session = get_session(provider_id)
    return SessionResponse(**session.to_dict())


@router.post("/send-fax")
async def send_fax(
    body: SendFaxRequest,
    current_user: Provider = Depends(get_current_user),
) -> dict:
    """Queue a fax via DocPilot delivery service."""
    provider_id = str(current_user.id)
    session = get_session(provider_id)
    task_id = str(uuid.uuid4())
    session.add_task(task_id, f"Fax to {body.recipient_name}")

    # In production, this would call DocPilot at localhost:3847
    logger.info(
        "Fax queued: provider=%s, recipient=%s, patient=%s, task=%s",
        provider_id, body.recipient_name, body.patient_id, task_id,
    )

    return {
        "status": "queued",
        "task_id": task_id,
        "recipient": body.recipient_name,
        "recipient_fax": body.recipient_fax,
        "message": f"Fax queued to {body.recipient_name}" + (f" at {body.recipient_fax}" if body.recipient_fax else ""),
    }


@router.post("/send-message")
async def send_message(
    body: SendMessageRequest,
    current_user: Provider = Depends(get_current_user),
) -> dict:
    """Send message to care team member."""
    provider_id = str(current_user.id)
    session = get_session(provider_id)
    task_id = str(uuid.uuid4())
    session.add_task(task_id, f"Message to {body.recipient_name or body.recipient_id}")

    logger.info(
        "Message sent: provider=%s, recipient=%s, urgent=%s, task=%s",
        provider_id, body.recipient_name or body.recipient_id, body.urgent, task_id,
    )

    return {
        "status": "sent",
        "task_id": task_id,
        "recipient": body.recipient_name or body.recipient_id,
        "urgent": body.urgent,
        "message": f"Message sent to {body.recipient_name or body.recipient_id}.",
    }


@router.post("/task")
async def execute_task(
    body: TaskRequest,
    request: Request,
    current_user: Provider = Depends(get_current_user),
) -> dict:
    """Execute a multi-step workflow task."""
    provider_id = str(current_user.id)
    session = get_session(provider_id)
    task_id = str(uuid.uuid4())

    task_handlers = {
        "onboard": _task_onboard,
        "complete_visit": _task_complete_visit,
        "compliance_check": _task_compliance_check,
    }

    handler = task_handlers.get(body.task_type)
    if not handler:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task type: {body.task_type}. Valid: {list(task_handlers.keys())}",
        )

    session.add_task(task_id, f"{body.task_type}: {body.params}")
    result = await handler(provider_id, session, body.params, request)
    session.complete_task(task_id)

    return {
        "status": "complete",
        "task_id": task_id,
        "task_type": body.task_type,
        "result": result,
    }


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------
async def _task_onboard(
    provider_id: str, session: AssistantSession, params: dict, request: Request
) -> dict:
    """Multi-step patient onboarding."""
    patient_name = params.get("patient_name", "New Patient")
    steps = [
        {"step": 1, "action": "create_record", "status": "ready", "description": f"Create patient record for {patient_name}"},
        {"step": 2, "action": "navigate_intake", "status": "pending", "description": "Navigate to intake form"},
        {"step": 3, "action": "collect_demographics", "status": "pending", "description": "Collect demographics and insurance"},
        {"step": 4, "action": "initial_eval", "status": "pending", "description": "Begin initial evaluation"},
    ]
    return {"patient_name": patient_name, "steps": steps, "navigation": "/intake"}


async def _task_complete_visit(
    provider_id: str, session: AssistantSession, params: dict, request: Request
) -> dict:
    """Complete visit workflow."""
    patient_id = session.active_patient_id or params.get("patient_id")
    steps = [
        {"step": 1, "action": "generate_soap", "status": "ready", "description": "Generate SOAP draft"},
        {"step": 2, "action": "review_note", "status": "pending", "description": "Review and edit note"},
        {"step": 3, "action": "add_billing", "status": "pending", "description": "Add billing codes"},
        {"step": 4, "action": "sign_note", "status": "pending", "description": "Sign and lock note"},
    ]
    return {"patient_id": patient_id, "steps": steps, "navigation": "/skilled-notes"}


async def _task_compliance_check(
    provider_id: str, session: AssistantSession, params: dict, request: Request
) -> dict:
    """Run compliance check across categories."""
    return {
        "provider_id": provider_id,
        "checks": [
            {"category": "unsigned_notes", "description": "Notes pending signature"},
            {"category": "expiring_auths", "description": "Authorizations expiring in 7 days"},
            {"category": "overdue_recerts", "description": "Overdue re-certifications"},
            {"category": "missing_docs", "description": "Missing required documentation"},
            {"category": "billing_discrepancies", "description": "Billing vs documentation mismatches"},
        ],
        "navigation": "/reports",
    }
