"""AI Assistant API routes."""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from rehab_os.api.dependencies import get_current_user
from rehab_os.assistant.coordinator import AssistantResponse, ChatCoordinator
from rehab_os.assistant.page_context import get_page_context, PAGE_CONTEXTS
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


class ChatResponseModel(BaseModel):
    type: str
    content: str
    data: Optional[dict[str, Any]] = None
    suggestions: Optional[list[str]] = None
    requires_approval: bool = False
    approval_action: Optional[str] = None


class PageContextResponse(BaseModel):
    label: str
    greeting: str
    commands: list[str]


class ApprovalResponse(BaseModel):
    status: str
    message: str


class HistoryEntry(BaseModel):
    role: str
    content: str
    timestamp: str
    response_type: Optional[str] = None


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
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/chat", response_model=ChatResponseModel)
async def chat(
    body: ChatRequest,
    request: Request,
    current_user: Provider = Depends(get_current_user),
) -> ChatResponseModel:
    """Main chat endpoint. Processes message through command detection or LLM."""
    provider_id = str(current_user.id)
    coordinator = _get_coordinator(request)
    history = _conversations[provider_id]

    # Record user message
    now = datetime.now(timezone.utc).isoformat()
    history.append({"role": "user", "content": body.message, "timestamp": now})

    # Process
    result: AssistantResponse = await coordinator.process(
        provider_id=provider_id,
        message=body.message,
        page_context=body.page,
        patient_id=body.patient_id,
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
        result.approval_action = action_id  # Replace action name with ID

    return ChatResponseModel(**result.to_dict())


@router.get("/context/{page}", response_model=PageContextResponse)
async def get_context(
    page: str,
    current_user: Provider = Depends(get_current_user),
) -> PageContextResponse:
    """Get page context, greeting, and available quick commands."""
    ctx = get_page_context(page)
    return PageContextResponse(
        label=ctx["label"],
        greeting=ctx["greeting"],
        commands=ctx["commands"],
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
        _pending_approvals[action_id] = pending  # Put it back
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
