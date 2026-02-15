"""Structured observability events for LLM and agent telemetry."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of observability events."""

    LLM_CALL_START = "llm_call_start"
    LLM_CALL_SUCCESS = "llm_call_success"
    LLM_CALL_ERROR = "llm_call_error"
    LLM_FALLBACK = "llm_fallback"
    AGENT_START = "agent_start"
    AGENT_SUCCESS = "agent_success"
    AGENT_ERROR = "agent_error"
    ORCHESTRATOR_START = "orchestrator_start"
    ORCHESTRATOR_SUCCESS = "orchestrator_success"
    ORCHESTRATOR_ERROR = "orchestrator_error"
    KNOWLEDGE_SEARCH = "knowledge_search"


class ObservabilityEvent(BaseModel):
    """Base class for all observability events."""

    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMCallEvent(ObservabilityEvent):
    """Event for LLM API calls."""

    provider: str
    model: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096

    # Response fields (populated on success)
    response_content: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Error fields (populated on error)
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Structured output
    structured_schema: Optional[str] = None

    # Routing info
    is_fallback: bool = False
    fallback_reason: Optional[str] = None


class AgentEvent(ObservabilityEvent):
    """Event for agent executions."""

    agent_name: str
    agent_description: Optional[str] = None
    model_tier: Optional[str] = None

    # Input summary (not full content for privacy)
    input_type: Optional[str] = None
    input_summary: Optional[str] = None

    # Output summary
    output_type: Optional[str] = None
    output_summary: Optional[str] = None

    # Performance
    llm_calls: int = 0
    total_tokens: Optional[int] = None

    # Confidence and quality metrics (for self-critique loops)
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Agent's self-estimated confidence"
    )
    evidence_quality: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Quality score of supporting evidence"
    )
    guideline_alignment: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Alignment with clinical guidelines"
    )
    uncertainty_flags: list[str] = Field(
        default_factory=list, description="Areas of uncertainty identified"
    )

    # Error fields
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Feedback annotation (populated by review)
    feedback_status: Optional[str] = Field(
        None, description="accepted, rejected, needs_review"
    )
    feedback_notes: Optional[str] = None
    feedback_tags: list[str] = Field(default_factory=list)


class OrchestratorEvent(ObservabilityEvent):
    """Event for orchestrator pipeline executions."""

    discipline: str
    setting: str
    query_summary: Optional[str] = None

    # Pipeline steps
    agents_called: list[str] = Field(default_factory=list)
    total_llm_calls: int = 0
    total_tokens: Optional[int] = None

    # Results
    has_red_flags: bool = False
    is_emergency: bool = False
    diagnosis_confidence: Optional[float] = None
    qa_score: Optional[float] = None

    # Error fields
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class KnowledgeSearchEvent(ObservabilityEvent):
    """Event for knowledge base searches."""

    event_type: EventType = EventType.KNOWLEDGE_SEARCH
    query: str
    top_k: int = 5
    results_count: int = 0
    source_type: str = "vector_store"  # vector_store, pubmed, etc.

    # Top result info (for relevance tracking)
    top_result_score: Optional[float] = None
    top_result_source: Optional[str] = None
