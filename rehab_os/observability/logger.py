"""Observability logger for structured telemetry."""

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from rehab_os.observability.events import (
    AgentEvent,
    EventType,
    LLMCallEvent,
    KnowledgeSearchEvent,
    ObservabilityEvent,
    OrchestratorEvent,
)

logger = logging.getLogger(__name__)


class ObservabilityLogger:
    """Central logger for LLM and agent observability events.

    Writes structured events to JSON Lines files for later analysis.
    Supports multiple output sinks and filtering.
    """

    _instance: Optional["ObservabilityLogger"] = None

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        enabled: bool = True,
        log_full_content: bool = False,
        max_content_length: int = 500,
    ):
        """Initialize observability logger.

        Args:
            log_dir: Directory for log files (default: data/logs)
            enabled: Whether logging is enabled
            log_full_content: Whether to log full message content
            max_content_length: Max length for truncated content
        """
        self.enabled = enabled
        self.log_full_content = log_full_content
        self.max_content_length = max_content_length

        if log_dir is None:
            log_dir = Path("data/logs")
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Separate files for different event types
        self._log_files: dict[str, Path] = {
            "llm": self.log_dir / "llm_calls.jsonl",
            "agents": self.log_dir / "agent_runs.jsonl",
            "orchestrator": self.log_dir / "orchestrator.jsonl",
            "knowledge": self.log_dir / "knowledge_search.jsonl",
        }

        # Event callbacks for real-time monitoring
        self._callbacks: list[Callable[[ObservabilityEvent], None]] = []

        # Session tracking
        self._current_session_id: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "ObservabilityLogger":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_session_id(self, session_id: str) -> None:
        """Set current session ID for event correlation."""
        self._current_session_id = session_id

    def generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())[:8]

    def add_callback(self, callback: Callable[[ObservabilityEvent], None]) -> None:
        """Add callback for real-time event monitoring."""
        self._callbacks.append(callback)

    def _write_event(self, event: ObservabilityEvent, log_type: str) -> None:
        """Write event to appropriate log file."""
        if not self.enabled:
            return

        # Add session ID if set
        if self._current_session_id and not event.session_id:
            event.session_id = self._current_session_id

        try:
            log_file = self._log_files.get(log_type)
            if log_file:
                with open(log_file, "a") as f:
                    f.write(event.model_dump_json() + "\n")

            # Trigger callbacks
            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.warning(f"Observability callback failed: {e}")

        except Exception as e:
            logger.warning(f"Failed to write observability event: {e}")

    def _truncate(self, content: str) -> str:
        """Truncate content if needed."""
        if self.log_full_content:
            return content
        if len(content) <= self.max_content_length:
            return content
        return content[: self.max_content_length] + "..."

    # LLM Call Logging

    @contextmanager
    def llm_call(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        request_id: Optional[str] = None,
    ):
        """Context manager for logging LLM calls.

        Usage:
            with obs.llm_call(provider, model, messages) as event:
                response = await llm.complete(...)
                event.response_content = response.content
                event.input_tokens = response.usage.input_tokens
        """
        start_time = time.time()
        request_id = request_id or self.generate_request_id()

        # Truncate message content for privacy
        logged_messages = [
            {"role": m["role"], "content": self._truncate(m.get("content", ""))}
            for m in messages
        ]

        event = LLMCallEvent(
            event_type=EventType.LLM_CALL_START,
            provider=provider,
            model=model,
            messages=logged_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            request_id=request_id,
        )

        try:
            yield event
            event.event_type = EventType.LLM_CALL_SUCCESS
            if event.response_content:
                event.response_content = self._truncate(event.response_content)

        except Exception as e:
            event.event_type = EventType.LLM_CALL_ERROR
            event.error_type = type(e).__name__
            event.error_message = str(e)[:200]
            raise

        finally:
            event.duration_ms = (time.time() - start_time) * 1000
            self._write_event(event, "llm")

    def log_llm_fallback(
        self,
        from_provider: str,
        to_provider: str,
        reason: str,
        request_id: Optional[str] = None,
    ) -> None:
        """Log LLM fallback event."""
        event = LLMCallEvent(
            event_type=EventType.LLM_FALLBACK,
            provider=to_provider,
            model="",
            is_fallback=True,
            fallback_reason=reason,
            request_id=request_id,
            metadata={"from_provider": from_provider},
        )
        self._write_event(event, "llm")

    # Agent Logging

    @contextmanager
    def agent_run(
        self,
        agent_name: str,
        agent_description: Optional[str] = None,
        model_tier: Optional[str] = None,
        input_type: Optional[str] = None,
        input_summary: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        """Context manager for logging agent runs.

        Usage:
            with obs.agent_run("diagnosis", input_type="PatientContext") as event:
                result = await agent.run(inputs)
                event.output_summary = result.primary_diagnosis
        """
        start_time = time.time()
        request_id = request_id or self.generate_request_id()

        event = AgentEvent(
            event_type=EventType.AGENT_START,
            agent_name=agent_name,
            agent_description=agent_description,
            model_tier=model_tier,
            input_type=input_type,
            input_summary=self._truncate(input_summary) if input_summary else None,
            request_id=request_id,
        )

        try:
            yield event
            event.event_type = EventType.AGENT_SUCCESS
            if event.output_summary:
                event.output_summary = self._truncate(event.output_summary)

        except Exception as e:
            event.event_type = EventType.AGENT_ERROR
            event.error_type = type(e).__name__
            event.error_message = str(e)[:200]
            raise

        finally:
            event.duration_ms = (time.time() - start_time) * 1000
            self._write_event(event, "agents")

    # Orchestrator Logging

    @contextmanager
    def orchestrator_run(
        self,
        discipline: str,
        setting: str,
        query_summary: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        """Context manager for logging orchestrator pipeline runs."""
        start_time = time.time()
        request_id = request_id or self.generate_request_id()

        event = OrchestratorEvent(
            event_type=EventType.ORCHESTRATOR_START,
            discipline=discipline,
            setting=setting,
            query_summary=self._truncate(query_summary) if query_summary else None,
            request_id=request_id,
        )

        try:
            yield event
            event.event_type = EventType.ORCHESTRATOR_SUCCESS

        except Exception as e:
            event.event_type = EventType.ORCHESTRATOR_ERROR
            event.error_type = type(e).__name__
            event.error_message = str(e)[:200]
            raise

        finally:
            event.duration_ms = (time.time() - start_time) * 1000
            self._write_event(event, "orchestrator")

    # Knowledge Base Logging

    def log_knowledge_search(
        self,
        query: str,
        top_k: int,
        results_count: int,
        source_type: str = "vector_store",
        top_result_score: Optional[float] = None,
        top_result_source: Optional[str] = None,
        duration_ms: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Log knowledge base search."""
        event = KnowledgeSearchEvent(
            query=self._truncate(query),
            top_k=top_k,
            results_count=results_count,
            source_type=source_type,
            top_result_score=top_result_score,
            top_result_source=top_result_source,
            duration_ms=duration_ms,
            request_id=request_id,
        )
        self._write_event(event, "knowledge")

    # Utility methods

    def get_recent_events(
        self,
        log_type: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read recent events from a log file."""
        log_file = self._log_files.get(log_type)
        if not log_file or not log_file.exists():
            return []

        events = []
        with open(log_file) as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return events[-limit:]

    def get_stats(self, log_type: str) -> dict[str, Any]:
        """Get basic statistics for a log type."""
        events = self.get_recent_events(log_type, limit=1000)
        if not events:
            return {"total": 0}

        total = len(events)
        errors = sum(1 for e in events if "error" in e.get("event_type", ""))
        avg_duration = sum(e.get("duration_ms", 0) for e in events) / total

        return {
            "total": total,
            "errors": errors,
            "error_rate": errors / total if total > 0 else 0,
            "avg_duration_ms": avg_duration,
        }


def get_observability_logger() -> ObservabilityLogger:
    """Get the global observability logger instance."""
    return ObservabilityLogger.get_instance()
