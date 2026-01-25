"""Tests for observability logger."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from rehab_os.observability import (
    ObservabilityLogger,
    EventType,
    LLMCallEvent,
    AgentEvent,
    get_observability_logger,
)


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create temporary log directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def obs_logger(temp_log_dir):
    """Create observability logger with temp directory."""
    return ObservabilityLogger(log_dir=temp_log_dir, enabled=True)


class TestObservabilityLogger:
    """Tests for ObservabilityLogger."""

    def test_init_creates_log_directory(self, tmp_path):
        """Test that init creates log directory if needed."""
        log_dir = tmp_path / "new_logs"
        logger = ObservabilityLogger(log_dir=log_dir)

        assert log_dir.exists()

    def test_disabled_logger_writes_nothing(self, temp_log_dir):
        """Test that disabled logger doesn't write."""
        logger = ObservabilityLogger(log_dir=temp_log_dir, enabled=False)

        with logger.llm_call(
            provider="test",
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
        ) as event:
            event.response_content = "response"

        log_file = temp_log_dir / "llm_calls.jsonl"
        if log_file.exists():
            assert log_file.read_text() == ""

    def test_llm_call_success(self, obs_logger, temp_log_dir):
        """Test logging successful LLM call."""
        with obs_logger.llm_call(
            provider="anthropic",
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
        ) as event:
            event.response_content = "Hi there!"
            event.input_tokens = 10
            event.output_tokens = 5
            event.total_tokens = 15

        log_file = temp_log_dir / "llm_calls.jsonl"
        assert log_file.exists()

        events = [json.loads(line) for line in log_file.read_text().strip().split("\n")]
        assert len(events) == 1
        assert events[0]["event_type"] == "llm_call_success"
        assert events[0]["provider"] == "anthropic"
        assert events[0]["model"] == "claude-3"
        assert events[0]["input_tokens"] == 10
        assert events[0]["duration_ms"] is not None

    def test_llm_call_error(self, obs_logger, temp_log_dir):
        """Test logging failed LLM call."""
        with pytest.raises(ValueError):
            with obs_logger.llm_call(
                provider="anthropic",
                model="claude-3",
                messages=[{"role": "user", "content": "Hello"}],
            ) as event:
                raise ValueError("Test error")

        log_file = temp_log_dir / "llm_calls.jsonl"
        events = [json.loads(line) for line in log_file.read_text().strip().split("\n")]
        assert len(events) == 1
        assert events[0]["event_type"] == "llm_call_error"
        assert events[0]["error_type"] == "ValueError"
        assert "Test error" in events[0]["error_message"]

    def test_agent_run_success(self, obs_logger, temp_log_dir):
        """Test logging successful agent run."""
        with obs_logger.agent_run(
            agent_name="diagnosis",
            agent_description="Clinical diagnosis agent",
            model_tier="standard",
            input_type="DiagnosisInput",
            input_summary="55yo male with knee pain",
        ) as event:
            event.output_type = "DiagnosisResult"
            event.output_summary = "Dx: Meniscal tear"
            event.llm_calls = 1

        log_file = temp_log_dir / "agent_runs.jsonl"
        assert log_file.exists()

        events = [json.loads(line) for line in log_file.read_text().strip().split("\n")]
        assert len(events) == 1
        assert events[0]["event_type"] == "agent_success"
        assert events[0]["agent_name"] == "diagnosis"
        assert events[0]["model_tier"] == "standard"
        assert events[0]["output_summary"] == "Dx: Meniscal tear"

    def test_agent_run_error(self, obs_logger, temp_log_dir):
        """Test logging failed agent run."""
        with pytest.raises(RuntimeError):
            with obs_logger.agent_run(
                agent_name="red_flag",
                model_tier="standard",
            ) as event:
                raise RuntimeError("Agent failed")

        log_file = temp_log_dir / "agent_runs.jsonl"
        events = [json.loads(line) for line in log_file.read_text().strip().split("\n")]
        assert len(events) == 1
        assert events[0]["event_type"] == "agent_error"
        assert events[0]["error_type"] == "RuntimeError"

    def test_orchestrator_run(self, obs_logger, temp_log_dir):
        """Test logging orchestrator run."""
        with obs_logger.orchestrator_run(
            discipline="PT",
            setting="outpatient",
            query_summary="Evaluate patient with back pain",
        ) as event:
            event.agents_called = ["red_flag", "diagnosis", "evidence", "plan"]
            event.total_llm_calls = 4
            event.has_red_flags = False
            event.diagnosis_confidence = 0.85
            event.qa_score = 0.9

        log_file = temp_log_dir / "orchestrator.jsonl"
        events = [json.loads(line) for line in log_file.read_text().strip().split("\n")]
        assert len(events) == 1
        assert events[0]["event_type"] == "orchestrator_success"
        assert events[0]["discipline"] == "PT"
        assert events[0]["diagnosis_confidence"] == 0.85

    def test_knowledge_search_logging(self, obs_logger, temp_log_dir):
        """Test logging knowledge base search."""
        obs_logger.log_knowledge_search(
            query="low back pain treatment",
            top_k=5,
            results_count=3,
            source_type="vector_store",
            top_result_score=0.85,
            top_result_source="APTA Guidelines",
            duration_ms=25.5,
        )

        log_file = temp_log_dir / "knowledge_search.jsonl"
        events = [json.loads(line) for line in log_file.read_text().strip().split("\n")]
        assert len(events) == 1
        assert events[0]["event_type"] == "knowledge_search"
        assert events[0]["query"] == "low back pain treatment"
        assert events[0]["results_count"] == 3
        assert events[0]["top_result_score"] == 0.85

    def test_content_truncation(self, obs_logger, temp_log_dir):
        """Test that long content is truncated."""
        long_content = "x" * 1000

        with obs_logger.llm_call(
            provider="test",
            model="test",
            messages=[{"role": "user", "content": long_content}],
        ) as event:
            event.response_content = long_content

        log_file = temp_log_dir / "llm_calls.jsonl"
        events = [json.loads(line) for line in log_file.read_text().strip().split("\n")]

        # Default max_content_length is 500
        assert len(events[0]["messages"][0]["content"]) <= 503  # 500 + "..."
        assert events[0]["response_content"].endswith("...")

    def test_full_content_logging(self, temp_log_dir):
        """Test logging full content when enabled."""
        logger = ObservabilityLogger(log_dir=temp_log_dir, log_full_content=True)
        long_content = "x" * 1000

        with logger.llm_call(
            provider="test",
            model="test",
            messages=[{"role": "user", "content": long_content}],
        ) as event:
            event.response_content = long_content

        log_file = temp_log_dir / "llm_calls.jsonl"
        events = [json.loads(line) for line in log_file.read_text().strip().split("\n")]

        # Full content should be preserved
        assert len(events[0]["messages"][0]["content"]) == 1000
        assert events[0]["response_content"] == long_content

    def test_session_id_propagation(self, obs_logger, temp_log_dir):
        """Test that session ID is added to events."""
        obs_logger.set_session_id("session-123")

        with obs_logger.llm_call(
            provider="test",
            model="test",
            messages=[],
        ) as event:
            pass

        log_file = temp_log_dir / "llm_calls.jsonl"
        events = [json.loads(line) for line in log_file.read_text().strip().split("\n")]
        assert events[0]["session_id"] == "session-123"

    def test_callback_on_event(self, obs_logger, temp_log_dir):
        """Test that callbacks are called on events."""
        callback_events = []

        def callback(event):
            callback_events.append(event)

        obs_logger.add_callback(callback)

        with obs_logger.llm_call(
            provider="test",
            model="test",
            messages=[],
        ) as event:
            pass

        assert len(callback_events) == 1
        assert callback_events[0].event_type == EventType.LLM_CALL_SUCCESS

    def test_get_recent_events(self, obs_logger, temp_log_dir):
        """Test retrieving recent events."""
        for i in range(5):
            with obs_logger.llm_call(
                provider=f"provider-{i}",
                model="test",
                messages=[],
            ):
                pass

        events = obs_logger.get_recent_events("llm", limit=3)
        assert len(events) == 3
        # Should be most recent
        assert events[-1]["provider"] == "provider-4"

    def test_get_stats(self, obs_logger, temp_log_dir):
        """Test getting statistics."""
        # Create some successful events
        for i in range(3):
            with obs_logger.llm_call(provider="test", model="test", messages=[]):
                pass

        # Create an error event
        with pytest.raises(ValueError):
            with obs_logger.llm_call(provider="test", model="test", messages=[]):
                raise ValueError("test")

        stats = obs_logger.get_stats("llm")
        assert stats["total"] == 4
        assert stats["errors"] == 1
        assert stats["error_rate"] == 0.25

    def test_llm_fallback_logging(self, obs_logger, temp_log_dir):
        """Test logging LLM fallback events."""
        obs_logger.log_llm_fallback(
            from_provider="local",
            to_provider="anthropic",
            reason="Connection timeout",
            request_id="req-123",
        )

        log_file = temp_log_dir / "llm_calls.jsonl"
        events = [json.loads(line) for line in log_file.read_text().strip().split("\n")]
        assert len(events) == 1
        assert events[0]["event_type"] == "llm_fallback"
        assert events[0]["is_fallback"] is True
        assert events[0]["fallback_reason"] == "Connection timeout"


class TestGetObservabilityLogger:
    """Tests for singleton getter."""

    def test_returns_singleton(self):
        """Test that get_observability_logger returns singleton."""
        # Reset singleton for test
        ObservabilityLogger._instance = None

        logger1 = get_observability_logger()
        logger2 = get_observability_logger()

        assert logger1 is logger2

    def test_singleton_persists(self):
        """Test singleton instance persists."""
        ObservabilityLogger._instance = None

        logger1 = get_observability_logger()
        logger1.set_session_id("test-session")

        logger2 = get_observability_logger()
        assert logger2._current_session_id == "test-session"


class TestEventModels:
    """Tests for event Pydantic models."""

    def test_llm_call_event_serialization(self):
        """Test LLMCallEvent serialization."""
        event = LLMCallEvent(
            event_type=EventType.LLM_CALL_SUCCESS,
            provider="anthropic",
            model="claude-3",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.5,
            max_tokens=100,
            response_content="response",
            input_tokens=10,
            output_tokens=5,
        )

        data = event.model_dump()
        assert data["provider"] == "anthropic"
        assert data["input_tokens"] == 10
        assert "timestamp" in data

    def test_agent_event_serialization(self):
        """Test AgentEvent serialization."""
        event = AgentEvent(
            event_type=EventType.AGENT_SUCCESS,
            agent_name="diagnosis",
            model_tier="standard",
            llm_calls=2,
            duration_ms=150.5,
        )

        json_str = event.model_dump_json()
        data = json.loads(json_str)
        assert data["agent_name"] == "diagnosis"
        assert data["llm_calls"] == 2
