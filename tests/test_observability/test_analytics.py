"""Tests for prompt effectiveness analytics."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from rehab_os.observability.analytics import (
    AgentMetrics,
    LLMMetrics,
    PromptAnalytics,
    PromptEffectivenessReport,
)


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create temporary log directory with sample data."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def sample_agent_events():
    """Sample agent events for testing."""
    now = datetime.utcnow()
    return [
        {
            "event_type": "agent_success",
            "timestamp": now.isoformat(),
            "agent_name": "diagnosis",
            "model_tier": "standard",
            "duration_ms": 1500,
            "total_tokens": 500,
            "confidence_score": 0.85,
            "request_id": "req-001",
        },
        {
            "event_type": "agent_success",
            "timestamp": now.isoformat(),
            "agent_name": "diagnosis",
            "model_tier": "standard",
            "duration_ms": 2000,
            "total_tokens": 600,
            "confidence_score": 0.75,
            "feedback_status": "accepted",
            "request_id": "req-002",
        },
        {
            "event_type": "agent_error",
            "timestamp": now.isoformat(),
            "agent_name": "diagnosis",
            "model_tier": "standard",
            "error_type": "LLMError",
            "error_message": "Timeout",
            "request_id": "req-003",
        },
        {
            "event_type": "agent_success",
            "timestamp": now.isoformat(),
            "agent_name": "plan",
            "model_tier": "complex",
            "duration_ms": 3000,
            "total_tokens": 1200,
            "confidence_score": 0.9,
            "feedback_status": "rejected",
            "request_id": "req-004",
        },
    ]


@pytest.fixture
def sample_llm_events():
    """Sample LLM call events for testing."""
    now = datetime.utcnow()
    return [
        {
            "event_type": "llm_call_success",
            "timestamp": now.isoformat(),
            "provider": "local",
            "model": "llama-3",
            "duration_ms": 1000,
            "input_tokens": 200,
            "output_tokens": 300,
            "is_fallback": False,
        },
        {
            "event_type": "llm_call_success",
            "timestamp": now.isoformat(),
            "provider": "anthropic",
            "model": "claude-3",
            "duration_ms": 2000,
            "input_tokens": 250,
            "output_tokens": 400,
            "is_fallback": True,
        },
        {
            "event_type": "llm_call_error",
            "timestamp": now.isoformat(),
            "provider": "local",
            "model": "llama-3",
            "error_type": "LLMConnectionError",
        },
        {
            "event_type": "llm_fallback",
            "timestamp": now.isoformat(),
            "provider": "anthropic",
            "model": "claude-3",
            "fallback_reason": "Connection timeout",
        },
    ]


@pytest.fixture
def populated_log_dir(temp_log_dir, sample_agent_events, sample_llm_events):
    """Log directory populated with sample data."""
    # Write agent events
    with open(temp_log_dir / "agent_runs.jsonl", "w") as f:
        for event in sample_agent_events:
            f.write(json.dumps(event) + "\n")

    # Write LLM events
    with open(temp_log_dir / "llm_calls.jsonl", "w") as f:
        for event in sample_llm_events:
            f.write(json.dumps(event) + "\n")

    return temp_log_dir


class TestAgentMetrics:
    """Tests for AgentMetrics dataclass."""

    def test_success_rate(self):
        """Test success rate calculation."""
        metrics = AgentMetrics(
            agent_name="test",
            total_runs=10,
            successful_runs=8,
            failed_runs=2,
        )
        assert metrics.success_rate == 0.8

    def test_success_rate_zero_runs(self):
        """Test success rate with zero runs."""
        metrics = AgentMetrics(agent_name="test")
        assert metrics.success_rate == 0.0

    def test_avg_latency(self):
        """Test average latency calculation."""
        metrics = AgentMetrics(
            agent_name="test",
            successful_runs=5,
            total_latency_ms=5000,
        )
        assert metrics.avg_latency_ms == 1000

    def test_avg_confidence(self):
        """Test average confidence calculation."""
        metrics = AgentMetrics(
            agent_name="test",
            confidence_scores=[0.8, 0.9, 0.7],
        )
        assert abs(metrics.avg_confidence - 0.8) < 0.01

    def test_avg_confidence_empty(self):
        """Test average confidence with no scores."""
        metrics = AgentMetrics(agent_name="test")
        assert metrics.avg_confidence is None

    def test_acceptance_rate(self):
        """Test acceptance rate calculation."""
        metrics = AgentMetrics(
            agent_name="test",
            accepted=7,
            rejected=2,
            needs_review=1,
        )
        assert metrics.acceptance_rate == 0.7


class TestLLMMetrics:
    """Tests for LLMMetrics dataclass."""

    def test_total_tokens(self):
        """Test total tokens calculation."""
        metrics = LLMMetrics(
            provider="test",
            model="test",
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        assert metrics.total_tokens == 1500

    def test_fallback_rate(self):
        """Test fallback rate calculation."""
        metrics = LLMMetrics(
            provider="test",
            model="test",
            total_calls=10,
            fallback_calls=3,
        )
        assert metrics.fallback_rate == 0.3

    def test_cost_estimate(self):
        """Test cost estimation."""
        metrics = LLMMetrics(
            provider="test",
            model="test",
            total_input_tokens=10000,
            total_output_tokens=5000,
        )
        # Default pricing: $0.003/1k input, $0.015/1k output
        expected = (10 * 0.003) + (5 * 0.015)
        assert abs(metrics.cost_estimate() - expected) < 0.001


class TestPromptAnalytics:
    """Tests for PromptAnalytics engine."""

    def test_compute_agent_metrics(self, populated_log_dir):
        """Test computing agent metrics from logs."""
        analytics = PromptAnalytics(populated_log_dir)
        metrics = analytics.compute_agent_metrics()

        assert "diagnosis" in metrics
        assert "plan" in metrics

        diag = metrics["diagnosis"]
        assert diag.total_runs == 3
        assert diag.successful_runs == 2
        assert diag.failed_runs == 1
        assert len(diag.confidence_scores) == 2

    def test_compute_llm_metrics(self, populated_log_dir):
        """Test computing LLM metrics from logs."""
        analytics = PromptAnalytics(populated_log_dir)
        metrics = analytics.compute_llm_metrics()

        assert "local/llama-3" in metrics
        assert "anthropic/claude-3" in metrics

        local = metrics["local/llama-3"]
        assert local.total_calls == 2
        assert local.successful_calls == 1
        assert local.failed_calls == 1

    def test_compute_model_tier_stats(self, populated_log_dir):
        """Test computing model tier statistics."""
        analytics = PromptAnalytics(populated_log_dir)
        stats = analytics.compute_model_tier_stats()

        assert "standard" in stats
        assert "complex" in stats
        assert stats["standard"]["total_runs"] == 3
        assert "diagnosis" in stats["standard"]["agents"]

    def test_generate_report(self, populated_log_dir):
        """Test generating full effectiveness report."""
        analytics = PromptAnalytics(populated_log_dir)
        # Use wide time range to include all test events
        end_time = datetime.now() + timedelta(days=1)
        start_time = end_time - timedelta(days=30)
        report = analytics.generate_report(start_time, end_time)

        assert isinstance(report, PromptEffectivenessReport)
        assert len(report.agent_metrics) > 0
        assert len(report.llm_metrics) > 0

    def test_report_to_dict(self, populated_log_dir):
        """Test report serialization to dict."""
        analytics = PromptAnalytics(populated_log_dir)
        report = analytics.generate_report()
        report_dict = report.to_dict()

        assert "time_range" in report_dict
        assert "summary" in report_dict
        assert "agents" in report_dict
        assert "attention_needed" in report_dict

    def test_get_tuning_candidates(self, populated_log_dir):
        """Test identifying prompt tuning candidates."""
        analytics = PromptAnalytics(populated_log_dir)
        candidates = analytics.get_prompt_tuning_candidates()

        # Should return list even if empty
        assert isinstance(candidates, list)

    def test_empty_log_dir(self, temp_log_dir):
        """Test analytics with empty log directory."""
        analytics = PromptAnalytics(temp_log_dir)
        metrics = analytics.compute_agent_metrics()

        assert metrics == {}

    def test_time_filtering(self, temp_log_dir):
        """Test time-based filtering of events."""
        # Create events at different times
        old_time = (datetime.utcnow() - timedelta(days=10)).isoformat()
        recent_time = datetime.utcnow().isoformat()

        events = [
            {"event_type": "agent_success", "timestamp": old_time, "agent_name": "old"},
            {"event_type": "agent_success", "timestamp": recent_time, "agent_name": "recent"},
        ]

        with open(temp_log_dir / "agent_runs.jsonl", "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

        analytics = PromptAnalytics(temp_log_dir)

        # Filter to last 7 days
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        metrics = analytics.compute_agent_metrics(start_time, end_time)

        # Should only include recent event
        assert "recent" in metrics
        assert "old" not in metrics


class TestPromptEffectivenessReport:
    """Tests for report generation."""

    def test_attention_needed_flags(self, populated_log_dir):
        """Test that attention flags are populated."""
        analytics = PromptAnalytics(populated_log_dir)
        report = analytics.generate_report(
            correction_threshold=0.0,  # Flag any corrections
            latency_threshold_ms=1000,  # Low threshold to flag
        )

        # With low thresholds, should flag some agents
        attention = report.to_dict()["attention_needed"]
        assert isinstance(attention["high_correction_rate"], list)
        assert isinstance(attention["high_latency"], list)

    def test_export_report(self, populated_log_dir, tmp_path):
        """Test exporting report to JSON."""
        analytics = PromptAnalytics(populated_log_dir)
        output_path = tmp_path / "report.json"

        analytics.export_report(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "agents" in data
