"""Analytics for prompt optimization and effectiveness scoring."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from rehab_os.observability.events import EventType


@dataclass
class AgentMetrics:
    """Aggregated metrics for a single agent."""

    agent_name: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0

    # Confidence metrics
    confidence_scores: list[float] = field(default_factory=list)
    evidence_quality_scores: list[float] = field(default_factory=list)
    guideline_alignment_scores: list[float] = field(default_factory=list)

    # QA corrections
    qa_corrections: int = 0

    # Feedback stats
    accepted: int = 0
    rejected: int = 0
    needs_review: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_runs == 0:
            return 0.0
        return self.successful_runs / self.total_runs

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_runs == 0:
            return 0.0
        return self.total_latency_ms / self.successful_runs

    @property
    def avg_tokens_per_call(self) -> float:
        """Calculate average tokens per call."""
        if self.successful_runs == 0:
            return 0.0
        return self.total_tokens / self.successful_runs

    @property
    def avg_confidence(self) -> Optional[float]:
        """Calculate average confidence score."""
        if not self.confidence_scores:
            return None
        return sum(self.confidence_scores) / len(self.confidence_scores)

    @property
    def avg_evidence_quality(self) -> Optional[float]:
        """Calculate average evidence quality."""
        if not self.evidence_quality_scores:
            return None
        return sum(self.evidence_quality_scores) / len(self.evidence_quality_scores)

    @property
    def correction_rate(self) -> float:
        """Calculate QA correction rate."""
        if self.successful_runs == 0:
            return 0.0
        return self.qa_corrections / self.successful_runs

    @property
    def acceptance_rate(self) -> Optional[float]:
        """Calculate feedback acceptance rate."""
        total_feedback = self.accepted + self.rejected + self.needs_review
        if total_feedback == 0:
            return None
        return self.accepted / total_feedback


@dataclass
class LLMMetrics:
    """Aggregated metrics for LLM usage."""

    provider: str
    model: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    fallback_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def fallback_rate(self) -> float:
        """Calculate fallback rate."""
        if self.total_calls == 0:
            return 0.0
        return self.fallback_calls / self.total_calls

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency_ms / self.successful_calls

    @property
    def tokens_per_call(self) -> float:
        """Average tokens per successful call."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_tokens / self.successful_calls

    def cost_estimate(self, input_price_per_1k: float = 0.003, output_price_per_1k: float = 0.015) -> float:
        """Estimate cost based on token pricing."""
        input_cost = (self.total_input_tokens / 1000) * input_price_per_1k
        output_cost = (self.total_output_tokens / 1000) * output_price_per_1k
        return input_cost + output_cost


@dataclass
class PromptEffectivenessReport:
    """Comprehensive report on prompt effectiveness."""

    time_range: tuple[datetime, datetime]
    agent_metrics: dict[str, AgentMetrics]
    llm_metrics: dict[str, LLMMetrics]
    model_tier_stats: dict[str, dict[str, Any]]

    # Flags for prompts needing attention
    high_correction_agents: list[str] = field(default_factory=list)
    high_latency_agents: list[str] = field(default_factory=list)
    low_confidence_agents: list[str] = field(default_factory=list)
    high_fallback_providers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "time_range": {
                "start": self.time_range[0].isoformat(),
                "end": self.time_range[1].isoformat(),
            },
            "summary": {
                "total_agent_runs": sum(m.total_runs for m in self.agent_metrics.values()),
                "total_llm_calls": sum(m.total_calls for m in self.llm_metrics.values()),
                "total_tokens": sum(m.total_tokens for m in self.llm_metrics.values()),
                "estimated_cost": sum(m.cost_estimate() for m in self.llm_metrics.values()),
            },
            "agents": {
                name: {
                    "total_runs": m.total_runs,
                    "success_rate": m.success_rate,
                    "avg_latency_ms": m.avg_latency_ms,
                    "avg_tokens": m.avg_tokens_per_call,
                    "avg_confidence": m.avg_confidence,
                    "correction_rate": m.correction_rate,
                    "acceptance_rate": m.acceptance_rate,
                }
                for name, m in self.agent_metrics.items()
            },
            "llm_providers": {
                key: {
                    "total_calls": m.total_calls,
                    "success_rate": m.success_rate,
                    "fallback_rate": m.fallback_rate,
                    "avg_latency_ms": m.avg_latency_ms,
                    "total_tokens": m.total_tokens,
                    "cost_estimate": m.cost_estimate(),
                }
                for key, m in self.llm_metrics.items()
            },
            "model_tiers": self.model_tier_stats,
            "attention_needed": {
                "high_correction_rate": self.high_correction_agents,
                "high_latency": self.high_latency_agents,
                "low_confidence": self.low_confidence_agents,
                "high_fallback": self.high_fallback_providers,
            },
        }


class PromptAnalytics:
    """Analytics engine for prompt effectiveness scoring."""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize analytics engine.

        Args:
            log_dir: Directory containing log files
        """
        if log_dir is None:
            log_dir = Path("data/logs")
        self.log_dir = log_dir

    def _load_events(
        self,
        log_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """Load events from log file with optional time filtering."""
        log_file = self.log_dir / f"{log_type}.jsonl"
        if not log_file.exists():
            return []

        events = []
        with open(log_file) as f:
            for line in f:
                try:
                    event = json.loads(line)
                    timestamp_str = event.get("timestamp", "")
                    if not timestamp_str:
                        continue

                    timestamp = datetime.fromisoformat(timestamp_str)

                    # Make comparison timezone-aware/naive consistent
                    if timestamp.tzinfo is not None:
                        timestamp = timestamp.replace(tzinfo=None)

                    if start_time:
                        cmp_start = start_time.replace(tzinfo=None) if start_time.tzinfo else start_time
                        if timestamp < cmp_start:
                            continue
                    if end_time:
                        cmp_end = end_time.replace(tzinfo=None) if end_time.tzinfo else end_time
                        if timestamp > cmp_end:
                            continue

                    events.append(event)
                except (json.JSONDecodeError, ValueError):
                    continue

        return events

    def compute_agent_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[str, AgentMetrics]:
        """Compute metrics for each agent."""
        events = self._load_events("agent_runs", start_time, end_time)
        metrics: dict[str, AgentMetrics] = {}

        for event in events:
            agent_name = event.get("agent_name", "unknown")

            if agent_name not in metrics:
                metrics[agent_name] = AgentMetrics(agent_name=agent_name)

            m = metrics[agent_name]
            m.total_runs += 1

            event_type = event.get("event_type", "")
            if event_type == "agent_success":
                m.successful_runs += 1
                m.total_latency_ms += event.get("duration_ms", 0)
                m.total_tokens += event.get("total_tokens", 0) or 0

                # Confidence metrics
                if event.get("confidence_score") is not None:
                    m.confidence_scores.append(event["confidence_score"])
                if event.get("evidence_quality") is not None:
                    m.evidence_quality_scores.append(event["evidence_quality"])
                if event.get("guideline_alignment") is not None:
                    m.guideline_alignment_scores.append(event["guideline_alignment"])

                # Feedback stats
                feedback = event.get("feedback_status")
                if feedback == "accepted":
                    m.accepted += 1
                elif feedback == "rejected":
                    m.rejected += 1
                elif feedback == "needs_review":
                    m.needs_review += 1
            else:
                m.failed_runs += 1

        return metrics

    def compute_llm_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[str, LLMMetrics]:
        """Compute metrics for each LLM provider/model."""
        events = self._load_events("llm_calls", start_time, end_time)
        metrics: dict[str, LLMMetrics] = {}

        for event in events:
            provider = event.get("provider", "unknown")
            model = event.get("model", "unknown")
            key = f"{provider}/{model}"

            if key not in metrics:
                metrics[key] = LLMMetrics(provider=provider, model=model)

            m = metrics[key]
            m.total_calls += 1

            event_type = event.get("event_type", "")
            if event_type == "llm_call_success":
                m.successful_calls += 1
                m.total_latency_ms += event.get("duration_ms", 0)
                m.total_input_tokens += event.get("input_tokens", 0) or 0
                m.total_output_tokens += event.get("output_tokens", 0) or 0

                if event.get("is_fallback"):
                    m.fallback_calls += 1
            elif event_type == "llm_fallback":
                m.fallback_calls += 1
            else:
                m.failed_calls += 1

        return metrics

    def compute_model_tier_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[str, dict[str, Any]]:
        """Compute statistics by model tier."""
        events = self._load_events("agent_runs", start_time, end_time)
        tier_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "total_runs": 0,
                "successful_runs": 0,
                "total_tokens": 0,
                "total_latency_ms": 0,
                "agents": set(),
            }
        )

        for event in events:
            tier = event.get("model_tier", "unknown")
            stats = tier_stats[tier]
            stats["total_runs"] += 1
            stats["agents"].add(event.get("agent_name", "unknown"))

            if event.get("event_type") == "agent_success":
                stats["successful_runs"] += 1
                stats["total_tokens"] += event.get("total_tokens", 0) or 0
                stats["total_latency_ms"] += event.get("duration_ms", 0)

        # Convert sets to lists for JSON serialization
        for tier in tier_stats:
            tier_stats[tier]["agents"] = list(tier_stats[tier]["agents"])
            runs = tier_stats[tier]["successful_runs"]
            if runs > 0:
                tier_stats[tier]["avg_tokens"] = tier_stats[tier]["total_tokens"] / runs
                tier_stats[tier]["avg_latency_ms"] = tier_stats[tier]["total_latency_ms"] / runs
            else:
                tier_stats[tier]["avg_tokens"] = 0
                tier_stats[tier]["avg_latency_ms"] = 0

        return dict(tier_stats)

    def compute_qa_correction_rate(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[str, float]:
        """Compute QA correction rate per agent from orchestrator logs."""
        events = self._load_events("orchestrator", start_time, end_time)

        agent_runs: dict[str, int] = defaultdict(int)
        qa_corrections: dict[str, int] = defaultdict(int)

        for event in events:
            agents_called = event.get("agents_called", [])
            qa_score = event.get("qa_score")

            for agent in agents_called:
                if agent != "qa":
                    agent_runs[agent] += 1
                    # Low QA score indicates corrections needed
                    if qa_score is not None and qa_score < 0.7:
                        qa_corrections[agent] += 1

        return {
            agent: qa_corrections[agent] / runs if runs > 0 else 0
            for agent, runs in agent_runs.items()
        }

    def generate_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        correction_threshold: float = 0.2,
        latency_threshold_ms: float = 5000,
        confidence_threshold: float = 0.6,
        fallback_threshold: float = 0.3,
    ) -> PromptEffectivenessReport:
        """Generate comprehensive effectiveness report.

        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            correction_threshold: Flag agents with correction rate above this
            latency_threshold_ms: Flag agents with latency above this
            confidence_threshold: Flag agents with confidence below this
            fallback_threshold: Flag providers with fallback rate above this

        Returns:
            PromptEffectivenessReport with all metrics and flags
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=7)

        agent_metrics = self.compute_agent_metrics(start_time, end_time)
        llm_metrics = self.compute_llm_metrics(start_time, end_time)
        model_tier_stats = self.compute_model_tier_stats(start_time, end_time)
        qa_corrections = self.compute_qa_correction_rate(start_time, end_time)

        # Update agent metrics with QA corrections
        for agent, correction_rate in qa_corrections.items():
            if agent in agent_metrics:
                agent_metrics[agent].qa_corrections = int(
                    correction_rate * agent_metrics[agent].successful_runs
                )

        # Identify agents/providers needing attention
        high_correction = [
            name for name, m in agent_metrics.items()
            if m.correction_rate > correction_threshold
        ]
        high_latency = [
            name for name, m in agent_metrics.items()
            if m.avg_latency_ms > latency_threshold_ms
        ]
        low_confidence = [
            name for name, m in agent_metrics.items()
            if m.avg_confidence is not None and m.avg_confidence < confidence_threshold
        ]
        high_fallback = [
            key for key, m in llm_metrics.items()
            if m.fallback_rate > fallback_threshold
        ]

        return PromptEffectivenessReport(
            time_range=(start_time, end_time),
            agent_metrics=agent_metrics,
            llm_metrics=llm_metrics,
            model_tier_stats=model_tier_stats,
            high_correction_agents=high_correction,
            high_latency_agents=high_latency,
            low_confidence_agents=low_confidence,
            high_fallback_providers=high_fallback,
        )

    def export_report(
        self,
        output_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """Export effectiveness report to JSON file."""
        report = self.generate_report(start_time, end_time)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    def get_prompt_tuning_candidates(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """Identify prompts that need tuning based on multiple signals.

        Returns list of agents with reasons why they need attention.
        """
        report = self.generate_report(start_time, end_time)
        candidates = []

        for agent_name, metrics in report.agent_metrics.items():
            reasons = []

            if agent_name in report.high_correction_agents:
                reasons.append(f"High QA correction rate: {metrics.correction_rate:.1%}")

            if agent_name in report.high_latency_agents:
                reasons.append(f"High latency: {metrics.avg_latency_ms:.0f}ms")

            if agent_name in report.low_confidence_agents:
                reasons.append(f"Low confidence: {metrics.avg_confidence:.2f}")

            if metrics.acceptance_rate is not None and metrics.acceptance_rate < 0.7:
                reasons.append(f"Low acceptance rate: {metrics.acceptance_rate:.1%}")

            if reasons:
                candidates.append({
                    "agent": agent_name,
                    "reasons": reasons,
                    "metrics": {
                        "success_rate": metrics.success_rate,
                        "avg_latency_ms": metrics.avg_latency_ms,
                        "avg_confidence": metrics.avg_confidence,
                        "correction_rate": metrics.correction_rate,
                    },
                })

        # Sort by number of issues
        candidates.sort(key=lambda x: len(x["reasons"]), reverse=True)
        return candidates
