"""Prompt optimization using logs and feedback."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from rehab_os.observability import PromptAnalytics

logger = logging.getLogger(__name__)


class PromptVersion(BaseModel):
    """A versioned prompt with metadata."""

    version: str
    agent_name: str
    prompt_text: str
    created_at: datetime = Field(default_factory=datetime.now)
    parent_version: Optional[str] = None

    # Performance metrics at time of creation
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)

    # Optimization metadata
    optimization_reason: Optional[str] = None
    changes_made: list[str] = Field(default_factory=list)

    # A/B test results (populated after evaluation)
    test_results: Optional[dict[str, Any]] = None
    is_active: bool = False


class OptimizationResult(BaseModel):
    """Result of a prompt optimization attempt."""

    agent_name: str
    original_version: str
    new_version: str
    optimization_type: str  # feedback_based, metric_based, llm_suggested

    changes: list[str]
    reasoning: str

    # Metrics comparison
    before_metrics: dict[str, Any]
    predicted_improvement: dict[str, float]

    # Status
    status: str = "pending"  # pending, testing, accepted, rejected
    created_at: datetime = Field(default_factory=datetime.now)


@dataclass
class FeedbackSummary:
    """Aggregated feedback for an agent."""

    agent_name: str
    total_annotations: int = 0
    accepted: int = 0
    rejected: int = 0
    needs_review: int = 0

    common_issues: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)
    common_tags: list[str] = field(default_factory=list)

    @property
    def rejection_rate(self) -> float:
        if self.total_annotations == 0:
            return 0.0
        return self.rejected / self.total_annotations


class PromptOptimizer:
    """Optimizes agent prompts based on logs, feedback, and metrics.

    Supports multiple optimization strategies:
    1. Feedback-based: Uses human annotations to identify issues
    2. Metric-based: Uses performance metrics to identify problems
    3. LLM-suggested: Uses an LLM to suggest improvements
    """

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
    ):
        """Initialize prompt optimizer.

        Args:
            prompts_dir: Directory for storing prompt versions
            logs_dir: Directory containing observability logs
        """
        self.prompts_dir = prompts_dir or Path("data/prompts")
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = logs_dir or Path("data/logs")
        self.analytics = PromptAnalytics(self.logs_dir)

        # Load existing prompt versions
        self._versions: dict[str, list[PromptVersion]] = {}
        self._load_versions()

    def _load_versions(self) -> None:
        """Load existing prompt versions from disk."""
        versions_file = self.prompts_dir / "versions.jsonl"
        if not versions_file.exists():
            return

        with open(versions_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    version = PromptVersion(**data)
                    agent = version.agent_name
                    if agent not in self._versions:
                        self._versions[agent] = []
                    self._versions[agent].append(version)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to load prompt version: {e}")

    def _save_version(self, version: PromptVersion) -> None:
        """Save a prompt version to disk."""
        versions_file = self.prompts_dir / "versions.jsonl"
        with open(versions_file, "a") as f:
            f.write(version.model_dump_json() + "\n")

        # Also save the full prompt text separately
        prompt_file = self.prompts_dir / f"{version.agent_name}_{version.version}.txt"
        with open(prompt_file, "w") as f:
            f.write(version.prompt_text)

    def get_current_prompt(self, agent_name: str) -> Optional[PromptVersion]:
        """Get the currently active prompt for an agent."""
        if agent_name not in self._versions:
            return None

        active = [v for v in self._versions[agent_name] if v.is_active]
        if active:
            return active[-1]

        # Return latest version if none active
        return self._versions[agent_name][-1] if self._versions[agent_name] else None

    def get_version_history(self, agent_name: str) -> list[PromptVersion]:
        """Get all prompt versions for an agent."""
        return self._versions.get(agent_name, [])

    def register_prompt(
        self,
        agent_name: str,
        prompt_text: str,
        version: Optional[str] = None,
        is_active: bool = True,
    ) -> PromptVersion:
        """Register a new prompt version.

        Args:
            agent_name: Name of the agent
            prompt_text: The prompt text
            version: Version string (auto-generated if not provided)
            is_active: Whether this is the active version

        Returns:
            The created PromptVersion
        """
        if agent_name not in self._versions:
            self._versions[agent_name] = []

        # Auto-generate version if not provided
        if version is None:
            existing = len(self._versions[agent_name])
            version = f"v{existing + 1}.0"

        # Get current metrics as baseline
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        metrics = self.analytics.compute_agent_metrics(start_time, end_time)
        baseline = {}
        if agent_name in metrics:
            m = metrics[agent_name]
            baseline = {
                "success_rate": m.success_rate,
                "avg_latency_ms": m.avg_latency_ms,
                "avg_confidence": m.avg_confidence,
                "correction_rate": m.correction_rate,
            }

        # Deactivate other versions if this is active
        if is_active:
            for v in self._versions[agent_name]:
                v.is_active = False

        # Get parent version
        parent = self.get_current_prompt(agent_name)
        parent_version = parent.version if parent else None

        prompt_version = PromptVersion(
            version=version,
            agent_name=agent_name,
            prompt_text=prompt_text,
            parent_version=parent_version,
            baseline_metrics=baseline,
            is_active=is_active,
        )

        self._versions[agent_name].append(prompt_version)
        self._save_version(prompt_version)

        logger.info(f"Registered prompt {agent_name} {version}")
        return prompt_version

    def analyze_feedback(
        self,
        agent_name: Optional[str] = None,
        days: int = 30,
    ) -> dict[str, FeedbackSummary]:
        """Analyze feedback annotations to identify improvement areas.

        Args:
            agent_name: Specific agent to analyze (None for all)
            days: Number of days of history to analyze

        Returns:
            Dict mapping agent names to feedback summaries
        """
        feedback_file = self.logs_dir / "feedback.jsonl"
        if not feedback_file.exists():
            return {}

        cutoff = datetime.now() - timedelta(days=days)
        summaries: dict[str, FeedbackSummary] = {}

        with open(feedback_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)

                    # Filter by agent if specified
                    entry_agent = entry.get("agent_name", "unknown")
                    if agent_name and entry_agent != agent_name:
                        continue

                    # Filter by time
                    timestamp = datetime.fromisoformat(entry.get("timestamp", ""))
                    if timestamp < cutoff:
                        continue

                    # Initialize summary if needed
                    if entry_agent not in summaries:
                        summaries[entry_agent] = FeedbackSummary(agent_name=entry_agent)

                    summary = summaries[entry_agent]
                    summary.total_annotations += 1

                    status = entry.get("status", "")
                    if status == "accepted":
                        summary.accepted += 1
                    elif status == "rejected":
                        summary.rejected += 1
                    elif status == "needs_review":
                        summary.needs_review += 1

                    # Collect improvement suggestions
                    if entry.get("suggested_improvement"):
                        summary.improvement_suggestions.append(
                            entry["suggested_improvement"]
                        )

                    # Collect tags
                    summary.common_tags.extend(entry.get("tags", []))

                    # Notes often contain issue descriptions
                    if entry.get("notes") and status == "rejected":
                        summary.common_issues.append(entry["notes"])

                except (json.JSONDecodeError, ValueError):
                    continue

        return summaries

    def generate_optimization(
        self,
        agent_name: str,
        strategy: str = "feedback_based",
    ) -> Optional[OptimizationResult]:
        """Generate a prompt optimization suggestion.

        Args:
            agent_name: Agent to optimize
            strategy: Optimization strategy (feedback_based, metric_based)

        Returns:
            OptimizationResult with suggested changes, or None if no optimization needed
        """
        current = self.get_current_prompt(agent_name)
        if not current:
            logger.warning(f"No registered prompt for {agent_name}")
            return None

        # Get current metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        metrics = self.analytics.compute_agent_metrics(start_time, end_time)
        agent_metrics = metrics.get(agent_name)

        if not agent_metrics:
            logger.warning(f"No metrics available for {agent_name}")
            return None

        changes = []
        reasoning_parts = []
        predicted_improvement = {}

        if strategy == "feedback_based":
            feedback = self.analyze_feedback(agent_name, days=30)
            if agent_name not in feedback:
                return None

            summary = feedback[agent_name]

            if summary.rejection_rate > 0.2:
                changes.append("Address common rejection reasons in prompt")
                reasoning_parts.append(
                    f"High rejection rate ({summary.rejection_rate:.1%})"
                )
                predicted_improvement["acceptance_rate"] = 0.1

            if summary.improvement_suggestions:
                # Group similar suggestions
                unique_suggestions = list(set(summary.improvement_suggestions[:5]))
                changes.append(f"Incorporate suggestions: {', '.join(unique_suggestions[:3])}")
                reasoning_parts.append(
                    f"Found {len(summary.improvement_suggestions)} improvement suggestions"
                )

            if summary.common_issues:
                changes.append("Add guidance to avoid common issues")
                reasoning_parts.append(
                    f"Found {len(summary.common_issues)} reported issues"
                )

        elif strategy == "metric_based":
            # Check for performance issues
            if agent_metrics.avg_confidence and agent_metrics.avg_confidence < 0.7:
                changes.append("Add more specific guidance to improve confidence")
                reasoning_parts.append(
                    f"Low confidence ({agent_metrics.avg_confidence:.2f})"
                )
                predicted_improvement["confidence"] = 0.1

            if agent_metrics.correction_rate > 0.2:
                changes.append("Align output format with QA expectations")
                reasoning_parts.append(
                    f"High QA correction rate ({agent_metrics.correction_rate:.1%})"
                )
                predicted_improvement["correction_rate"] = -0.1

            if agent_metrics.avg_latency_ms > 5000:
                changes.append("Simplify prompt structure for faster responses")
                reasoning_parts.append(
                    f"High latency ({agent_metrics.avg_latency_ms:.0f}ms)"
                )
                predicted_improvement["latency"] = -500

        if not changes:
            return None

        # Generate new version number
        new_version = f"v{len(self._versions.get(agent_name, [])) + 1}.0"

        return OptimizationResult(
            agent_name=agent_name,
            original_version=current.version,
            new_version=new_version,
            optimization_type=strategy,
            changes=changes,
            reasoning="; ".join(reasoning_parts),
            before_metrics={
                "success_rate": agent_metrics.success_rate,
                "avg_confidence": agent_metrics.avg_confidence,
                "correction_rate": agent_metrics.correction_rate,
                "avg_latency_ms": agent_metrics.avg_latency_ms,
            },
            predicted_improvement=predicted_improvement,
        )

    async def apply_optimization_with_llm(
        self,
        optimization: OptimizationResult,
        llm_router: Any,
    ) -> PromptVersion:
        """Use an LLM to apply optimization changes to a prompt.

        Args:
            optimization: The optimization to apply
            llm_router: LLM router for generating new prompt

        Returns:
            New PromptVersion with optimized prompt
        """
        from rehab_os.llm import Message, MessageRole

        current = self.get_current_prompt(optimization.agent_name)
        if not current:
            raise ValueError(f"No current prompt for {optimization.agent_name}")

        # Build optimization prompt
        system_prompt = """You are a prompt engineering expert specializing in clinical AI systems.
Your task is to improve an agent prompt based on specific feedback and metrics.

Guidelines:
- Maintain the core clinical accuracy and safety focus
- Keep the prompt structure clear and organized
- Address the specific issues identified
- Don't remove important clinical guidance
- Keep changes focused and minimal"""

        user_prompt = f"""Current prompt for the {optimization.agent_name} agent:

```
{current.prompt_text}
```

Issues identified:
{optimization.reasoning}

Required changes:
{chr(10).join(f'- {c}' for c in optimization.changes)}

Please provide an improved version of this prompt that addresses these issues.
Return ONLY the improved prompt text, nothing else."""

        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_prompt),
        ]

        response = await llm_router.complete(
            messages,
            temperature=0.3,
            max_tokens=4000,
        )

        new_prompt_text = response.content.strip()

        # Register the new version
        new_version = self.register_prompt(
            agent_name=optimization.agent_name,
            prompt_text=new_prompt_text,
            version=optimization.new_version,
            is_active=False,  # Don't activate until tested
        )

        new_version.optimization_reason = optimization.reasoning
        new_version.changes_made = optimization.changes

        # Update optimization status
        optimization.status = "testing"

        return new_version

    def get_optimization_candidates(
        self,
        min_rejection_rate: float = 0.2,
        min_correction_rate: float = 0.2,
        max_confidence: float = 0.7,
    ) -> list[str]:
        """Get list of agents that could benefit from optimization.

        Returns:
            List of agent names needing optimization
        """
        candidates = []

        # Check metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        metrics = self.analytics.compute_agent_metrics(start_time, end_time)

        for agent_name, m in metrics.items():
            needs_optimization = False

            if m.correction_rate > min_correction_rate:
                needs_optimization = True

            if m.avg_confidence and m.avg_confidence < max_confidence:
                needs_optimization = True

            if needs_optimization:
                candidates.append(agent_name)

        # Check feedback
        feedback = self.analyze_feedback(days=30)
        for agent_name, summary in feedback.items():
            if summary.rejection_rate > min_rejection_rate:
                if agent_name not in candidates:
                    candidates.append(agent_name)

        return candidates

    def export_optimization_report(
        self,
        output_path: Path,
    ) -> None:
        """Export a report of all optimization opportunities."""
        candidates = self.get_optimization_candidates()

        report = {
            "generated_at": datetime.now().isoformat(),
            "candidates": [],
        }

        for agent_name in candidates:
            # Try both strategies
            for strategy in ["feedback_based", "metric_based"]:
                opt = self.generate_optimization(agent_name, strategy)
                if opt:
                    report["candidates"].append(opt.model_dump())

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
