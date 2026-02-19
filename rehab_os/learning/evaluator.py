"""Agent evaluation and A/B testing framework."""

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EvalTestCase(BaseModel):
    """A test case for agent evaluation."""

    id: str
    agent_name: str
    input_data: dict[str, Any]
    expected_output: Optional[dict[str, Any]] = None

    # Quality criteria
    required_fields: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)

    # Reference answer for comparison
    reference_answer: Optional[str] = None

    # Tags for filtering
    tags: list[str] = Field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard


class EvaluationResult(BaseModel):
    """Result of evaluating a single test case."""

    test_case_id: str
    agent_name: str
    prompt_version: str

    # Execution
    success: bool
    error_message: Optional[str] = None
    duration_ms: float = 0

    # Output analysis
    output: Optional[dict[str, Any]] = None
    output_summary: Optional[str] = None

    # Quality scores (0-1)
    completeness_score: float = 0  # Has all required fields
    accuracy_score: Optional[float] = None  # Matches expected/reference
    format_score: float = 0  # No forbidden patterns

    # Agent-reported confidence
    confidence_score: Optional[float] = None

    # Timestamps
    evaluated_at: datetime = Field(default_factory=datetime.now)


class EvaluationRun(BaseModel):
    """A complete evaluation run across multiple test cases."""

    run_id: str
    agent_name: str
    prompt_version: str
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Configuration
    test_suite: str
    total_cases: int = 0
    completed_cases: int = 0

    # Results
    results: list[EvaluationResult] = Field(default_factory=list)

    # Aggregated metrics
    success_rate: float = 0
    avg_completeness: float = 0
    avg_accuracy: Optional[float] = None
    avg_format_score: float = 0
    avg_confidence: Optional[float] = None
    avg_duration_ms: float = 0

    # Status
    status: str = "running"  # running, completed, failed

    def compute_metrics(self) -> None:
        """Compute aggregated metrics from results."""
        if not self.results:
            return

        n = len(self.results)
        successful = [r for r in self.results if r.success]

        self.success_rate = len(successful) / n if n > 0 else 0
        self.avg_completeness = sum(r.completeness_score for r in self.results) / n
        self.avg_format_score = sum(r.format_score for r in self.results) / n
        self.avg_duration_ms = sum(r.duration_ms for r in self.results) / n

        # Accuracy (only for results with scores)
        accuracy_scores = [r.accuracy_score for r in self.results if r.accuracy_score is not None]
        if accuracy_scores:
            self.avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)

        # Confidence (only for results with scores)
        confidence_scores = [r.confidence_score for r in self.results if r.confidence_score is not None]
        if confidence_scores:
            self.avg_confidence = sum(confidence_scores) / len(confidence_scores)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing two prompt versions."""

    agent_name: str
    version_a: str  # Control
    version_b: str  # Treatment
    test_suite: str
    sample_size: int = 100
    split_ratio: float = 0.5  # Fraction going to version B


class ABTestResult(BaseModel):
    """Result of an A/B test between two prompt versions."""

    agent_name: str
    version_a: str
    version_b: str

    run_a: EvaluationRun
    run_b: EvaluationRun

    # Statistical comparison
    success_rate_diff: float = 0  # B - A
    completeness_diff: float = 0
    accuracy_diff: Optional[float] = None
    confidence_diff: Optional[float] = None
    latency_diff: float = 0

    # Recommendation
    winner: Optional[str] = None  # version_a, version_b, or None (no significant diff)
    confidence_level: float = 0  # Statistical confidence in winner

    completed_at: datetime = Field(default_factory=datetime.now)


class AgentEvaluator:
    """Evaluates agent performance using test suites."""

    def __init__(
        self,
        test_suites_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ):
        """Initialize evaluator.

        Args:
            test_suites_dir: Directory containing test case files
            results_dir: Directory for storing evaluation results
        """
        self.test_suites_dir = test_suites_dir or Path("data/test_suites")
        self.test_suites_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = results_dir or Path("data/evaluation_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._test_cases: dict[str, list[EvalTestCase]] = {}

    def load_test_suite(self, suite_name: str) -> list[EvalTestCase]:
        """Load test cases from a suite file."""
        if suite_name in self._test_cases:
            return self._test_cases[suite_name]

        suite_file = self.test_suites_dir / f"{suite_name}.jsonl"
        if not suite_file.exists():
            logger.warning(f"Test suite not found: {suite_file}")
            return []

        cases = []
        with open(suite_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    cases.append(EvalTestCase(**data))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to load test case: {e}")

        self._test_cases[suite_name] = cases
        return cases

    def create_test_suite(
        self,
        suite_name: str,
        cases: list[EvalTestCase],
    ) -> None:
        """Create a new test suite."""
        suite_file = self.test_suites_dir / f"{suite_name}.jsonl"
        with open(suite_file, "w") as f:
            for case in cases:
                f.write(case.model_dump_json() + "\n")

        self._test_cases[suite_name] = cases
        logger.info(f"Created test suite {suite_name} with {len(cases)} cases")

    async def evaluate_agent(
        self,
        agent: Any,
        test_suite: str,
        prompt_version: str,
        max_cases: Optional[int] = None,
    ) -> EvaluationRun:
        """Run evaluation on an agent.

        Args:
            agent: The agent to evaluate
            test_suite: Name of test suite to use
            prompt_version: Version string of the prompt being tested
            max_cases: Maximum number of test cases (None for all)

        Returns:
            EvaluationRun with all results
        """
        import uuid
        import time

        cases = self.load_test_suite(test_suite)
        if not cases:
            raise ValueError(f"No test cases found in suite: {test_suite}")

        if max_cases:
            cases = cases[:max_cases]

        run = EvaluationRun(
            run_id=str(uuid.uuid4())[:8],
            agent_name=agent.name,
            prompt_version=prompt_version,
            test_suite=test_suite,
            total_cases=len(cases),
        )

        for case in cases:
            result = await self._evaluate_single(agent, case, prompt_version)
            run.results.append(result)
            run.completed_cases += 1

        run.compute_metrics()
        run.completed_at = datetime.now()
        run.status = "completed"

        # Save results
        self._save_run(run)

        return run

    async def _evaluate_single(
        self,
        agent: Any,
        case: EvalTestCase,
        prompt_version: str,
    ) -> EvaluationResult:
        """Evaluate a single test case."""
        import time
        from rehab_os.agents.base import AgentContext

        result = EvaluationResult(
            test_case_id=case.id,
            agent_name=agent.name,
            prompt_version=prompt_version,
        )

        try:
            # Prepare input
            input_model = agent.output_schema  # Get expected input type
            # For now, assume input_data can be passed directly
            context = AgentContext()

            start_time = time.time()

            # Run agent
            output = await agent.run(case.input_data, context)

            result.duration_ms = (time.time() - start_time) * 1000
            result.success = True

            # Convert output to dict for analysis
            if hasattr(output, "model_dump"):
                result.output = output.model_dump()
            else:
                result.output = dict(output) if output else {}

            # Extract confidence if available
            if hasattr(output, "confidence"):
                result.confidence_score = output.confidence

            # Compute quality scores
            result.completeness_score = self._score_completeness(
                result.output, case.required_fields
            )
            result.format_score = self._score_format(
                result.output, case.forbidden_patterns
            )

            # Compute accuracy if reference available
            if case.expected_output:
                result.accuracy_score = self._score_accuracy(
                    result.output, case.expected_output
                )

            # Generate summary
            result.output_summary = self._summarize_output(result.output)

        except Exception as e:
            result.success = False
            result.error_message = str(e)[:500]
            logger.error(f"Evaluation failed for {case.id}: {e}")

        return result

    def _score_completeness(
        self,
        output: dict[str, Any],
        required_fields: list[str],
    ) -> float:
        """Score how many required fields are present and non-empty."""
        if not required_fields:
            return 1.0

        present = 0
        for field in required_fields:
            # Handle nested fields with dot notation
            parts = field.split(".")
            value = output
            try:
                for part in parts:
                    value = value[part]
                if value is not None and value != "":
                    present += 1
            except (KeyError, TypeError):
                pass

        return present / len(required_fields)

    def _score_format(
        self,
        output: dict[str, Any],
        forbidden_patterns: list[str],
    ) -> float:
        """Score based on absence of forbidden patterns."""
        if not forbidden_patterns:
            return 1.0

        import re

        output_str = json.dumps(output).lower()
        violations = 0

        for pattern in forbidden_patterns:
            if re.search(pattern.lower(), output_str):
                violations += 1

        return 1.0 - (violations / len(forbidden_patterns))

    def _score_accuracy(
        self,
        output: dict[str, Any],
        expected: dict[str, Any],
    ) -> float:
        """Score accuracy against expected output."""
        if not expected:
            return 1.0

        matches = 0
        total = 0

        for key, expected_value in expected.items():
            total += 1
            if key in output:
                actual_value = output[key]
                # Flexible matching
                if actual_value == expected_value:
                    matches += 1
                elif str(actual_value).lower() == str(expected_value).lower():
                    matches += 0.8  # Partial credit for case differences
                elif expected_value in str(actual_value):
                    matches += 0.5  # Partial credit for containment

        return matches / total if total > 0 else 1.0

    def _summarize_output(self, output: dict[str, Any]) -> str:
        """Generate a brief summary of the output."""
        parts = []

        # Common fields to include in summary
        summary_fields = [
            "primary_diagnosis",
            "is_safe_to_treat",
            "clinical_summary",
            "overall_quality",
        ]

        for field in summary_fields:
            if field in output and output[field] is not None:
                value = output[field]
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                parts.append(f"{field}: {value}")

        return "; ".join(parts[:3]) if parts else "No summary available"

    def _save_run(self, run: EvaluationRun) -> None:
        """Save evaluation run to disk."""
        run_file = self.results_dir / f"{run.agent_name}_{run.run_id}.json"
        with open(run_file, "w") as f:
            f.write(run.model_dump_json(indent=2))

    async def run_ab_test(
        self,
        config: ABTestConfig,
        agent_factory: Any,  # Callable that creates agent with specific prompt
    ) -> ABTestResult:
        """Run A/B test between two prompt versions.

        Args:
            config: A/B test configuration
            agent_factory: Factory function(prompt_version) -> agent

        Returns:
            ABTestResult with comparison
        """
        cases = self.load_test_suite(config.test_suite)
        if not cases:
            raise ValueError(f"No test cases in suite: {config.test_suite}")

        # Sample and split cases
        sample = random.sample(cases, min(config.sample_size, len(cases)))
        split_idx = int(len(sample) * config.split_ratio)

        cases_a = sample[:split_idx]
        cases_b = sample[split_idx:]

        # Create temporary test suites
        suite_a = f"_ab_test_a_{config.agent_name}"
        suite_b = f"_ab_test_b_{config.agent_name}"
        self.create_test_suite(suite_a, cases_a)
        self.create_test_suite(suite_b, cases_b)

        # Run evaluations
        agent_a = agent_factory(config.version_a)
        agent_b = agent_factory(config.version_b)

        run_a = await self.evaluate_agent(agent_a, suite_a, config.version_a)
        run_b = await self.evaluate_agent(agent_b, suite_b, config.version_b)

        # Compare results
        result = ABTestResult(
            agent_name=config.agent_name,
            version_a=config.version_a,
            version_b=config.version_b,
            run_a=run_a,
            run_b=run_b,
        )

        # Compute differences (B - A)
        result.success_rate_diff = run_b.success_rate - run_a.success_rate
        result.completeness_diff = run_b.avg_completeness - run_a.avg_completeness
        result.latency_diff = run_b.avg_duration_ms - run_a.avg_duration_ms

        if run_a.avg_accuracy is not None and run_b.avg_accuracy is not None:
            result.accuracy_diff = run_b.avg_accuracy - run_a.avg_accuracy

        if run_a.avg_confidence is not None and run_b.avg_confidence is not None:
            result.confidence_diff = run_b.avg_confidence - run_a.avg_confidence

        # Determine winner (simple heuristic - could use statistical tests)
        score_a = run_a.success_rate + run_a.avg_completeness
        score_b = run_b.success_rate + run_b.avg_completeness

        if abs(score_b - score_a) > 0.1:  # Significant difference
            result.winner = config.version_b if score_b > score_a else config.version_a
            result.confidence_level = abs(score_b - score_a)
        else:
            result.winner = None
            result.confidence_level = 0

        # Save A/B test result
        self._save_ab_result(result)

        return result

    def _save_ab_result(self, result: ABTestResult) -> None:
        """Save A/B test result."""
        result_file = self.results_dir / f"ab_test_{result.agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, "w") as f:
            f.write(result.model_dump_json(indent=2))

    def get_evaluation_history(
        self,
        agent_name: str,
        limit: int = 10,
    ) -> list[EvaluationRun]:
        """Get recent evaluation runs for an agent."""
        runs = []

        for result_file in sorted(self.results_dir.glob(f"{agent_name}_*.json"), reverse=True):
            if len(runs) >= limit:
                break

            try:
                with open(result_file) as f:
                    data = json.load(f)
                runs.append(EvaluationRun(**data))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load evaluation run: {e}")

        return runs

    def compare_versions(
        self,
        agent_name: str,
        version_a: str,
        version_b: str,
    ) -> dict[str, Any]:
        """Compare metrics between two versions from historical runs."""
        history = self.get_evaluation_history(agent_name, limit=50)

        runs_a = [r for r in history if r.prompt_version == version_a]
        runs_b = [r for r in history if r.prompt_version == version_b]

        if not runs_a or not runs_b:
            return {"error": "Insufficient data for comparison"}

        # Average metrics across runs
        def avg_metrics(runs: list[EvaluationRun]) -> dict[str, float]:
            return {
                "success_rate": sum(r.success_rate for r in runs) / len(runs),
                "completeness": sum(r.avg_completeness for r in runs) / len(runs),
                "duration_ms": sum(r.avg_duration_ms for r in runs) / len(runs),
            }

        metrics_a = avg_metrics(runs_a)
        metrics_b = avg_metrics(runs_b)

        return {
            "version_a": {
                "version": version_a,
                "num_runs": len(runs_a),
                **metrics_a,
            },
            "version_b": {
                "version": version_b,
                "num_runs": len(runs_b),
                **metrics_b,
            },
            "differences": {
                "success_rate": metrics_b["success_rate"] - metrics_a["success_rate"],
                "completeness": metrics_b["completeness"] - metrics_a["completeness"],
                "duration_ms": metrics_b["duration_ms"] - metrics_a["duration_ms"],
            },
        }
