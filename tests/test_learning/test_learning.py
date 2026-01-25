"""Tests for learning module components."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from rehab_os.learning.prompt_optimizer import (
    PromptOptimizer,
    PromptVersion,
    OptimizationResult,
    FeedbackSummary,
)
from rehab_os.learning.evaluator import (
    AgentEvaluator,
    EvaluationRun,
    EvaluationResult,
    TestCase,
    ABTestConfig,
)
from rehab_os.learning.scheduler import (
    LearningScheduler,
    ScheduledTask,
    TaskType,
    TaskStatus,
)


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for testing."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    return {"prompts": prompts_dir, "logs": logs_dir}


@pytest.fixture
def sample_feedback_data(temp_dirs):
    """Create sample feedback data."""
    feedback = [
        {
            "event_id": "req-001",
            "agent_name": "diagnosis",
            "status": "accepted",
            "notes": "Good diagnosis",
            "timestamp": datetime.now().isoformat(),
        },
        {
            "event_id": "req-002",
            "agent_name": "diagnosis",
            "status": "rejected",
            "notes": "Missed key finding",
            "suggested_improvement": "Add guidance for red flags",
            "timestamp": datetime.now().isoformat(),
        },
        {
            "event_id": "req-003",
            "agent_name": "plan",
            "status": "needs_review",
            "tags": ["unclear", "incomplete"],
            "timestamp": datetime.now().isoformat(),
        },
    ]

    feedback_file = temp_dirs["logs"] / "feedback.jsonl"
    with open(feedback_file, "w") as f:
        for entry in feedback:
            f.write(json.dumps(entry) + "\n")

    return feedback


class TestPromptVersion:
    """Tests for PromptVersion model."""

    def test_create_version(self):
        """Test creating a prompt version."""
        version = PromptVersion(
            version="v1.0",
            agent_name="diagnosis",
            prompt_text="You are a diagnosis agent...",
        )

        assert version.version == "v1.0"
        assert version.agent_name == "diagnosis"
        assert not version.is_active

    def test_version_with_metadata(self):
        """Test version with optimization metadata."""
        version = PromptVersion(
            version="v2.0",
            agent_name="diagnosis",
            prompt_text="Improved prompt...",
            parent_version="v1.0",
            optimization_reason="Low confidence scores",
            changes_made=["Added specificity", "Improved structure"],
        )

        assert version.parent_version == "v1.0"
        assert len(version.changes_made) == 2


class TestPromptOptimizer:
    """Tests for PromptOptimizer."""

    def test_init(self, temp_dirs):
        """Test optimizer initialization."""
        optimizer = PromptOptimizer(
            prompts_dir=temp_dirs["prompts"],
            logs_dir=temp_dirs["logs"],
        )

        assert optimizer.prompts_dir.exists()

    def test_register_prompt(self, temp_dirs):
        """Test registering a new prompt."""
        optimizer = PromptOptimizer(
            prompts_dir=temp_dirs["prompts"],
            logs_dir=temp_dirs["logs"],
        )

        version = optimizer.register_prompt(
            agent_name="diagnosis",
            prompt_text="You are a diagnosis agent.",
            version="v1.0",
            is_active=True,
        )

        assert version.version == "v1.0"
        assert version.is_active

        # Should be retrievable
        current = optimizer.get_current_prompt("diagnosis")
        assert current is not None
        assert current.version == "v1.0"

    def test_version_history(self, temp_dirs):
        """Test getting version history."""
        optimizer = PromptOptimizer(
            prompts_dir=temp_dirs["prompts"],
            logs_dir=temp_dirs["logs"],
        )

        optimizer.register_prompt("diagnosis", "Prompt v1", "v1.0")
        optimizer.register_prompt("diagnosis", "Prompt v2", "v2.0")

        history = optimizer.get_version_history("diagnosis")
        assert len(history) == 2

    def test_analyze_feedback(self, temp_dirs, sample_feedback_data):
        """Test feedback analysis."""
        optimizer = PromptOptimizer(
            prompts_dir=temp_dirs["prompts"],
            logs_dir=temp_dirs["logs"],
        )

        summaries = optimizer.analyze_feedback()

        assert "diagnosis" in summaries
        assert summaries["diagnosis"].total_annotations == 2
        assert summaries["diagnosis"].accepted == 1
        assert summaries["diagnosis"].rejected == 1
        assert len(summaries["diagnosis"].improvement_suggestions) == 1

    def test_analyze_feedback_by_agent(self, temp_dirs, sample_feedback_data):
        """Test filtering feedback by agent."""
        optimizer = PromptOptimizer(
            prompts_dir=temp_dirs["prompts"],
            logs_dir=temp_dirs["logs"],
        )

        summaries = optimizer.analyze_feedback(agent_name="plan")

        assert "plan" in summaries
        assert "diagnosis" not in summaries


class TestAgentEvaluator:
    """Tests for AgentEvaluator."""

    def test_init(self, tmp_path):
        """Test evaluator initialization."""
        evaluator = AgentEvaluator(
            test_suites_dir=tmp_path / "suites",
            results_dir=tmp_path / "results",
        )

        assert evaluator.test_suites_dir.exists()
        assert evaluator.results_dir.exists()

    def test_create_test_suite(self, tmp_path):
        """Test creating a test suite."""
        evaluator = AgentEvaluator(
            test_suites_dir=tmp_path / "suites",
            results_dir=tmp_path / "results",
        )

        cases = [
            TestCase(
                id="test-001",
                agent_name="diagnosis",
                input_data={"patient": {"age": 55}},
                required_fields=["primary_diagnosis"],
            ),
            TestCase(
                id="test-002",
                agent_name="diagnosis",
                input_data={"patient": {"age": 30}},
            ),
        ]

        evaluator.create_test_suite("clinical_basic", cases)

        loaded = evaluator.load_test_suite("clinical_basic")
        assert len(loaded) == 2

    def test_score_completeness(self, tmp_path):
        """Test completeness scoring."""
        evaluator = AgentEvaluator(test_suites_dir=tmp_path)

        output = {"primary_diagnosis": "LBP", "confidence": 0.9}
        required = ["primary_diagnosis", "confidence", "missing_field"]

        score = evaluator._score_completeness(output, required)
        assert abs(score - 0.67) < 0.1  # 2/3 fields present

    def test_score_format(self, tmp_path):
        """Test format scoring with forbidden patterns."""
        evaluator = AgentEvaluator(test_suites_dir=tmp_path)

        output = {"text": "This is good output"}
        forbidden = ["error", "fail", "unknown"]

        score = evaluator._score_format(output, forbidden)
        assert score == 1.0  # No forbidden patterns

        output_bad = {"text": "Error occurred"}
        score_bad = evaluator._score_format(output_bad, forbidden)
        assert score_bad < 1.0

    def test_score_accuracy(self, tmp_path):
        """Test accuracy scoring."""
        evaluator = AgentEvaluator(test_suites_dir=tmp_path)

        output = {"diagnosis": "LBP", "severity": "moderate"}
        expected = {"diagnosis": "LBP", "severity": "moderate"}

        score = evaluator._score_accuracy(output, expected)
        assert score == 1.0

        output_partial = {"diagnosis": "LBP", "severity": "mild"}
        score_partial = evaluator._score_accuracy(output_partial, expected)
        assert score_partial < 1.0


class TestEvaluationRun:
    """Tests for EvaluationRun model."""

    def test_compute_metrics(self):
        """Test aggregating metrics from results."""
        run = EvaluationRun(
            run_id="run-001",
            agent_name="diagnosis",
            prompt_version="v1.0",
            test_suite="basic",
            results=[
                EvaluationResult(
                    test_case_id="t1",
                    agent_name="diagnosis",
                    prompt_version="v1.0",
                    success=True,
                    completeness_score=0.8,
                    format_score=1.0,
                    duration_ms=1000,
                ),
                EvaluationResult(
                    test_case_id="t2",
                    agent_name="diagnosis",
                    prompt_version="v1.0",
                    success=True,
                    completeness_score=0.9,
                    format_score=0.9,
                    duration_ms=1500,
                ),
                EvaluationResult(
                    test_case_id="t3",
                    agent_name="diagnosis",
                    prompt_version="v1.0",
                    success=False,
                    completeness_score=0.0,
                    format_score=0.0,
                    duration_ms=500,
                ),
            ],
        )

        run.compute_metrics()

        assert abs(run.success_rate - 0.67) < 0.1
        assert abs(run.avg_completeness - 0.57) < 0.1
        assert run.avg_duration_ms == 1000


class TestLearningScheduler:
    """Tests for LearningScheduler."""

    def test_init(self, tmp_path):
        """Test scheduler initialization."""
        scheduler = LearningScheduler(
            config_dir=tmp_path / "scheduler",
            results_dir=tmp_path / "results",
        )

        assert scheduler.config_dir.exists()

    def test_add_task(self, tmp_path):
        """Test adding a scheduled task."""
        scheduler = LearningScheduler(config_dir=tmp_path / "scheduler")

        task = scheduler.add_task(
            task_id="test_eval",
            task_type=TaskType.EVALUATION,
            schedule="daily@02:00",
            config={"agents": ["diagnosis"]},
        )

        assert task.task_id == "test_eval"
        assert task.task_type == TaskType.EVALUATION
        assert task.next_run is not None

    def test_calculate_next_run_daily(self, tmp_path):
        """Test calculating next run for daily schedule."""
        scheduler = LearningScheduler(config_dir=tmp_path / "scheduler")

        now = datetime.now()
        next_run = scheduler._calculate_next_run("daily@02:00", now)

        assert next_run.hour == 2
        assert next_run.minute == 0

    def test_calculate_next_run_hourly(self, tmp_path):
        """Test calculating next run for hourly schedule."""
        scheduler = LearningScheduler(config_dir=tmp_path / "scheduler")

        now = datetime.now()
        next_run = scheduler._calculate_next_run("hourly", now)

        assert next_run > now
        assert (next_run - now).total_seconds() <= 3600

    def test_calculate_next_run_interval(self, tmp_path):
        """Test calculating next run for interval schedule."""
        scheduler = LearningScheduler(config_dir=tmp_path / "scheduler")

        now = datetime.now()
        next_run = scheduler._calculate_next_run("every:30:minutes", now)

        expected = now + timedelta(minutes=30)
        assert abs((next_run - expected).total_seconds()) < 1

    def test_list_tasks(self, tmp_path):
        """Test listing tasks."""
        scheduler = LearningScheduler(config_dir=tmp_path / "scheduler")

        scheduler.add_task("task1", TaskType.EVALUATION, "daily@01:00")
        scheduler.add_task("task2", TaskType.REPORT, "weekly@sun:02:00")

        tasks = scheduler.list_tasks()
        assert len(tasks) == 2

    def test_remove_task(self, tmp_path):
        """Test removing a task."""
        scheduler = LearningScheduler(config_dir=tmp_path / "scheduler")

        scheduler.add_task("task_to_remove", TaskType.CLEANUP, "daily@05:00")
        assert len(scheduler.list_tasks()) == 1

        result = scheduler.remove_task("task_to_remove")
        assert result is True
        assert len(scheduler.list_tasks()) == 0

    def test_setup_default_schedule(self, tmp_path):
        """Test setting up default schedule."""
        scheduler = LearningScheduler(config_dir=tmp_path / "scheduler")

        scheduler.setup_default_schedule()

        tasks = scheduler.list_tasks()
        task_ids = [t.task_id for t in tasks]

        assert "nightly_evaluation" in task_ids
        assert "daily_optimization" in task_ids
        assert "weekly_report" in task_ids

    @pytest.mark.asyncio
    async def test_run_report_task(self, tmp_path):
        """Test running a report task."""
        # Create logs directory with minimal data
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        scheduler = LearningScheduler(
            config_dir=tmp_path / "scheduler",
            results_dir=tmp_path / "results",
        )

        result = await scheduler._run_report_task({"days": 7})

        assert "report_path" in result
        assert "summary" in result


class TestScheduledTask:
    """Tests for ScheduledTask model."""

    def test_create_task(self):
        """Test creating a scheduled task."""
        task = ScheduledTask(
            task_id="test",
            task_type=TaskType.EVALUATION,
            schedule="daily@00:00",
            config={"test": True},
        )

        assert task.task_id == "test"
        assert task.enabled is True
        assert task.run_count == 0

    def test_task_status(self):
        """Test task status tracking."""
        task = ScheduledTask(
            task_id="test",
            task_type=TaskType.REPORT,
            schedule="hourly",
        )

        assert task.last_status == TaskStatus.PENDING

        task.last_status = TaskStatus.COMPLETED
        task.run_count = 1

        assert task.last_status == TaskStatus.COMPLETED
        assert task.run_count == 1
