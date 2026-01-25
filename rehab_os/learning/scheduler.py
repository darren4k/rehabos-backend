"""Learning loop scheduler for automated optimization."""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of scheduled tasks."""

    EVALUATION = "evaluation"  # Run agent evaluations
    OPTIMIZATION = "optimization"  # Generate optimization suggestions
    AB_TEST = "ab_test"  # Run A/B tests
    REPORT = "report"  # Generate effectiveness reports
    CLEANUP = "cleanup"  # Clean up old logs/results


class TaskStatus(str, Enum):
    """Status of a scheduled task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ScheduledTask(BaseModel):
    """A scheduled task in the learning loop."""

    task_id: str
    task_type: TaskType
    schedule: str  # cron-like or interval description
    config: dict[str, Any] = Field(default_factory=dict)

    # Execution tracking
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0

    # Results
    last_status: TaskStatus = TaskStatus.PENDING
    last_error: Optional[str] = None
    last_result: Optional[dict[str, Any]] = None

    enabled: bool = True


class TaskResult(BaseModel):
    """Result of executing a scheduled task."""

    task_id: str
    task_type: TaskType
    started_at: datetime
    completed_at: datetime
    status: TaskStatus
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class LearningScheduler:
    """Scheduler for the automated learning loop.

    Manages scheduled tasks for:
    - Nightly evaluation runs
    - Periodic optimization analysis
    - A/B test execution
    - Report generation
    - Log cleanup

    Can run as a background service or be triggered manually.
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ):
        """Initialize scheduler.

        Args:
            config_dir: Directory for scheduler configuration
            results_dir: Directory for task results
        """
        self.config_dir = config_dir or Path("data/scheduler")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = results_dir or Path("data/scheduler_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._task_handlers: dict[TaskType, Callable] = {}

        # Load saved tasks
        self._load_tasks()

        # Register default handlers
        self._register_default_handlers()

    def _load_tasks(self) -> None:
        """Load scheduled tasks from disk."""
        tasks_file = self.config_dir / "tasks.json"
        if not tasks_file.exists():
            return

        try:
            with open(tasks_file) as f:
                data = json.load(f)
                for task_data in data.get("tasks", []):
                    task = ScheduledTask(**task_data)
                    self._tasks[task.task_id] = task
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to load tasks: {e}")

    def _save_tasks(self) -> None:
        """Save scheduled tasks to disk."""
        tasks_file = self.config_dir / "tasks.json"
        data = {
            "tasks": [t.model_dump() for t in self._tasks.values()],
            "saved_at": datetime.now().isoformat(),
        }
        with open(tasks_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _register_default_handlers(self) -> None:
        """Register default task handlers."""
        self._task_handlers[TaskType.EVALUATION] = self._run_evaluation_task
        self._task_handlers[TaskType.OPTIMIZATION] = self._run_optimization_task
        self._task_handlers[TaskType.AB_TEST] = self._run_ab_test_task
        self._task_handlers[TaskType.REPORT] = self._run_report_task
        self._task_handlers[TaskType.CLEANUP] = self._run_cleanup_task

    def register_handler(
        self,
        task_type: TaskType,
        handler: Callable,
    ) -> None:
        """Register a custom task handler."""
        self._task_handlers[task_type] = handler

    def add_task(
        self,
        task_id: str,
        task_type: TaskType,
        schedule: str,
        config: Optional[dict[str, Any]] = None,
    ) -> ScheduledTask:
        """Add a new scheduled task.

        Args:
            task_id: Unique identifier for the task
            task_type: Type of task
            schedule: Schedule string (e.g., "daily@02:00", "hourly", "weekly@sun:03:00")
            config: Task-specific configuration

        Returns:
            The created ScheduledTask
        """
        task = ScheduledTask(
            task_id=task_id,
            task_type=task_type,
            schedule=schedule,
            config=config or {},
            next_run=self._calculate_next_run(schedule),
        )

        self._tasks[task_id] = task
        self._save_tasks()

        logger.info(f"Added scheduled task: {task_id} ({task_type})")
        return task

    def remove_task(self, task_id: str) -> bool:
        """Remove a scheduled task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            self._save_tasks()
            return True
        return False

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[ScheduledTask]:
        """List all scheduled tasks."""
        return list(self._tasks.values())

    def _calculate_next_run(
        self,
        schedule: str,
        from_time: Optional[datetime] = None,
    ) -> datetime:
        """Calculate next run time from schedule string.

        Supported formats:
        - "daily@HH:MM" - Run daily at specified time
        - "hourly" - Run every hour
        - "weekly@DAY:HH:MM" - Run weekly on specified day
        - "every:N:minutes" - Run every N minutes
        """
        now = from_time or datetime.now()

        if schedule.startswith("daily@"):
            time_str = schedule.split("@")[1]
            hour, minute = map(int, time_str.split(":"))
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run

        elif schedule == "hourly":
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return next_run

        elif schedule.startswith("weekly@"):
            parts = schedule.split("@")[1].split(":")
            day_name = parts[0].lower()
            hour = int(parts[1])
            minute = int(parts[2]) if len(parts) > 2 else 0

            days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
            target_day = days.index(day_name[:3])
            current_day = now.weekday()

            days_ahead = target_day - current_day
            if days_ahead <= 0:
                days_ahead += 7

            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return next_run

        elif schedule.startswith("every:"):
            parts = schedule.split(":")
            interval = int(parts[1])
            unit = parts[2] if len(parts) > 2 else "minutes"

            if unit == "minutes":
                return now + timedelta(minutes=interval)
            elif unit == "hours":
                return now + timedelta(hours=interval)
            elif unit == "days":
                return now + timedelta(days=interval)

        # Default: run in 1 hour
        return now + timedelta(hours=1)

    async def run_task(self, task_id: str) -> TaskResult:
        """Manually run a specific task."""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        return await self._execute_task(task)

    async def _execute_task(self, task: ScheduledTask) -> TaskResult:
        """Execute a scheduled task."""
        started_at = datetime.now()
        task.last_status = TaskStatus.RUNNING

        result = TaskResult(
            task_id=task.task_id,
            task_type=task.task_type,
            started_at=started_at,
            completed_at=started_at,  # Updated after completion
            status=TaskStatus.RUNNING,
        )

        try:
            handler = self._task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type: {task.task_type}")

            task_result = await handler(task.config)

            result.status = TaskStatus.COMPLETED
            result.result = task_result
            task.last_status = TaskStatus.COMPLETED
            task.last_result = task_result

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            result.status = TaskStatus.FAILED
            result.error = str(e)[:500]
            task.last_status = TaskStatus.FAILED
            task.last_error = str(e)[:500]

        finally:
            result.completed_at = datetime.now()
            task.last_run = started_at
            task.next_run = self._calculate_next_run(task.schedule)
            task.run_count += 1
            self._save_tasks()
            self._save_result(result)

        return result

    def _save_result(self, result: TaskResult) -> None:
        """Save task result to disk."""
        result_file = self.results_dir / f"{result.task_id}_{result.started_at.strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, "w") as f:
            f.write(result.model_dump_json(indent=2))

    # Default task handlers

    async def _run_evaluation_task(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run agent evaluation task."""
        from rehab_os.learning.evaluator import AgentEvaluator

        evaluator = AgentEvaluator()

        agents = config.get("agents", [])
        test_suite = config.get("test_suite", "default")
        results = {}

        for agent_name in agents:
            try:
                # This would need proper agent instantiation
                # For now, just log what would be evaluated
                logger.info(f"Would evaluate {agent_name} with suite {test_suite}")
                results[agent_name] = {"status": "skipped", "reason": "Agent instantiation not implemented"}
            except Exception as e:
                results[agent_name] = {"status": "error", "error": str(e)}

        return {"evaluated_agents": list(results.keys()), "results": results}

    async def _run_optimization_task(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run prompt optimization analysis."""
        from rehab_os.learning.prompt_optimizer import PromptOptimizer

        optimizer = PromptOptimizer()

        agents = config.get("agents")  # None means all
        strategy = config.get("strategy", "metric_based")

        candidates = optimizer.get_optimization_candidates()

        if agents:
            candidates = [c for c in candidates if c in agents]

        optimizations = []
        for agent_name in candidates:
            opt = optimizer.generate_optimization(agent_name, strategy)
            if opt:
                optimizations.append(opt.model_dump())

        return {
            "candidates_found": len(candidates),
            "optimizations_generated": len(optimizations),
            "optimizations": optimizations,
        }

    async def _run_ab_test_task(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run A/B test task."""
        # A/B tests typically need specific setup
        # This is a placeholder that logs the configuration
        agent_name = config.get("agent_name")
        version_a = config.get("version_a")
        version_b = config.get("version_b")

        logger.info(f"Would run A/B test for {agent_name}: {version_a} vs {version_b}")

        return {
            "status": "skipped",
            "reason": "A/B tests require manual configuration",
            "config": config,
        }

    async def _run_report_task(self, config: dict[str, Any]) -> dict[str, Any]:
        """Generate effectiveness report."""
        from rehab_os.observability import PromptAnalytics

        analytics = PromptAnalytics()
        days = config.get("days", 7)

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        report = analytics.generate_report(start_time, end_time)
        report_dict = report.to_dict()

        # Save report
        output_dir = Path("data/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"effectiveness_{end_time.strftime('%Y%m%d')}.json"

        with open(output_file, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        return {
            "report_path": str(output_file),
            "summary": report_dict.get("summary", {}),
            "attention_needed": report_dict.get("attention_needed", {}),
        }

    async def _run_cleanup_task(self, config: dict[str, Any]) -> dict[str, Any]:
        """Clean up old logs and results."""
        max_age_days = config.get("max_age_days", 30)
        cutoff = datetime.now() - timedelta(days=max_age_days)

        cleaned = {"logs": 0, "results": 0}

        # Clean old log files
        logs_dir = Path("data/logs")
        if logs_dir.exists():
            for log_file in logs_dir.glob("*.jsonl"):
                # Check file modification time
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if mtime < cutoff:
                    # Archive instead of delete
                    archive_dir = logs_dir / "archive"
                    archive_dir.mkdir(exist_ok=True)
                    log_file.rename(archive_dir / log_file.name)
                    cleaned["logs"] += 1

        # Clean old evaluation results
        results_dir = Path("data/evaluation_results")
        if results_dir.exists():
            for result_file in results_dir.glob("*.json"):
                mtime = datetime.fromtimestamp(result_file.stat().st_mtime)
                if mtime < cutoff:
                    archive_dir = results_dir / "archive"
                    archive_dir.mkdir(exist_ok=True)
                    result_file.rename(archive_dir / result_file.name)
                    cleaned["results"] += 1

        return cleaned

    async def run_loop(
        self,
        check_interval: int = 60,
        max_iterations: Optional[int] = None,
    ) -> None:
        """Run the scheduler loop.

        Args:
            check_interval: Seconds between checking for due tasks
            max_iterations: Maximum iterations (None for infinite)
        """
        self._running = True
        iteration = 0

        logger.info("Starting learning loop scheduler")

        while self._running:
            if max_iterations and iteration >= max_iterations:
                break

            # Check for due tasks
            now = datetime.now()
            for task in self._tasks.values():
                if not task.enabled:
                    continue

                if task.next_run and task.next_run <= now:
                    logger.info(f"Running due task: {task.task_id}")
                    await self._execute_task(task)

            iteration += 1
            await asyncio.sleep(check_interval)

        logger.info("Learning loop scheduler stopped")

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False

    def setup_default_schedule(self) -> None:
        """Set up default scheduled tasks for the learning loop."""
        # Nightly evaluation at 2 AM
        if "nightly_evaluation" not in self._tasks:
            self.add_task(
                task_id="nightly_evaluation",
                task_type=TaskType.EVALUATION,
                schedule="daily@02:00",
                config={
                    "agents": ["red_flag", "diagnosis", "plan", "qa"],
                    "test_suite": "clinical_scenarios",
                },
            )

        # Daily optimization analysis at 3 AM
        if "daily_optimization" not in self._tasks:
            self.add_task(
                task_id="daily_optimization",
                task_type=TaskType.OPTIMIZATION,
                schedule="daily@03:00",
                config={
                    "strategy": "metric_based",
                },
            )

        # Weekly effectiveness report on Sundays at 4 AM
        if "weekly_report" not in self._tasks:
            self.add_task(
                task_id="weekly_report",
                task_type=TaskType.REPORT,
                schedule="weekly@sun:04:00",
                config={
                    "days": 7,
                },
            )

        # Monthly cleanup on 1st at 5 AM
        if "monthly_cleanup" not in self._tasks:
            self.add_task(
                task_id="monthly_cleanup",
                task_type=TaskType.CLEANUP,
                schedule="daily@05:00",  # Actually monthly, but simplified
                config={
                    "max_age_days": 90,
                },
            )

        logger.info("Default schedule configured")


def create_learning_scheduler() -> LearningScheduler:
    """Create and configure the learning scheduler."""
    scheduler = LearningScheduler()
    scheduler.setup_default_schedule()
    return scheduler
