"""Goal metric calculator — uses AGGREGATE queries only.

Every metric returns a single number. No PHI is exposed or stored.
Production implementations should use SQL COUNT/SUM/AVG with provider_id filters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric definition
# ---------------------------------------------------------------------------

@dataclass
class MetricDefinition:
    """Describes a calculable aggregate metric."""

    name: str
    description: str
    unit: str  # "count", "percent", "hours", "dollars", "score"

    async def calculate(self, user_id: str, period: str) -> float:
        """Calculate the metric value. Override in subclasses for real queries."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete metrics (mock — production: real SQL queries)
# ---------------------------------------------------------------------------

class UnsignedNotesOver24h(MetricDefinition):
    """COUNT of notes with status=draft older than 24h for this provider."""

    def __init__(self) -> None:
        super().__init__(
            name="unsigned_notes_over_24h",
            description="Notes unsigned for more than 24 hours",
            unit="count",
        )

    async def calculate(self, user_id: str, period: str) -> float:
        # Production: SELECT COUNT(*) FROM clinical_notes
        #   WHERE provider_id = :uid AND status = 'draft'
        #   AND created_at < NOW() - INTERVAL '24 hours'
        return 3.0


class VisitsCompleted(MetricDefinition):
    """COUNT of completed encounters this period."""

    def __init__(self) -> None:
        super().__init__(
            name="visits_completed",
            description="Completed visits in period",
            unit="count",
        )

    async def calculate(self, user_id: str, period: str) -> float:
        # Production: SELECT COUNT(*) FROM encounters
        #   WHERE provider_id = :uid AND status = 'completed'
        #   AND date >= <period_start>
        return 12.0


class DocCompliancePct(MetricDefinition):
    """Signed / total * 100 for the period."""

    def __init__(self) -> None:
        super().__init__(
            name="doc_compliance_pct",
            description="Percentage of notes signed within compliance window",
            unit="percent",
        )

    async def calculate(self, user_id: str, period: str) -> float:
        # Production: SELECT
        #   COUNT(*) FILTER (WHERE status='signed') * 100.0 / NULLIF(COUNT(*), 0)
        #   FROM clinical_notes WHERE provider_id = :uid AND date >= <period_start>
        return 88.0


class RevenueCollected(MetricDefinition):
    """SUM from billing for the period."""

    def __init__(self) -> None:
        super().__init__(
            name="revenue_collected",
            description="Total revenue collected in period",
            unit="dollars",
        )

    async def calculate(self, user_id: str, period: str) -> float:
        # Production: SELECT COALESCE(SUM(amount_collected), 0)
        #   FROM billing WHERE provider_id = :uid AND date >= <period_start>
        return 4200.0


class PatientSatisfaction(MetricDefinition):
    """AVG satisfaction score for the period."""

    def __init__(self) -> None:
        super().__init__(
            name="patient_satisfaction",
            description="Average patient satisfaction score (1-5)",
            unit="score",
        )

    async def calculate(self, user_id: str, period: str) -> float:
        # Production: SELECT AVG(score) FROM feedback
        #   WHERE provider_id = :uid AND date >= <period_start>
        return 4.6


class NoteTurnaroundHours(MetricDefinition):
    """AVG time from encounter to signed note, in hours."""

    def __init__(self) -> None:
        super().__init__(
            name="note_turnaround_hours",
            description="Average hours from encounter to note signing",
            unit="hours",
        )

    async def calculate(self, user_id: str, period: str) -> float:
        # Production: SELECT AVG(EXTRACT(EPOCH FROM (signed_at - encounter_date))/3600)
        #   FROM clinical_notes WHERE provider_id = :uid AND signed_at IS NOT NULL
        #   AND encounter_date >= <period_start>
        return 6.5


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: dict[str, MetricDefinition] = {}


def _register_defaults() -> None:
    """Register all built-in metrics."""
    for cls in (
        UnsignedNotesOver24h,
        VisitsCompleted,
        DocCompliancePct,
        RevenueCollected,
        PatientSatisfaction,
        NoteTurnaroundHours,
    ):
        inst = cls()
        _METRIC_REGISTRY[inst.name] = inst


_register_defaults()


def get_metric(name: str) -> Optional[MetricDefinition]:
    """Look up a metric by name."""
    return _METRIC_REGISTRY.get(name)


def list_metrics() -> list[dict]:
    """Return metadata for all registered metrics."""
    return [
        {"name": m.name, "description": m.description, "unit": m.unit}
        for m in _METRIC_REGISTRY.values()
    ]


async def calculate_metric(
    metric_name: str, user_id: str, period: str = "weekly"
) -> Optional[float]:
    """Calculate a single metric value. Returns None if metric not found."""
    metric = _METRIC_REGISTRY.get(metric_name)
    if metric is None:
        logger.warning("Unknown metric: %s", metric_name)
        return None
    return await metric.calculate(user_id, period)


# ---------------------------------------------------------------------------
# Goal progress updater
# ---------------------------------------------------------------------------

async def update_goal_progress(user_id: str, goals: list) -> list:
    """Refresh current values on a list of AgentGoal objects.

    Returns the same list with ``current`` fields updated from live metrics.
    """
    for goal in goals:
        value = await calculate_metric(goal.metric, user_id, goal.period)
        if value is not None:
            goal.current = value
    return goals
