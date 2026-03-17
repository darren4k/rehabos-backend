"""Simplified HIPAA Breach Notification Tracking.

Tracks breach incidents and 60-day notification deadlines
per 45 CFR 164.404-414.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

NOTIFICATION_DEADLINE_DAYS = 60


class BreachSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BreachStatus(str, Enum):
    REPORTED = "reported"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    NOT_A_BREACH = "not_a_breach"
    NOTIFIED = "notified"
    CLOSED = "closed"


@dataclass
class BreachIncident:
    incident_id: str = field(default_factory=lambda: f"breach_{uuid.uuid4().hex[:12]}")
    discovered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reported_by: str = ""
    severity: str = BreachSeverity.LOW.value
    status: str = BreachStatus.REPORTED.value
    description: str = ""
    phi_types_involved: list[str] = field(default_factory=list)
    individuals_affected: int = 0
    notification_deadline: str = ""
    closed_at: str = ""
    resolution_notes: str = ""

    def __post_init__(self):
        if not self.notification_deadline:
            discovered = datetime.fromisoformat(self.discovered_at)
            deadline = discovered + timedelta(days=NOTIFICATION_DEADLINE_DAYS)
            self.notification_deadline = deadline.isoformat()


class BreachService:
    """Track breach incidents with JSONL persistence."""

    def __init__(self, data_path: str = "./data/compliance/breaches.jsonl"):
        self._path = Path(data_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def report_incident(
        self,
        reported_by: str,
        description: str,
        severity: str = BreachSeverity.LOW.value,
        phi_types: Optional[list[str]] = None,
        individuals_affected: int = 0,
    ) -> BreachIncident:
        """Create and persist a new breach incident."""
        incident = BreachIncident(
            reported_by=reported_by,
            severity=severity,
            description=description,
            phi_types_involved=phi_types or [],
            individuals_affected=individuals_affected,
        )
        self._append(incident)
        logger.warning(
            "Breach incident reported: %s severity=%s by=%s",
            incident.incident_id, severity, reported_by,
        )
        return incident

    def get_open_incidents(self) -> list[BreachIncident]:
        """Return all incidents that are not closed or dismissed."""
        closed = {BreachStatus.CLOSED.value, BreachStatus.NOT_A_BREACH.value}
        return [i for i in self._load_all() if i.status not in closed]

    def close_incident(self, incident_id: str, resolution_notes: str = "") -> Optional[BreachIncident]:
        """Close an incident by rewriting the log."""
        incidents = self._load_all()
        target = None
        for inc in incidents:
            if inc.incident_id == incident_id:
                inc.status = BreachStatus.CLOSED.value
                inc.closed_at = datetime.now(timezone.utc).isoformat()
                inc.resolution_notes = resolution_notes
                target = inc
                break
        if target:
            self._rewrite(incidents)
            logger.info("Breach incident closed: %s", incident_id)
        return target

    def _append(self, incident: BreachIncident) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(incident), separators=(",", ":")) + "\n")

    def _load_all(self) -> list[BreachIncident]:
        if not self._path.exists():
            return []
        results = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(BreachIncident(**json.loads(line)))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return results

    def _rewrite(self, incidents: list[BreachIncident]) -> None:
        with open(self._path, "w") as f:
            for inc in incidents:
                f.write(json.dumps(asdict(inc), separators=(",", ":")) + "\n")
