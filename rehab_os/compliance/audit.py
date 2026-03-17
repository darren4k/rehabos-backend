"""HIPAA Audit Trail Service.

Append-only JSONL audit log for PHI access tracking.
Retention: 7 years per HIPAA §164.530(j).
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AuditEvent:
    """Single audit trail entry."""

    __slots__ = (
        "event_id",
        "timestamp",
        "user_id",
        "patient_id",
        "action",
        "resource_type",
        "resource_id",
        "ip_address",
        "session_id",
        "detail",
        "retention_until",
    )

    VALID_ACTIONS = ("view", "create", "modify", "delete", "export", "login", "logout")

    def __init__(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str = "",
        patient_id: str = "",
        ip_address: str = "",
        session_id: str = "",
        detail: str = "",
        retention_years: int = 7,
    ):
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Must be one of {self.VALID_ACTIONS}")
        now = datetime.now(timezone.utc)
        self.event_id = str(uuid.uuid4())
        self.timestamp = now.isoformat()
        self.user_id = user_id
        self.patient_id = patient_id
        self.action = action
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.ip_address = ip_address
        self.session_id = session_id
        self.detail = detail
        self.retention_until = now.replace(year=now.year + retention_years).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}


class AuditService:
    """Thread-safe append-only HIPAA audit trail.

    Writes JSONL to a file that must not be modified or deleted
    until the retention period expires.
    """

    def __init__(self, log_path: str = "./data/audit/hipaa_audit.jsonl", retention_years: int = 7):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._retention_years = retention_years
        self._lock = threading.Lock()

    def log(self, event: AuditEvent) -> str:
        """Append an audit event. Returns the event_id."""
        line = json.dumps(event.to_dict(), separators=(",", ":"))
        with self._lock:
            with open(self._log_path, "a") as f:
                f.write(line + "\n")
        logger.debug("Audit event %s: %s %s %s", event.event_id, event.action, event.resource_type, event.resource_id)
        return event.event_id

    def log_quick(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str = "",
        patient_id: str = "",
        ip_address: str = "",
        session_id: str = "",
        detail: str = "",
    ) -> str:
        """Convenience wrapper that creates an AuditEvent and logs it."""
        event = AuditEvent(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            patient_id=patient_id,
            ip_address=ip_address,
            session_id=session_id,
            detail=detail,
            retention_years=self._retention_years,
        )
        return self.log(event)

    def query(
        self,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Query audit events with optional filters. Returns newest first."""
        if not self._log_path.exists():
            return []

        results: list[dict[str, Any]] = []
        with open(self._log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if user_id and entry.get("user_id") != user_id:
                    continue
                if patient_id and entry.get("patient_id") != patient_id:
                    continue
                if action and entry.get("action") != action:
                    continue
                if resource_type and entry.get("resource_type") != resource_type:
                    continue
                if start_date and entry.get("timestamp", "") < start_date:
                    continue
                if end_date and entry.get("timestamp", "") > end_date:
                    continue

                results.append(entry)

        # Newest first, limited
        results.reverse()
        return results[:limit]


# Module-level singleton
_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Get or create the global audit service."""
    global _audit_service
    if _audit_service is None:
        from rehab_os.config import get_settings
        settings = get_settings()
        _audit_service = AuditService(
            log_path=settings.audit_log_path,
            retention_years=settings.audit_retention_years,
        )
    return _audit_service
