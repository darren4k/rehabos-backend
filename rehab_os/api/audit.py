"""HIPAA-compliant PHI access audit logger.

Writes structured audit entries to data/logs/phi_access.jsonl.
Use alongside the DB-backed AuditRepository for endpoints with DB sessions.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_AUDIT_DIR = Path("data/logs")
_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
_AUDIT_FILE = _AUDIT_DIR / "phi_access.jsonl"


def log_phi_access(
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str = "",
    ip_address: str = "",
    details: Optional[dict] = None,
) -> None:
    """Append a PHI access record to the audit log."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "ip_address": ip_address,
    }
    if details:
        entry["details"] = details

    try:
        with open(_AUDIT_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        logger.error("Failed to write PHI audit log entry")
