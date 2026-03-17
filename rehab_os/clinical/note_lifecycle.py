"""Note lifecycle management for RehabOS.

Handles note signing, locking, co-signing, amendments, and version history.
Enforces Medicare compliance: DRAFT -> SIGNED -> LOCKED (no skipping).
Amendments preserve originals (signed notes are never overwritten).
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Medicare requires notes signed within 30 days of service
MEDICARE_SIGNING_DEADLINE_DAYS = 30


class NoteStatus(str, Enum):
    DRAFT = "draft"
    PENDING_COSIGN = "pending_cosign"
    SIGNED = "signed"
    LOCKED = "locked"
    AMENDED = "amended"


# Valid state transitions — no skipping allowed
_VALID_TRANSITIONS: dict[NoteStatus, set[NoteStatus]] = {
    NoteStatus.DRAFT: {NoteStatus.SIGNED, NoteStatus.PENDING_COSIGN},
    NoteStatus.PENDING_COSIGN: {NoteStatus.SIGNED},
    NoteStatus.SIGNED: {NoteStatus.LOCKED, NoteStatus.AMENDED},
    NoteStatus.LOCKED: {NoteStatus.AMENDED},
    NoteStatus.AMENDED: set(),  # Terminal — create new amendment instead
}


@dataclass
class NoteVersion:
    """Snapshot of note content at a point in time."""
    version_id: str
    note_id: str
    content: dict  # Full SOAP content at this version
    author_id: str
    created_at: str
    change_reason: str | None = None


@dataclass
class NoteAmendment:
    """An amendment to a signed/locked note. Preserves original text."""
    amendment_id: str
    note_id: str
    section: str  # "subjective", "objective", "assessment", "plan"
    original_text: str
    amended_text: str
    reason: str
    author_id: str
    created_at: str


@dataclass
class NoteLifecycleRecord:
    """In-memory record tracking lifecycle state for a note."""
    note_id: str
    status: NoteStatus = NoteStatus.DRAFT
    signer_id: str | None = None
    signed_at: str | None = None
    cosigner_id: str | None = None
    cosigner_requested_id: str | None = None
    cosigned_at: str | None = None
    locked_at: str | None = None
    service_date: str | None = None
    versions: list[NoteVersion] = field(default_factory=list)
    amendments: list[NoteAmendment] = field(default_factory=list)


class NoteLifecycleError(Exception):
    """Raised when a lifecycle operation violates rules."""
    pass


class NoteLifecycleManager:
    """Manages note signing, locking, co-signing, amendments, and versions.

    Storage is in-memory (dict-backed). In production this would be
    backed by PostgreSQL via the ClinicalNote model's status column
    plus dedicated version/amendment tables.
    """

    def __init__(self) -> None:
        self._records: dict[str, NoteLifecycleRecord] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_record(self, note_id: str) -> NoteLifecycleRecord:
        rec = self._records.get(note_id)
        if rec is None:
            raise NoteLifecycleError(f"Note {note_id} not tracked. Call register_note first.")
        return rec

    def _assert_transition(self, current: NoteStatus, target: NoteStatus) -> None:
        allowed = _VALID_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise NoteLifecycleError(
                f"Invalid transition: {current.value} -> {target.value}. "
                f"Allowed: {', '.join(s.value for s in allowed) or 'none'}"
            )

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _snapshot_content(self, note_content: dict, note_id: str, author_id: str,
                          reason: str | None = None) -> NoteVersion:
        """Create a version snapshot of the current note content."""
        version = NoteVersion(
            version_id=str(uuid.uuid4()),
            note_id=note_id,
            content=dict(note_content),
            author_id=author_id,
            created_at=self._now_iso(),
            change_reason=reason,
        )
        self._records[note_id].versions.append(version)
        return version

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_note(self, note_id: str, service_date: str | None = None,
                      initial_content: dict | None = None,
                      author_id: str | None = None) -> NoteLifecycleRecord:
        """Start tracking a note. Called when a ClinicalNote is created."""
        if note_id in self._records:
            return self._records[note_id]
        rec = NoteLifecycleRecord(note_id=note_id, service_date=service_date)
        self._records[note_id] = rec
        if initial_content and author_id:
            self._snapshot_content(initial_content, note_id, author_id, "initial draft")
        logger.info("Registered note %s for lifecycle tracking", note_id)
        return rec

    def get_status(self, note_id: str) -> dict[str, Any]:
        """Return current lifecycle state."""
        rec = self._get_record(note_id)
        return {
            "note_id": rec.note_id,
            "status": rec.status.value,
            "signer_id": rec.signer_id,
            "signed_at": rec.signed_at,
            "cosigner_id": rec.cosigner_id,
            "cosigned_at": rec.cosigned_at,
            "locked_at": rec.locked_at,
            "version_count": len(rec.versions),
            "amendment_count": len(rec.amendments),
        }

    def sign_note(self, note_id: str, signer_id: str,
                  note_content: dict | None = None) -> dict[str, Any]:
        """Sign a draft or pending-cosign note. Snapshots content."""
        rec = self._get_record(note_id)
        self._assert_transition(rec.status, NoteStatus.SIGNED)
        if note_content:
            self._snapshot_content(note_content, note_id, signer_id, "signed")
        rec.status = NoteStatus.SIGNED
        rec.signer_id = signer_id
        rec.signed_at = self._now_iso()
        logger.info("Note %s signed by %s", note_id, signer_id)
        return self.get_status(note_id)

    def lock_note(self, note_id: str) -> dict[str, Any]:
        """Lock a signed note — makes it read-only."""
        rec = self._get_record(note_id)
        self._assert_transition(rec.status, NoteStatus.LOCKED)
        rec.status = NoteStatus.LOCKED
        rec.locked_at = self._now_iso()
        logger.info("Note %s locked", note_id)
        return self.get_status(note_id)

    def request_cosign(self, note_id: str, cosigner_id: str) -> dict[str, Any]:
        """Request a co-signature (e.g., PTA note needing PT cosign)."""
        rec = self._get_record(note_id)
        self._assert_transition(rec.status, NoteStatus.PENDING_COSIGN)
        rec.status = NoteStatus.PENDING_COSIGN
        rec.cosigner_requested_id = cosigner_id
        logger.info("Note %s pending cosign from %s", note_id, cosigner_id)
        return self.get_status(note_id)

    def cosign_note(self, note_id: str, cosigner_id: str) -> dict[str, Any]:
        """Co-sign a note that is pending co-signature."""
        rec = self._get_record(note_id)
        if rec.status != NoteStatus.PENDING_COSIGN:
            raise NoteLifecycleError(
                f"Note {note_id} is not pending cosign (status: {rec.status.value})"
            )
        if rec.cosigner_requested_id and rec.cosigner_requested_id != cosigner_id:
            raise NoteLifecycleError(
                f"Cosign requested from {rec.cosigner_requested_id}, not {cosigner_id}"
            )
        rec.status = NoteStatus.SIGNED
        rec.cosigner_id = cosigner_id
        rec.cosigned_at = self._now_iso()
        logger.info("Note %s cosigned by %s", note_id, cosigner_id)
        return self.get_status(note_id)

    def amend_note(self, note_id: str, section: str, new_text: str,
                   reason: str, author_id: str) -> NoteAmendment:
        """Amend a signed or locked note. Never overwrites the original.

        Creates an amendment record and transitions status to AMENDED.
        The original content is preserved in version history.
        """
        rec = self._get_record(note_id)
        if rec.status not in (NoteStatus.SIGNED, NoteStatus.LOCKED):
            raise NoteLifecycleError(
                f"Only signed/locked notes can be amended (status: {rec.status.value})"
            )
        valid_sections = {"subjective", "objective", "assessment", "plan"}
        if section not in valid_sections:
            raise NoteLifecycleError(
                f"Invalid section '{section}'. Must be one of: {', '.join(sorted(valid_sections))}"
            )
        # Find original text from last version
        original_text = ""
        if rec.versions:
            last_content = rec.versions[-1].content
            original_text = last_content.get(section, "")

        amendment = NoteAmendment(
            amendment_id=str(uuid.uuid4()),
            note_id=note_id,
            section=section,
            original_text=original_text,
            amended_text=new_text,
            reason=reason,
            author_id=author_id,
            created_at=self._now_iso(),
        )
        rec.amendments.append(amendment)
        rec.status = NoteStatus.AMENDED
        logger.info("Note %s amended by %s (section: %s)", note_id, author_id, section)
        return amendment

    def get_versions(self, note_id: str) -> list[NoteVersion]:
        """Return all version snapshots for a note."""
        rec = self._get_record(note_id)
        return list(rec.versions)

    def get_amendments(self, note_id: str) -> list[NoteAmendment]:
        """Return all amendments for a note."""
        rec = self._get_record(note_id)
        return list(rec.amendments)

    # ------------------------------------------------------------------
    # Compliance
    # ------------------------------------------------------------------

    def check_signing_compliance(self, note_id: str) -> dict[str, Any]:
        """Check Medicare signing compliance for a note.

        Returns:
            signed: whether note has been signed
            cosigned: whether note has been cosigned (if applicable)
            days_since_service: days between service date and now
            overdue: True if unsigned past Medicare 30-day deadline
        """
        rec = self._get_record(note_id)
        now = datetime.now(timezone.utc)

        days_since_service: int | None = None
        overdue = False
        if rec.service_date:
            try:
                svc = datetime.fromisoformat(rec.service_date)
                if svc.tzinfo is None:
                    svc = svc.replace(tzinfo=timezone.utc)
                days_since_service = (now - svc).days
                if rec.status == NoteStatus.DRAFT and days_since_service > MEDICARE_SIGNING_DEADLINE_DAYS:
                    overdue = True
            except (ValueError, TypeError):
                pass

        return {
            "note_id": note_id,
            "signed": rec.status in (NoteStatus.SIGNED, NoteStatus.LOCKED, NoteStatus.AMENDED),
            "cosigned": rec.cosigner_id is not None,
            "cosign_pending": rec.status == NoteStatus.PENDING_COSIGN,
            "days_since_service": days_since_service,
            "overdue": overdue,
            "status": rec.status.value,
        }
