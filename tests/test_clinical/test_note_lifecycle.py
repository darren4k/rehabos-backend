"""Tests for rehab_os.clinical.note_lifecycle — note signing, locking, amendments."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from rehab_os.clinical.note_lifecycle import (
    MEDICARE_SIGNING_DEADLINE_DAYS,
    NoteAmendment,
    NoteLifecycleError,
    NoteLifecycleManager,
    NoteStatus,
    NoteVersion,
)


@pytest.fixture
def mgr() -> NoteLifecycleManager:
    return NoteLifecycleManager()


@pytest.fixture
def draft_note(mgr: NoteLifecycleManager) -> str:
    """Register a DRAFT note and return its ID."""
    note_id = "note-001"
    mgr.register_note(
        note_id,
        service_date=datetime.now(timezone.utc).isoformat(),
        initial_content={"subjective": "Pt reports pain 5/10", "objective": "ROM flex 95"},
        author_id="therapist-1",
    )
    return note_id


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

class TestSignNote:
    def test_sign_draft(self, mgr, draft_note):
        result = mgr.sign_note(draft_note, "therapist-1", {"subjective": "S", "objective": "O"})
        assert result["status"] == "signed"
        assert result["signer_id"] == "therapist-1"
        assert result["signed_at"] is not None

    def test_sign_already_signed_raises(self, mgr, draft_note):
        mgr.sign_note(draft_note, "therapist-1")
        with pytest.raises(NoteLifecycleError, match="Invalid transition"):
            mgr.sign_note(draft_note, "therapist-1")

    def test_sign_unregistered_raises(self, mgr):
        with pytest.raises(NoteLifecycleError, match="not tracked"):
            mgr.sign_note("nonexistent", "therapist-1")


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

class TestLockNote:
    def test_lock_signed(self, mgr, draft_note):
        mgr.sign_note(draft_note, "therapist-1")
        result = mgr.lock_note(draft_note)
        assert result["status"] == "locked"
        assert result["locked_at"] is not None

    def test_lock_unsigned_raises(self, mgr, draft_note):
        """Cannot skip DRAFT -> LOCKED (must go through SIGNED)."""
        with pytest.raises(NoteLifecycleError, match="Invalid transition"):
            mgr.lock_note(draft_note)


# ---------------------------------------------------------------------------
# Amendments
# ---------------------------------------------------------------------------

class TestAmendNote:
    def test_amend_locked_note(self, mgr, draft_note):
        mgr.sign_note(draft_note, "therapist-1", {"subjective": "original S"})
        mgr.lock_note(draft_note)
        amendment = mgr.amend_note(
            draft_note, "subjective", "corrected S", "Typo fix", "therapist-1"
        )
        assert isinstance(amendment, NoteAmendment)
        assert amendment.amended_text == "corrected S"
        assert amendment.reason == "Typo fix"
        # Original preserved
        assert amendment.original_text == "original S"
        # Status changed
        status = mgr.get_status(draft_note)
        assert status["status"] == "amended"

    def test_amend_signed_note(self, mgr, draft_note):
        mgr.sign_note(draft_note, "therapist-1", {"objective": "ROM 90"})
        amendment = mgr.amend_note(
            draft_note, "objective", "ROM 95", "Measurement correction", "therapist-1"
        )
        assert amendment.section == "objective"

    def test_amend_draft_raises(self, mgr, draft_note):
        with pytest.raises(NoteLifecycleError, match="Only signed/locked"):
            mgr.amend_note(draft_note, "subjective", "new", "reason", "therapist-1")

    def test_amend_invalid_section_raises(self, mgr, draft_note):
        mgr.sign_note(draft_note, "therapist-1")
        with pytest.raises(NoteLifecycleError, match="Invalid section"):
            mgr.amend_note(draft_note, "billing", "new", "reason", "therapist-1")


# ---------------------------------------------------------------------------
# Co-signing
# ---------------------------------------------------------------------------

class TestCosign:
    def test_request_cosign(self, mgr, draft_note):
        result = mgr.request_cosign(draft_note, "pt-supervisor")
        assert result["status"] == "pending_cosign"

    def test_cosign_note(self, mgr, draft_note):
        mgr.request_cosign(draft_note, "pt-supervisor")
        result = mgr.cosign_note(draft_note, "pt-supervisor")
        assert result["status"] == "signed"
        assert result["cosigner_id"] == "pt-supervisor"
        assert result["cosigned_at"] is not None

    def test_cosign_wrong_person_raises(self, mgr, draft_note):
        mgr.request_cosign(draft_note, "pt-supervisor")
        with pytest.raises(NoteLifecycleError, match="not"):
            mgr.cosign_note(draft_note, "wrong-person")

    def test_cosign_non_pending_raises(self, mgr, draft_note):
        with pytest.raises(NoteLifecycleError, match="not pending cosign"):
            mgr.cosign_note(draft_note, "pt-supervisor")


# ---------------------------------------------------------------------------
# Versions & amendments retrieval
# ---------------------------------------------------------------------------

class TestVersionsAndAmendments:
    def test_get_versions(self, mgr, draft_note):
        versions = mgr.get_versions(draft_note)
        assert len(versions) == 1  # initial draft snapshot
        assert versions[0].change_reason == "initial draft"

    def test_get_versions_after_sign(self, mgr, draft_note):
        mgr.sign_note(draft_note, "therapist-1", {"subjective": "S2"})
        versions = mgr.get_versions(draft_note)
        assert len(versions) == 2

    def test_get_amendments_empty(self, mgr, draft_note):
        amendments = mgr.get_amendments(draft_note)
        assert amendments == []

    def test_get_amendments_after_amend(self, mgr, draft_note):
        mgr.sign_note(draft_note, "therapist-1", {"plan": "P1"})
        mgr.amend_note(draft_note, "plan", "P2", "Updated plan", "therapist-1")
        amendments = mgr.get_amendments(draft_note)
        assert len(amendments) == 1
        assert amendments[0].section == "plan"


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------

class TestSigningCompliance:
    def test_within_30_days_ok(self, mgr):
        svc_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        mgr.register_note("note-comp", service_date=svc_date)
        result = mgr.check_signing_compliance("note-comp")
        assert result["overdue"] is False
        assert result["days_since_service"] == 10

    def test_over_30_days_overdue(self, mgr):
        svc_date = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        mgr.register_note("note-late", service_date=svc_date)
        result = mgr.check_signing_compliance("note-late")
        assert result["overdue"] is True
        assert result["days_since_service"] == 45

    def test_signed_not_overdue(self, mgr):
        """Even if service date is old, a signed note is not overdue."""
        svc_date = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        mgr.register_note("note-ok", service_date=svc_date)
        mgr.sign_note("note-ok", "therapist-1")
        result = mgr.check_signing_compliance("note-ok")
        assert result["overdue"] is False
        assert result["signed"] is True
