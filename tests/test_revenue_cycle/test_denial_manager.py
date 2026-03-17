"""Tests for denial tracking, analytics, and appeal lifecycle."""

import pytest
from datetime import datetime, timedelta, timezone

from rehab_os.revenue_cycle.denial_manager import (
    APPEAL_DEADLINES,
    DENIAL_STATUSES,
    Denial,
    DenialManager,
)
from rehab_os.revenue_cycle.remittance import PaymentLine


def _denied_line(**overrides) -> PaymentLine:
    defaults = dict(
        claim_id="CLM001",
        cpt_code="97110",
        billed_amount=150.0,
        paid_amount=0.0,
        adjustment_reason="CO-50",
        denial_code="50",
    )
    defaults.update(overrides)
    return PaymentLine(**defaults)


# ---------------------------------------------------------------------------
# Record denial
# ---------------------------------------------------------------------------

class TestRecordDenial:
    def test_record_creates_denial(self):
        mgr = DenialManager()
        line = _denied_line()
        denial = mgr.record_denial(line, patient_id="PAT1", patient_name="Jane Doe",
                                   payer_name="BCBS")
        assert denial.denial_id.startswith("DEN-")
        assert denial.claim_id == "CLM001"
        assert denial.patient_id == "PAT1"
        assert denial.billed_amount == 150.0
        assert denial.status == "new"

    def test_record_sets_appeal_deadline(self):
        mgr = DenialManager()
        line = _denied_line()
        denial = mgr.record_denial(line, payer_type="Medicare")
        assert denial.appeal_deadline is not None
        # Medicare = 120 days
        deadline = datetime.strptime(denial.appeal_deadline, "%Y%m%d")
        expected_min = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=119)
        assert deadline >= expected_min

    def test_record_commercial_deadline(self):
        mgr = DenialManager()
        line = _denied_line()
        denial = mgr.record_denial(line, payer_type="Commercial")
        deadline = datetime.strptime(denial.appeal_deadline, "%Y%m%d")
        expected_min = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=179)
        assert deadline >= expected_min

    def test_record_default_deadline(self):
        mgr = DenialManager()
        line = _denied_line()
        denial = mgr.record_denial(line, payer_type="unknown_type")
        # default = 90 days
        deadline = datetime.strptime(denial.appeal_deadline, "%Y%m%d")
        expected_min = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=89)
        assert deadline >= expected_min

    def test_record_populates_denial_reason(self):
        mgr = DenialManager()
        line = _denied_line(adjustment_reason="CO-50", denial_code="50")
        denial = mgr.record_denial(line)
        assert "medical necessity" in denial.denial_reason.lower() or "CO" in denial.denial_reason

    def test_record_uses_claim_id_from_line_if_not_provided(self):
        mgr = DenialManager()
        line = _denied_line(claim_id="LINE_CLM")
        denial = mgr.record_denial(line)
        assert denial.claim_id == "LINE_CLM"

    def test_record_overrides_claim_id(self):
        mgr = DenialManager()
        line = _denied_line(claim_id="LINE_CLM")
        denial = mgr.record_denial(line, claim_id="OVERRIDE_CLM")
        assert denial.claim_id == "OVERRIDE_CLM"


# ---------------------------------------------------------------------------
# Get open denials
# ---------------------------------------------------------------------------

class TestGetOpenDenials:
    def test_returns_new_reviewing_appealing(self):
        mgr = DenialManager()
        d1 = mgr.record_denial(_denied_line())
        d2 = mgr.record_denial(_denied_line())
        d3 = mgr.record_denial(_denied_line())
        mgr.update_status(d2.denial_id, "reviewing")
        mgr.update_status(d3.denial_id, "won")
        open_denials = mgr.get_open_denials()
        open_ids = {d.denial_id for d in open_denials}
        assert d1.denial_id in open_ids
        assert d2.denial_id in open_ids
        assert d3.denial_id not in open_ids

    def test_empty_when_none(self):
        mgr = DenialManager()
        assert mgr.get_open_denials() == []


# ---------------------------------------------------------------------------
# Status lifecycle
# ---------------------------------------------------------------------------

class TestDenialStatusLifecycle:
    def test_new_to_reviewing(self):
        mgr = DenialManager()
        denial = mgr.record_denial(_denied_line())
        assert denial.status == "new"
        updated = mgr.update_status(denial.denial_id, "reviewing")
        assert updated.status == "reviewing"

    def test_reviewing_to_appealing(self):
        mgr = DenialManager()
        denial = mgr.record_denial(_denied_line())
        mgr.update_status(denial.denial_id, "reviewing")
        updated = mgr.update_status(denial.denial_id, "appealing")
        assert updated.status == "appealing"

    def test_appealing_to_won(self):
        mgr = DenialManager()
        denial = mgr.record_denial(_denied_line())
        mgr.update_status(denial.denial_id, "appealing")
        updated = mgr.update_status(denial.denial_id, "won", notes="Appeal successful")
        assert updated.status == "won"
        assert updated.notes == "Appeal successful"

    def test_invalid_status_raises(self):
        mgr = DenialManager()
        denial = mgr.record_denial(_denied_line())
        with pytest.raises(ValueError, match="Invalid status"):
            mgr.update_status(denial.denial_id, "invalid_status")

    def test_update_nonexistent_returns_none(self):
        mgr = DenialManager()
        result = mgr.update_status("DEN-NONEXIST", "reviewing")
        assert result is None

    def test_update_sets_updated_at(self):
        mgr = DenialManager()
        denial = mgr.record_denial(_denied_line())
        original_updated = denial.updated_at
        updated = mgr.update_status(denial.denial_id, "reviewing")
        assert updated.updated_at  # non-empty

    def test_all_valid_statuses(self):
        for status in DENIAL_STATUSES:
            mgr = DenialManager()
            denial = mgr.record_denial(_denied_line())
            updated = mgr.update_status(denial.denial_id, status)
            assert updated.status == status


# ---------------------------------------------------------------------------
# Expiring appeals
# ---------------------------------------------------------------------------

class TestExpiringAppeals:
    def test_expiring_within_window(self):
        mgr = DenialManager()
        line = _denied_line()
        denial = mgr.record_denial(line, payer_type="Medicaid")  # 60-day deadline
        # The denial just got created with a 60-day deadline, so it should
        # appear when querying for 90 days out
        expiring = mgr.get_expiring_appeals(days=90)
        assert any(d.denial_id == denial.denial_id for d in expiring)

    def test_not_expiring_within_short_window(self):
        mgr = DenialManager()
        denial = mgr.record_denial(_denied_line(), payer_type="Commercial")  # 180 days
        expiring = mgr.get_expiring_appeals(days=7)
        assert len(expiring) == 0

    def test_won_denial_not_in_expiring(self):
        mgr = DenialManager()
        denial = mgr.record_denial(_denied_line(), payer_type="Medicaid")
        mgr.update_status(denial.denial_id, "won")
        expiring = mgr.get_expiring_appeals(days=90)
        assert len(expiring) == 0

    def test_sorted_by_deadline(self):
        mgr = DenialManager()
        d1 = mgr.record_denial(_denied_line(), payer_type="Medicaid")   # 60 days
        d2 = mgr.record_denial(_denied_line(), payer_type="Medicare")   # 120 days
        expiring = mgr.get_expiring_appeals(days=365)
        if len(expiring) >= 2:
            assert expiring[0].appeal_deadline <= expiring[1].appeal_deadline


# ---------------------------------------------------------------------------
# Denial stats
# ---------------------------------------------------------------------------

class TestDenialStats:
    def test_empty_stats(self):
        mgr = DenialManager()
        stats = mgr.get_denial_stats()
        assert stats["total_denials"] == 0
        assert stats["total_amount"] == 0.0
        assert stats["appeal_success_rate"] == 0.0

    def test_stats_with_denials(self):
        mgr = DenialManager()
        mgr.record_denial(_denied_line(billed_amount=100.0, denial_code="50"),
                          payer_name="BCBS")
        mgr.record_denial(_denied_line(billed_amount=200.0, denial_code="50"),
                          payer_name="BCBS")
        mgr.record_denial(_denied_line(billed_amount=150.0, denial_code="96"),
                          payer_name="Aetna")
        stats = mgr.get_denial_stats()
        assert stats["total_denials"] == 3
        assert stats["total_amount"] == 450.0
        assert stats["open_count"] == 3
        assert stats["by_code"]["50"] == 2
        assert stats["by_code"]["96"] == 1
        assert stats["by_payer"]["BCBS"] == 2
        assert stats["by_payer"]["Aetna"] == 1

    def test_stats_appeal_success_rate(self):
        mgr = DenialManager()
        d1 = mgr.record_denial(_denied_line())
        d2 = mgr.record_denial(_denied_line())
        d3 = mgr.record_denial(_denied_line())
        mgr.update_status(d1.denial_id, "won")
        mgr.update_status(d2.denial_id, "lost")
        stats = mgr.get_denial_stats()
        # 1 won / 2 resolved = 50%
        assert stats["appeal_success_rate"] == 50.0
        assert stats["won_count"] == 1
        assert stats["lost_count"] == 1


# ---------------------------------------------------------------------------
# Denial dataclass
# ---------------------------------------------------------------------------

class TestDenialDataclass:
    def test_auto_generated_id(self):
        d = Denial()
        assert d.denial_id.startswith("DEN-")

    def test_auto_filled_reason_from_code(self):
        d = Denial(denial_code="45")
        assert "fee schedule" in d.denial_reason.lower()

    def test_is_appealable_new(self):
        d = Denial(status="new")
        assert d.is_appealable is True

    def test_is_appealable_won(self):
        d = Denial(status="won")
        assert d.is_appealable is False

    def test_is_appealable_lost(self):
        d = Denial(status="lost")
        assert d.is_appealable is False

    def test_is_appealable_written_off(self):
        d = Denial(status="written_off")
        assert d.is_appealable is False

    def test_days_until_deadline_none_when_no_deadline(self):
        d = Denial()
        assert d.days_until_appeal_deadline is None

    def test_days_until_deadline_future(self):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y%m%d")
        d = Denial(appeal_deadline=future)
        remaining = d.days_until_appeal_deadline
        assert remaining is not None
        assert 29 <= remaining <= 31

    def test_days_until_deadline_past(self):
        past = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y%m%d")
        d = Denial(appeal_deadline=past)
        assert d.days_until_appeal_deadline == 0
