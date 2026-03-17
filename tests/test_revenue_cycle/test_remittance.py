"""Tests for EDI 835 remittance advice parsing."""

import pytest

from rehab_os.revenue_cycle.remittance import (
    CARC_DESCRIPTIONS,
    GROUP_CODES,
    PaymentLine,
    RemittanceAdvice,
    RemittanceParser,
)


def _build_835(segments: list[str], element_sep: str = "*", seg_term: str = "~") -> str:
    """Build a minimal 835 EDI string from segment strings."""
    return seg_term.join(segments) + seg_term


# ---------------------------------------------------------------------------
# Parse 835
# ---------------------------------------------------------------------------

class TestParse835:
    def test_basic_payment_lines_extracted(self):
        edi = _build_835([
            # Pad ISA to 106+ chars so seg_term detection works
            "ISA*00*          *00*          *ZZ*SENDER         *ZZ*RECVR          *260315*1200*^*00501*000000001*0*T*:~",
            "BPR*I*250.00*C*ACH*CTX*01*999999*DA*12345678***01*888888*DA*87654321*20260315",
            "TRN*1*ERA12345",
            "N1*PR*BlueCross*PI*BCBS01",
            "CLP*CLM001*1*300.00*250.00*20.00",
            "SVC*HC:97110:GP*150.00*125.00",
            "CAS*CO*45*25.00",
            "SVC*HC:97140:GP*150.00*125.00",
            "CAS*PR*1*20.00",
        ])
        parser = RemittanceParser()
        era = parser.parse_835(edi)

        assert era.total_paid == 250.0
        assert era.payer_name == "BlueCross"
        assert era.payer_id == "BCBS01"
        assert era.era_id == "ERA12345"
        assert era.check_number == "ERA12345"
        assert len(era.lines) == 2

    def test_svc_cpt_code_parsed(self):
        edi = _build_835([
            "ISA*00*          *00*          *ZZ*SENDER         *ZZ*RECVR          *260315*1200*^*00501*000000001*0*T*:~",
            "BPR*I*100.00",
            "CLP*CLM001*1*100.00*100.00",
            "SVC*HC:97530*100.00*100.00",
        ])
        parser = RemittanceParser()
        era = parser.parse_835(edi)
        assert era.lines[0].cpt_code == "97530"

    def test_payment_date_from_bpr(self):
        edi = _build_835([
            "ISA*00*          *00*          *ZZ*SENDER         *ZZ*RECVR          *260315*1200*^*00501*000000001*0*T*:~",
            "BPR*I*100.00*C*ACH*CTX*01*999999*DA*12345678***01*888888*DA*87654321*20260315*CC",
        ])
        parser = RemittanceParser()
        era = parser.parse_835(edi)
        assert era.payment_date == "20260315"


# ---------------------------------------------------------------------------
# Denials
# ---------------------------------------------------------------------------

class TestGetDenials:
    def test_denied_lines_have_zero_paid(self):
        era = RemittanceAdvice(
            lines=[
                PaymentLine(claim_id="C1", cpt_code="97110", billed_amount=100.0, paid_amount=0.0),
                PaymentLine(claim_id="C1", cpt_code="97140", billed_amount=80.0, paid_amount=80.0),
                PaymentLine(claim_id="C2", cpt_code="97530", billed_amount=60.0, paid_amount=0.0),
            ]
        )
        parser = RemittanceParser()
        denials = parser.get_denials(era)
        assert len(denials) == 2
        assert all(d.is_denied for d in denials)

    def test_no_denials_when_all_paid(self):
        era = RemittanceAdvice(
            lines=[
                PaymentLine(claim_id="C1", cpt_code="97110", billed_amount=100.0, paid_amount=80.0),
            ]
        )
        parser = RemittanceParser()
        assert len(parser.get_denials(era)) == 0

    def test_denial_count_property(self):
        era = RemittanceAdvice(
            lines=[
                PaymentLine(billed_amount=100.0, paid_amount=0.0),
                PaymentLine(billed_amount=50.0, paid_amount=0.0),
                PaymentLine(billed_amount=75.0, paid_amount=75.0),
            ]
        )
        assert era.denial_count == 2


# ---------------------------------------------------------------------------
# Write-offs
# ---------------------------------------------------------------------------

class TestWriteOffs:
    def test_calculate_write_offs(self):
        era = RemittanceAdvice(total_billed=1000.0, total_allowed=750.0)
        parser = RemittanceParser()
        assert parser.calculate_write_offs(era) == 250.0

    def test_write_off_zero_when_fully_allowed(self):
        era = RemittanceAdvice(total_billed=500.0, total_allowed=500.0)
        assert era.write_off_amount == 0.0

    def test_write_off_property(self):
        era = RemittanceAdvice(total_billed=300.0, total_allowed=200.0)
        assert era.write_off_amount == 100.0

    def test_write_off_never_negative(self):
        era = RemittanceAdvice(total_billed=100.0, total_allowed=200.0)
        assert era.write_off_amount == 0.0


# ---------------------------------------------------------------------------
# Adjustment reason codes
# ---------------------------------------------------------------------------

class TestAdjustmentReasonCodes:
    def test_payment_line_adjustment_description_co_45(self):
        line = PaymentLine(adjustment_reason="CO-45")
        desc = line.adjustment_description
        assert "Contractual Obligation" in desc
        assert "fee schedule" in desc.lower()

    def test_payment_line_adjustment_description_pr_1(self):
        line = PaymentLine(adjustment_reason="PR-1")
        desc = line.adjustment_description
        assert "Patient Responsibility" in desc
        assert "Deductible" in desc

    def test_payment_line_adjustment_description_none(self):
        line = PaymentLine(adjustment_reason=None)
        assert line.adjustment_description == ""

    def test_payment_line_adjustment_description_unknown_code(self):
        line = PaymentLine(adjustment_reason="CO-9999")
        desc = line.adjustment_description
        assert "Contractual Obligation" in desc
        assert "9999" in desc

    def test_carc_descriptions_has_common_codes(self):
        assert "1" in CARC_DESCRIPTIONS
        assert "45" in CARC_DESCRIPTIONS
        assert "50" in CARC_DESCRIPTIONS
        assert "197" in CARC_DESCRIPTIONS

    def test_group_codes(self):
        assert GROUP_CODES["CO"] == "Contractual Obligation"
        assert GROUP_CODES["PR"] == "Patient Responsibility"

    def test_payment_rate(self):
        era = RemittanceAdvice(total_billed=400.0, total_paid=300.0)
        assert era.payment_rate == 75.0

    def test_payment_rate_zero_billed(self):
        era = RemittanceAdvice(total_billed=0.0, total_paid=0.0)
        assert era.payment_rate == 0.0


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_summary_keys(self):
        era = RemittanceAdvice(
            era_id="ERA1",
            payer_name="BCBS",
            check_number="CHK123",
            payment_date="20260315",
            total_billed=500.0,
            total_allowed=400.0,
            total_paid=380.0,
            total_patient_responsibility=20.0,
            lines=[PaymentLine(billed_amount=500.0, paid_amount=380.0)],
        )
        parser = RemittanceParser()
        summary = parser.summarize(era)
        assert summary["era_id"] == "ERA1"
        assert summary["payer"] == "BCBS"
        assert summary["total_billed"] == 500.0
        assert summary["total_paid"] == 380.0
        assert summary["line_count"] == 1
        assert summary["denial_count"] == 0
        assert "write_off" in summary
        assert "payment_rate_pct" in summary
