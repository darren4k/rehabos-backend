"""Tests for Medicare 8-minute rule validation."""

import pytest

from rehab_os.billing.cpt_mapper import CPTBillingLine
from rehab_os.billing.eight_minute_rule import UnitValidation, _minutes_to_units, validate_units


class TestMinutesToUnits:
    def test_zero_minutes(self):
        assert _minutes_to_units(0) == 0

    def test_less_than_8_minutes(self):
        assert _minutes_to_units(7) == 0

    def test_8_minutes_is_1_unit(self):
        assert _minutes_to_units(8) == 1

    def test_15_minutes_is_1_unit(self):
        assert _minutes_to_units(15) == 1

    def test_22_minutes_is_1_unit(self):
        assert _minutes_to_units(22) == 1

    def test_23_minutes_is_2_units(self):
        assert _minutes_to_units(23) == 2

    def test_37_minutes_is_2_units(self):
        assert _minutes_to_units(37) == 2

    def test_38_minutes_is_3_units(self):
        assert _minutes_to_units(38) == 3

    def test_52_minutes_is_3_units(self):
        assert _minutes_to_units(52) == 3

    def test_53_minutes_is_4_units(self):
        assert _minutes_to_units(53) == 4

    def test_67_minutes_is_4_units(self):
        assert _minutes_to_units(67) == 4


class TestValidateUnits:
    def test_single_timed_code(self):
        lines = [
            CPTBillingLine(code="97110", description="Ther Ex", units=0, minutes=15, category="timed")
        ]
        result = validate_units(lines)
        assert result.total_timed_units == 1
        assert result.billing_lines[0].units == 1

    def test_single_untimed_code(self):
        lines = [
            CPTBillingLine(code="97010", description="Hot Pack", units=0, minutes=10, category="untimed")
        ]
        result = validate_units(lines)
        assert result.total_untimed_units == 1
        assert result.total_timed_units == 0
        assert result.billing_lines[0].units == 1

    def test_multiple_timed_codes_total_distribution(self):
        lines = [
            CPTBillingLine(code="97110", description="Ther Ex", units=0, minutes=15, category="timed"),
            CPTBillingLine(code="97116", description="Gait", units=0, minutes=15, category="timed"),
        ]
        result = validate_units(lines)
        # 30 total minutes = 2 units, distributed evenly
        assert result.total_timed_minutes == 30
        assert result.total_timed_units == 2
        total_assigned = sum(l.units for l in result.billing_lines if l.category == "timed")
        assert total_assigned == 2

    def test_three_codes_uneven_distribution(self):
        lines = [
            CPTBillingLine(code="97110", description="Ther Ex", units=0, minutes=20, category="timed"),
            CPTBillingLine(code="97116", description="Gait", units=0, minutes=10, category="timed"),
            CPTBillingLine(code="97140", description="Manual", units=0, minutes=10, category="timed"),
        ]
        result = validate_units(lines)
        # 40 total minutes = 3 units
        assert result.total_timed_minutes == 40
        assert result.total_timed_units == 3
        # 97110 should get more units (20/40 of 3 = 1.5, rounded up)
        timed = {l.code: l.units for l in result.billing_lines if l.category == "timed"}
        assert sum(timed.values()) == 3
        assert timed["97110"] >= 1

    def test_less_than_8_minutes_no_units(self):
        lines = [
            CPTBillingLine(code="97110", description="Ther Ex", units=0, minutes=5, category="timed")
        ]
        result = validate_units(lines)
        assert result.total_timed_units == 0
        assert len(result.warnings) > 0

    def test_mixed_timed_and_untimed(self):
        lines = [
            CPTBillingLine(code="97110", description="Ther Ex", units=0, minutes=15, category="timed"),
            CPTBillingLine(code="97010", description="Hot Pack", units=0, minutes=10, category="untimed"),
        ]
        result = validate_units(lines)
        assert result.total_timed_units == 1
        assert result.total_untimed_units == 1
        assert result.total_units == 2

    def test_empty_input(self):
        result = validate_units([])
        assert result.total_units == 0
        assert result.is_valid

    def test_eval_code_treated_as_non_timed(self):
        lines = [
            CPTBillingLine(code="97162", description="PT Eval", units=1, minutes=30, category="eval"),
            CPTBillingLine(code="97110", description="Ther Ex", units=0, minutes=15, category="timed"),
        ]
        result = validate_units(lines)
        # Eval should be untimed (1 unit), ther ex should be 1 timed unit
        assert result.total_timed_units == 1
