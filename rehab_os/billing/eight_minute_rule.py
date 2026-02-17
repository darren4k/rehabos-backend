"""Medicare 8-Minute Rule validation for timed CPT codes.

The 8-minute rule determines how many units can be billed for timed services.
A provider must spend at least 8 minutes on a service to bill 1 unit.
Untimed codes are always 1 unit regardless of time.

CMS 8-Minute Rule table:
  >= 8 min  and <= 22 min  = 1 unit
  >= 23 min and <= 37 min  = 2 units
  >= 38 min and <= 52 min  = 3 units
  >= 53 min and <= 67 min  = 4 units
  >= 68 min and <= 82 min  = 5 units
  (pattern: each additional unit = +15 min)

For multiple timed codes, total minutes across all timed codes determines
the total billable units, then distributed proportionally.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from rehab_os.billing.cpt_mapper import CPTBillingLine


class UnitValidation(BaseModel):
    """Result of 8-minute rule validation."""

    billing_lines: list[CPTBillingLine] = Field(default_factory=list)
    total_timed_minutes: int = 0
    total_timed_units: int = 0
    total_untimed_units: int = 0
    total_units: int = 0
    is_valid: bool = True
    warnings: list[str] = Field(default_factory=list)


def _minutes_to_units(minutes: int) -> int:
    """Convert total timed minutes to billable units using the 8-minute rule.

    Must have at least 8 minutes to bill any unit.
    Each unit after the first requires an additional 15 minutes.
    """
    if minutes < 8:
        return 0
    if minutes <= 22:
        return 1
    # After first unit (8-22 min), each additional unit boundary is +15 min
    # 23-37 = 2, 38-52 = 3, 53-67 = 4, etc.
    return 1 + ((minutes - 8) // 15)


def validate_units(billing_lines: list[CPTBillingLine]) -> UnitValidation:
    """Validate and calculate correct units using the Medicare 8-minute rule.

    For timed codes: applies the 8-minute rule to total minutes across all
    timed services, then distributes units proportionally (largest remainder method).

    For untimed codes: always 1 unit each.

    Args:
        billing_lines: List of CPTBillingLine items from cpt_mapper.

    Returns:
        UnitValidation with corrected units and any warnings.
    """
    warnings: list[str] = []
    result_lines: list[CPTBillingLine] = []

    # Separate timed vs untimed
    timed: list[CPTBillingLine] = []
    untimed: list[CPTBillingLine] = []

    for line in billing_lines:
        if line.category == "timed":
            timed.append(line.model_copy())
        else:
            untimed.append(line.model_copy())

    # Untimed codes: always 1 unit
    total_untimed = 0
    for line in untimed:
        line.units = 1
        total_untimed += 1
        result_lines.append(line)

    # Timed codes: apply 8-minute rule to total, then distribute
    total_timed_minutes = sum(line.minutes for line in timed)
    total_timed_units = _minutes_to_units(total_timed_minutes)

    if total_timed_minutes > 0 and total_timed_units == 0:
        warnings.append(
            f"Total timed minutes ({total_timed_minutes}) is less than 8. "
            "No timed units billable under the 8-minute rule."
        )

    if timed and total_timed_units > 0:
        # Distribute units proportionally (largest remainder method)
        # Each code gets units proportional to its share of total minutes
        quotas = []
        for line in timed:
            share = (line.minutes / total_timed_minutes) * total_timed_units
            quotas.append((line, int(share), share - int(share)))

        # Assign integer part
        distributed = sum(q[1] for q in quotas)
        for line, base_units, _ in quotas:
            line.units = base_units

        # Distribute remaining units by largest remainder
        remaining = total_timed_units - distributed
        quotas.sort(key=lambda q: q[2], reverse=True)
        for i in range(remaining):
            quotas[i][0].units += 1

        # Ensure every timed code with time gets at least 0 units
        # (code with <8 min that's part of a larger total can still get units)
        for line, _, _ in quotas:
            result_lines.append(line)
    else:
        for line in timed:
            line.units = 0
            result_lines.append(line)

    # Validate: no individual timed code should have more units than minutes allow
    for line in result_lines:
        if line.category == "timed" and line.units > 0:
            max_possible = _minutes_to_units(line.minutes)
            if line.units > max_possible + 1:
                warnings.append(
                    f"{line.code} ({line.description}): {line.units} units for "
                    f"{line.minutes} min may be questioned."
                )

    # Total treatment time warning (Medicare expects ~1 unit per 15 min)
    if total_timed_minutes > 0 and total_timed_units > 0:
        expected_min = total_timed_units * 15
        if total_timed_minutes < expected_min - 7:
            warnings.append(
                f"Billing {total_timed_units} timed units but only {total_timed_minutes} "
                f"total minutes documented. Consider reviewing time allocation."
            )

    return UnitValidation(
        billing_lines=result_lines,
        total_timed_minutes=total_timed_minutes,
        total_timed_units=total_timed_units,
        total_untimed_units=total_untimed,
        total_units=total_timed_units + total_untimed,
        is_valid=len(warnings) == 0,
        warnings=warnings,
    )
