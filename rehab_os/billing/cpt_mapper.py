"""CPT code mapping from rehabilitation interventions.

Maps documented interventions to the correct CPT codes used in
physical therapy, occupational therapy, and speech-language pathology billing.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CPTCode(BaseModel):
    """A single CPT billing code with metadata."""

    code: str
    description: str
    category: str  # timed | untimed | eval
    default_minutes: int = 15
    requires_direct_contact: bool = True


# ── CPT Code Reference Table ─────────────────────────────────────────────────
# Based on AMA CPT and CMS Medicare guidelines for rehab therapy

CPT_CODES: dict[str, CPTCode] = {
    "97110": CPTCode(
        code="97110",
        description="Therapeutic Exercise",
        category="timed",
        default_minutes=15,
    ),
    "97112": CPTCode(
        code="97112",
        description="Neuromuscular Re-education",
        category="timed",
        default_minutes=15,
    ),
    "97116": CPTCode(
        code="97116",
        description="Gait Training",
        category="timed",
        default_minutes=15,
    ),
    "97140": CPTCode(
        code="97140",
        description="Manual Therapy",
        category="timed",
        default_minutes=15,
    ),
    "97530": CPTCode(
        code="97530",
        description="Therapeutic Activities",
        category="timed",
        default_minutes=15,
    ),
    "97535": CPTCode(
        code="97535",
        description="Self-Care/Home Management Training",
        category="timed",
        default_minutes=15,
    ),
    "97542": CPTCode(
        code="97542",
        description="Wheelchair Management Training",
        category="timed",
        default_minutes=15,
    ),
    "97150": CPTCode(
        code="97150",
        description="Group Therapeutic Procedures",
        category="timed",
        default_minutes=15,
        requires_direct_contact=False,
    ),
    "97010": CPTCode(
        code="97010",
        description="Hot/Cold Packs",
        category="untimed",
        requires_direct_contact=False,
    ),
    "97014": CPTCode(
        code="97014",
        description="Electrical Stimulation (unattended)",
        category="untimed",
        requires_direct_contact=False,
    ),
    "97032": CPTCode(
        code="97032",
        description="Electrical Stimulation (attended)",
        category="timed",
        default_minutes=15,
    ),
    "97035": CPTCode(
        code="97035",
        description="Ultrasound",
        category="timed",
        default_minutes=15,
    ),
    "97033": CPTCode(
        code="97033",
        description="Iontophoresis",
        category="timed",
        default_minutes=15,
    ),
    "97760": CPTCode(
        code="97760",
        description="Orthotic Management and Training",
        category="timed",
        default_minutes=15,
    ),
    "97761": CPTCode(
        code="97761",
        description="Prosthetic Training",
        category="timed",
        default_minutes=15,
    ),
    "97763": CPTCode(
        code="97763",
        description="Orthotic/Prosthetic Checkout",
        category="timed",
        default_minutes=15,
    ),
    "97161": CPTCode(
        code="97161",
        description="PT Evaluation - Low Complexity",
        category="eval",
        default_minutes=20,
    ),
    "97162": CPTCode(
        code="97162",
        description="PT Evaluation - Moderate Complexity",
        category="eval",
        default_minutes=30,
    ),
    "97163": CPTCode(
        code="97163",
        description="PT Evaluation - High Complexity",
        category="eval",
        default_minutes=45,
    ),
    "97164": CPTCode(
        code="97164",
        description="PT Re-evaluation",
        category="eval",
        default_minutes=20,
    ),
    "97530_balance": CPTCode(
        code="97530",
        description="Therapeutic Activities (Balance)",
        category="timed",
        default_minutes=15,
    ),
}

# ── Intervention → CPT Mapping ────────────────────────────────────────────────
# Maps the intervention names extracted by EncounterBrain to CPT codes.
# An intervention can map to multiple possible codes.

INTERVENTION_TO_CPT: dict[str, list[str]] = {
    "therapeutic exercise": ["97110"],
    "gait training": ["97116"],
    "balance training": ["97530"],
    "manual therapy": ["97140"],
    "neuromuscular re-education": ["97112"],
    "transfer training": ["97530", "97535"],
    "patient education": ["97535"],
    "modalities": ["97010", "97035", "97032"],
    # Expanded mappings for common variations
    "strengthening": ["97110"],
    "stretching": ["97110"],
    "rom exercises": ["97110"],
    "ambulation": ["97116"],
    "walking": ["97116"],
    "balance activities": ["97530"],
    "therapeutic activities": ["97530"],
    "soft tissue mobilization": ["97140"],
    "joint mobilization": ["97140"],
    "motor control": ["97112"],
    "bed mobility": ["97530", "97535"],
    "sit to stand": ["97530"],
    "wheelchair training": ["97542"],
    "ultrasound": ["97035"],
    "e-stim": ["97032"],
    "electrical stimulation": ["97032"],
    "hot pack": ["97010"],
    "cold pack": ["97010"],
    "ice": ["97010"],
    "iontophoresis": ["97033"],
    "TENS": ["97014"],
}


class CPTBillingLine(BaseModel):
    """A single billing line item."""

    code: str
    description: str
    units: int = 1
    minutes: int = 15
    category: str = "timed"
    intervention_source: str = ""


def map_interventions_to_cpt(
    interventions: list[dict],
    note_type: str = "daily_note",
) -> list[CPTBillingLine]:
    """Map documented interventions to CPT billing codes.

    Args:
        interventions: List of intervention dicts with 'name' and optional 'duration_minutes'.
        note_type: Type of note (affects eval code selection).

    Returns:
        List of CPTBillingLine items with codes, descriptions, and units.
    """
    billing_lines: list[CPTBillingLine] = []
    seen_codes: set[str] = set()

    # Add eval code if this is an evaluation note
    if note_type in ("initial_evaluation", "evaluation"):
        billing_lines.append(
            CPTBillingLine(
                code="97162",
                description="PT Evaluation - Moderate Complexity",
                units=1,
                minutes=30,
                category="eval",
                intervention_source="evaluation",
            )
        )
        seen_codes.add("97162")
    elif note_type == "recertification":
        billing_lines.append(
            CPTBillingLine(
                code="97164",
                description="PT Re-evaluation",
                units=1,
                minutes=20,
                category="eval",
                intervention_source="re-evaluation",
            )
        )
        seen_codes.add("97164")

    for intervention in interventions:
        name = intervention.get("name", "").lower().strip()
        duration = intervention.get("duration_minutes") or 15

        # Find matching CPT codes
        cpt_codes = INTERVENTION_TO_CPT.get(name, [])
        if not cpt_codes:
            # Try partial matching
            for key, codes in INTERVENTION_TO_CPT.items():
                if key in name or name in key:
                    cpt_codes = codes
                    break

        if not cpt_codes:
            continue

        # Use the first (primary) CPT code for this intervention
        code = cpt_codes[0]
        if code in seen_codes:
            # Merge duration into existing line
            for line in billing_lines:
                if line.code == code:
                    line.minutes += duration
                    break
            continue

        cpt_info = CPT_CODES.get(code)
        if not cpt_info:
            continue

        billing_lines.append(
            CPTBillingLine(
                code=code,
                description=cpt_info.description,
                units=1,  # Will be calculated by 8-minute rule
                minutes=duration,
                category=cpt_info.category,
                intervention_source=name,
            )
        )
        seen_codes.add(code)

    return billing_lines
