"""ICD-10 code suggestion from clinical context.

Maps common rehabilitation diagnoses and conditions to ICD-10-CM codes.
Uses keyword matching against a curated rehab-focused code table.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field


class ICD10Suggestion(BaseModel):
    """A suggested ICD-10 code with confidence."""

    code: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # "diagnosis", "chief_complaint", "intervention_context"


# ── ICD-10 Code Reference (Rehab-focused) ────────────────────────────────────

ICD10_TABLE: dict[str, tuple[str, str]] = {
    # Knee
    "M17.11": ("Primary osteoarthritis, right knee", "knee"),
    "M17.12": ("Primary osteoarthritis, left knee", "knee"),
    "M17.0": ("Bilateral primary osteoarthritis of knee", "knee"),
    "M23.51": ("Chronic instability of knee, right", "knee"),
    "M23.52": ("Chronic instability of knee, left", "knee"),
    "S83.511A": ("Sprain of ACL of right knee, initial", "knee"),
    "S83.512A": ("Sprain of ACL of left knee, initial", "knee"),
    "Z96.651": ("Presence of right artificial knee joint", "knee"),
    "Z96.652": ("Presence of left artificial knee joint", "knee"),
    "M25.561": ("Pain in right knee", "knee"),
    "M25.562": ("Pain in left knee", "knee"),
    # Hip
    "M16.11": ("Primary osteoarthritis, right hip", "hip"),
    "M16.12": ("Primary osteoarthritis, left hip", "hip"),
    "Z96.641": ("Presence of right artificial hip joint", "hip"),
    "Z96.642": ("Presence of left artificial hip joint", "hip"),
    "S72.001A": ("Fracture of unspecified part of neck of right femur, initial", "hip"),
    "M25.551": ("Pain in right hip", "hip"),
    "M25.552": ("Pain in left hip", "hip"),
    # Shoulder
    "M75.111": ("Rotator cuff tear, right shoulder, incomplete", "shoulder"),
    "M75.112": ("Rotator cuff tear, left shoulder, incomplete", "shoulder"),
    "M75.101": ("Rotator cuff syndrome, right shoulder", "shoulder"),
    "M75.102": ("Rotator cuff syndrome, left shoulder", "shoulder"),
    "M75.01": ("Adhesive capsulitis of right shoulder", "shoulder"),
    "M75.02": ("Adhesive capsulitis of left shoulder", "shoulder"),
    "M25.511": ("Pain in right shoulder", "shoulder"),
    "M25.512": ("Pain in left shoulder", "shoulder"),
    # Back / Spine
    "M54.5": ("Low back pain", "back"),
    "M54.50": ("Low back pain, unspecified", "back"),
    "M54.51": ("Vertebrogenic low back pain", "back"),
    "M54.2": ("Cervicalgia", "neck"),
    "M47.816": ("Spondylosis without myelopathy, lumbar", "back"),
    "M47.812": ("Spondylosis without myelopathy, cervical", "neck"),
    "M51.16": ("Intervertebral disc degeneration, lumbar", "back"),
    "M54.41": ("Lumbago with sciatica, right side", "back"),
    "M54.42": ("Lumbago with sciatica, left side", "back"),
    # Neurological
    "I63.9": ("Cerebral infarction, unspecified (CVA)", "neuro"),
    "G81.91": ("Hemiplegia, unspecified, right dominant side", "neuro"),
    "G81.92": ("Hemiplegia, unspecified, left dominant side", "neuro"),
    "G20": ("Parkinson's disease", "neuro"),
    "G35": ("Multiple sclerosis", "neuro"),
    "G82.20": ("Paraplegia, unspecified", "neuro"),
    # Ankle / Foot
    "S93.401A": ("Sprain of unspecified ligament of right ankle, initial", "ankle"),
    "S93.402A": ("Sprain of unspecified ligament of left ankle, initial", "ankle"),
    "M25.571": ("Pain in right ankle and joints of right foot", "ankle"),
    "M25.572": ("Pain in left ankle and joints of left foot", "ankle"),
    # General / Functional
    "R26.2": ("Difficulty in walking, not elsewhere classified", "gait"),
    "R26.81": ("Unsteadiness on feet", "balance"),
    "R26.89": ("Other abnormalities of gait and mobility", "gait"),
    "R29.6": ("Repeated falls", "balance"),
    "Z87.39": ("Personal history of other musculoskeletal disorders", "general"),
    "M62.81": ("Muscle weakness (generalized)", "general"),
    "R53.1": ("Weakness", "general"),
    "Z51.89": ("Encounter for other specified aftercare", "general"),
    # Wrist / Hand / Elbow
    "M25.531": ("Pain in right wrist", "wrist"),
    "M25.532": ("Pain in left wrist", "wrist"),
    "M25.521": ("Pain in right elbow", "elbow"),
    "M25.522": ("Pain in left elbow", "elbow"),
}

# ── Keyword → Body Region → ICD-10 Mapping ───────────────────────────────────

KEYWORD_PATTERNS: list[tuple[str, str, str]] = [
    # (regex pattern, laterality_hint, body_region)
    (r"(?:right|r)\s*knee\s*(?:oa|osteoarthritis)", "right", "M17.11"),
    (r"(?:left|l)\s*knee\s*(?:oa|osteoarthritis)", "left", "M17.12"),
    (r"knee\s*(?:oa|osteoarthritis)", "", "M17.0"),
    (r"(?:right|r)\s*(?:tka|total knee|knee replacement)", "right", "Z96.651"),
    (r"(?:left|l)\s*(?:tka|total knee|knee replacement)", "left", "Z96.652"),
    (r"(?:right|r)\s*(?:tha|total hip|hip replacement)", "right", "Z96.641"),
    (r"(?:left|l)\s*(?:tha|total hip|hip replacement)", "left", "Z96.642"),
    (r"(?:right|r)\s*hip\s*(?:oa|osteoarthritis)", "right", "M16.11"),
    (r"(?:left|l)\s*hip\s*(?:oa|osteoarthritis)", "left", "M16.12"),
    (r"(?:right|r)\s*rotator\s*cuff", "right", "M75.111"),
    (r"(?:left|l)\s*rotator\s*cuff", "left", "M75.112"),
    (r"(?:right|r)\s*frozen\s*shoulder|(?:right|r)\s*adhesive\s*capsulitis", "right", "M75.01"),
    (r"(?:left|l)\s*frozen\s*shoulder|(?:left|l)\s*adhesive\s*capsulitis", "left", "M75.02"),
    (r"(?:low\s*back|lumbar|lbp)", "", "M54.5"),
    (r"(?:neck|cervical)\s*pain", "", "M54.2"),
    (r"sciatica.*(?:right|r)", "right", "M54.41"),
    (r"sciatica.*(?:left|l)", "left", "M54.42"),
    (r"(?:cva|stroke|cerebral\s*infarct)", "", "I63.9"),
    (r"(?:hemiplegia|hemiparesis).*(?:right|r)", "right", "G81.91"),
    (r"(?:hemiplegia|hemiparesis).*(?:left|l)", "left", "G81.92"),
    (r"parkinson", "", "G20"),
    (r"(?:ms|multiple\s*sclerosis)", "", "G35"),
    (r"(?:right|r)\s*ankle\s*sprain", "right", "S93.401A"),
    (r"(?:left|l)\s*ankle\s*sprain", "left", "S93.402A"),
    (r"(?:fall risk|repeated falls|history of falls)", "", "R29.6"),
    (r"(?:gait\s*(?:abnormality|deviation|deficit)|difficulty walking)", "", "R26.2"),
    (r"(?:balance\s*(?:deficit|impairment)|unsteady)", "", "R26.81"),
    (r"(?:general\s*)?(?:weakness|deconditioning)", "", "M62.81"),
    (r"(?:right|r)\s*knee\s*pain", "right", "M25.561"),
    (r"(?:left|l)\s*knee\s*pain", "left", "M25.562"),
    (r"(?:right|r)\s*hip\s*pain", "right", "M25.551"),
    (r"(?:left|l)\s*hip\s*pain", "left", "M25.552"),
    (r"(?:right|r)\s*shoulder\s*pain", "right", "M25.511"),
    (r"(?:left|l)\s*shoulder\s*pain", "left", "M25.512"),
]


def suggest_icd10_codes(
    diagnosis_list: list[str],
    chief_complaint: str | None = None,
    pain_location: str | None = None,
) -> list[ICD10Suggestion]:
    """Suggest ICD-10 codes from diagnosis list, chief complaint, and pain location.

    Args:
        diagnosis_list: Known diagnoses (from patient history or referral).
        chief_complaint: Current chief complaint text.
        pain_location: Documented pain location.

    Returns:
        List of ICD10Suggestion objects, sorted by confidence (highest first).
    """
    suggestions: dict[str, ICD10Suggestion] = {}

    # 1. Match from explicit diagnosis list (highest confidence)
    for dx in diagnosis_list:
        _match_text(dx, "diagnosis", 0.95, suggestions)

    # 2. Match from chief complaint
    if chief_complaint:
        _match_text(chief_complaint, "chief_complaint", 0.75, suggestions)

    # 3. Match from pain location
    if pain_location:
        _match_text(pain_location, "pain_location", 0.60, suggestions)

    # Sort by confidence descending
    result = sorted(suggestions.values(), key=lambda s: s.confidence, reverse=True)
    return result


def _match_text(
    text: str,
    source: str,
    base_confidence: float,
    suggestions: dict[str, ICD10Suggestion],
) -> None:
    """Match text against keyword patterns and add to suggestions."""
    lower = text.lower().strip()

    # Try direct ICD-10 code reference (e.g., "M17.11 R knee OA")
    code_match = re.findall(r"[A-Z]\d{2}\.?\d{0,4}[A-Z]?", text)
    for code_raw in code_match:
        code = code_raw.replace(".", "")
        # Try with and without dot
        for c in (code_raw, code[:3] + "." + code[3:] if len(code) > 3 else code):
            if c in ICD10_TABLE and c not in suggestions:
                desc, _ = ICD10_TABLE[c]
                suggestions[c] = ICD10Suggestion(
                    code=c, description=desc, confidence=0.99, source=source
                )

    # Try keyword pattern matching
    for pattern, _, icd_code in KEYWORD_PATTERNS:
        if re.search(pattern, lower):
            if icd_code not in suggestions:
                desc = ICD10_TABLE.get(icd_code, (icd_code, ""))[0]
                suggestions[icd_code] = ICD10Suggestion(
                    code=icd_code,
                    description=desc,
                    confidence=base_confidence,
                    source=source,
                )
