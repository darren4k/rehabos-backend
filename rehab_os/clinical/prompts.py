"""Medicare-compliant SOAP prompt templates for PT/OT/SLP documentation.

Adapted from DocPilot's MediMarvel standards. Produces skilled clinical
documentation that is Medicare-compliant, functionally justified, defensible
under audit, and reflective of true skilled clinical reasoning.

Setting-aware: outpatient, homecare, snf — each adjusts prompt context.
"""
from __future__ import annotations

from typing import Optional

from rehab_os.clinical.settings import ClinicalSetting

# ============================================================================
# PROHIBITED PHRASES — Never use these in documentation
# ============================================================================

PROHIBITED_PHRASES = [
    "tolerated treatment well",
    "tolerated well",
    "patient tolerated",
    "continues to benefit",
    "continues to progress",
    "patient did exercises",
    "completed exercises independently",
    "treatment as planned",
    "no complaints",
    "doing well",
    "stable",
    "maintained",
    "good session",
    "patient cooperative",
    "pleasant patient",
    "uneventful session",
]

SKILLED_REPLACEMENTS: dict[str, str] = {
    "tolerated treatment well": (
        "Patient demonstrated [specific response] with [cueing level] "
        "required for [functional task]"
    ),
    "continues to benefit": (
        "Patient shows measurable improvement in [metric] from [baseline] "
        "to [current], indicating skilled intervention remains medically necessary"
    ),
    "continues to progress": (
        "Patient progressed from [previous level] to [current level] as "
        "measured by [objective test/observation]"
    ),
    "patient did exercises": (
        "Therapist instructed patient in [specific exercise] with "
        "[cueing/modifications] to address [deficit/goal]"
    ),
    "completed exercises independently": (
        "Patient performed [exercise] with [cueing level], demonstrating "
        "[skill level] requiring therapist supervision for [reason]"
    ),
    "treatment as planned": (
        "Skilled interventions addressed [specific deficits] through "
        "[intervention details] with [patient response]"
    ),
    "no complaints": (
        "Patient reports [specific symptoms or status]. Denies pain/discomfort "
        "with [specific activities]"
    ),
    "doing well": (
        "Patient demonstrates [specific measurable improvement] in [function]"
    ),
    "stable": (
        "Patient maintains [metric] at [value], which represents [context "
        "of significance]"
    ),
    "maintained": (
        "Patient sustained [specific level] as measured by [test/observation]"
    ),
    "good session": (
        "Patient demonstrated [specific response] to skilled interventions "
        "targeting [deficit]"
    ),
}


# ============================================================================
# Payer-Specific Guidance
# ============================================================================

PAYER_GUIDANCE: dict[str, str] = {
    "Medicare": """Medicare Part B Documentation Requirements (CMS Standards):
- Document skilled care necessity — why a licensed therapist is required
- Each intervention must explain what, why, and how skill was applied
- Link all interventions to functional deficits with measurable progress
- Note if patient is making reasonable progress; if slow, justify continued care
- Follow LCD guidelines for your MAC region
- Include prognosis for continued improvement""",

    "HMO": """HMO Documentation Requirements:
- Document visits used vs. authorized; reference authorization number
- Focus on measurable functional improvements with timeline
- Demonstrate efficient use of authorized visits
- Note communication with PCP or referring physician""",

    "Commercial": """Commercial Insurance Documentation Requirements:
- Clear documentation of all services with time per CPT code
- Link interventions to functional goals
- Document functional deficits requiring skilled care
- Measurable progress with specific values""",

    "Medicaid": """Medicaid Documentation Requirements:
- Thoroughly document medical necessity and functional impact
- Note prior authorization requirements and PA number
- Document patient/caregiver education and HEP instruction
- Include measurable, time-bound goals and progress
- Follow state Medicaid manual requirements""",
}

# ============================================================================
# Common CPT Codes by Discipline
# ============================================================================

COMMON_CODES_PT = """PT CPT Codes:
- 97110: Therapeutic Exercise (strengthening, ROM, flexibility)
- 97116: Gait Training (ambulation, stairs, varied surfaces)
- 97140: Manual Therapy (soft tissue/joint mobilization)
- 97530: Therapeutic Activities (balance, coordination, functional tasks)
- 97112: Neuromuscular Re-education (movement patterns, posture, proprioception)
- 97542: Wheelchair Assessment/Training
- 97750: Physical Performance Test (TUG, Berg, 6MWT)"""

COMMON_CODES_OT = """OT CPT Codes:
- 97530: Therapeutic Activities (functional tasks, coordination)
- 97535: Self-Care/Home Management Training (ADL/IADL)
- 97110: Therapeutic Exercise (UE/hand strengthening, ROM)
- 97140: Manual Therapy (soft tissue, joint mobilization)
- 97542: Wheelchair/Mobility Device Training
- 97112: Neuromuscular Re-education (coordination, motor planning)"""

COMMON_CODES_SLP = """SLP CPT Codes:
- 92507: Treatment of Speech/Language/Communication
- 92526: Treatment of Swallowing Dysfunction
- 92610: Evaluation of Swallowing Function
- 92523: Speech/Language Evaluation
- 97530: Therapeutic Activities (cognitive-linguistic)"""


# ============================================================================
# Setting-Specific Addendum
# ============================================================================

SETTING_ADDENDUM: dict[str, str] = {
    "outpatient": "",  # standard clinic — no addendum needed

    "homecare": """
## Home Health Additional Requirements:
- Reference OASIS data elements where applicable (functional items, GG scores)
- Document home safety observations and environmental barriers
- Include caregiver education, training, and competency demonstration
- Note functional environment (stairs, bathroom access, bedroom location)
- Document homebound status justification
- Note patient/caregiver willingness and ability to participate in HEP
- Reference 60-day certification period and visit utilization""",

    "snf": """
## SNF Additional Requirements:
- Include FIM scores or Section GG functional items where applicable
- Document discharge planning and projected discharge destination
- Reference interdisciplinary team goals and care conferences
- Note MDS assessment alignment
- Document minutes per discipline for PDPM compliance
- Include restorative nursing referral if appropriate
- Note coordination with nursing, MD, social work""",
}


# ============================================================================
# SOAP Generation Prompt
# ============================================================================

SOAP_GENERATION_PROMPT = """You are a skilled clinical documentation specialist for {discipline} therapists.
Generate a {payer_type}-compliant SOAP note from the therapist's raw input.

## Context:
- Date of Service: {date_of_service}
- Discipline: {discipline}
- Payer: {payer_type}
- Setting: {setting}

## Payer-Specific Guidelines
{payer_guidance}
{setting_addendum}

## Raw Input from Therapist:
\"\"\"{raw_soap_text}\"\"\"

## Time Data:
- Total visit minutes: {total_minutes}
- Timed CPT minutes: {timed_minutes}

## Interventions Mentioned:
{interventions}

## Section Requirements:

### SUBJECTIVE (S):
- Patient-reported symptoms, pain (0-10 scale), concerns
- Functional complaints (difficulty with specific activities)
- Changes since last visit
- NO clinician interpretation — ONLY patient/caregiver reports
- 2-4 sentences

### OBJECTIVE (O):
- Quantifiable measures only (ROM, strength grades, balance scores)
- Test results with numbers (TUG time, 5xSTS count, Berg score)
- Skilled interventions with WHAT/WHY/HOW format:
  "Therapist provided [intervention] with [cueing type] to facilitate \
[functional outcome], addressing [deficit/safety risk]"
- Use bullet points for clarity
- NO interpretation (facts only)

### ASSESSMENT (A):
**FOR DAILY NOTES:** Brief 2-3 sentences — patient response, notable \
progress or concerns, simple continued-benefit statement.
**FOR PROGRESS NOTES/EVALS:** 3-5 sentences with measurable progress, \
medical necessity, barriers, prognosis.

### PLAN (P):
- Frequency and duration
- Specific focus for next session(s) linked to functional goals
- Progressions planned with clinical rationale
- 2-4 sentences

## Billing Guidelines:
8-minute rule for timed codes: 8-22=1u, 23-37=2u, 38-52=3u, 53-67=4u, 68-82=5u

Common {discipline} codes:
{common_codes}

## Quality Control:
- NEVER invent patient data, scores, or subjective statements
- If data missing, use placeholder: "[Insert data if available]"
- Avoid contradictions between sections
- Ensure defensibility under payer/chart review

## PROHIBITED PHRASES (NEVER use):
"tolerated treatment well", "continues to benefit" (without metrics), \
"patient did exercises", "good session", "stable" (without context)

## Output Format:
Return JSON:
{{
  "sections": {{
    "subjective": "...",
    "objective": "...",
    "assessment": "...",
    "plan": "..."
  }},
  "billing": {{
    "codes": [{{"cpt": "97110", "units": 2, "minutes": 30, "description": "..."}}],
    "total_timed_minutes": {timed_minutes},
    "validation_status": "valid",
    "warnings": []
  }},
  "rendered_soap_text": "S: ...\\n\\nO: ...\\n\\nA: ...\\n\\nP: ..."
}}

Generate professional, skilled, defensible documentation now:"""


# ============================================================================
# Section Revision Prompt
# ============================================================================

SECTION_REVISION_PROMPT = """You are a clinical documentation quality specialist.
Revise ONLY the {section_upper} section of a {payer_type} {discipline} note.

## Current {section_upper} Text:
\"\"\"{current_text}\"\"\"

## Revision Instruction:
{instruction}

## Revision Standards:

### For "Make shorter":
- Condense without removing skilled justification or compliance elements
- Preserve medical necessity language and functional outcomes

### For "Make more skilled":
- Replace generic phrasing with skilled rationale (WHAT/WHY/HOW)
- Replace prohibited phrases with skilled alternatives

### For "Make more objective":
- Add measurable data and specific values
- Remove vague wording; include functional metrics

## PROHIBITED PHRASES (NEVER use):
"Tolerated treatment well", "Continues to benefit" (without metrics), \
"Patient did exercises", "Good session"

## Requirements:
- Preserve clinically important content
- Maintain {payer_type}-compliant language
- Ensure defensibility under audit

Return ONLY the revised {section_upper} text. No other sections or explanations."""


# ============================================================================
# QA Review Prompt
# ============================================================================

QA_REVIEW_PROMPT = """You are a clinical documentation quality auditor.
Analyze this {discipline} SOAP note for skilled documentation, medical \
necessity, and {payer_type} compliance.

## Full Note:
S: {subjective}
O: {objective}
A: {assessment}
P: {plan}

## Billing: {billing_summary}

## Review Criteria:

### 1. Skilled Documentation Quality
Check each intervention for WHAT/WHY/HOW. Flag prohibited phrases.

### 2. Medical Necessity
Is skilled care justified? Are functional limitations documented? Does \
each treatment connect to limitations and goals?

### 3. Section Compliance
- Subjective: ONLY patient/caregiver reports
- Objective: quantifiable, facts only, skilled interventions documented
- Assessment: clinical reasoning, progress with metrics, medical necessity
- Plan: specific actions, frequency, duration, focus

### 4. Billing Support
Do interventions support CPT codes? Is time reasonable? Are units \
appropriate per 8-minute rule?

## Output JSON:
{{
  "overall_score": "good | needs_attention | critical",
  "skilled_documentation": {{
    "is_skilled": true/false,
    "prohibited_phrases_found": [],
    "interventions_lacking_what_why_how": []
  }},
  "medical_necessity": {{
    "is_justified": true/false,
    "concerns": []
  }},
  "section_compliance": {{
    "subjective": {{"compliant": true/false, "issues": []}},
    "objective": {{"compliant": true/false, "issues": []}},
    "assessment": {{"compliant": true/false, "issues": []}},
    "plan": {{"compliant": true/false, "issues": []}}
  }},
  "billing_support": {{
    "codes_supported": true/false,
    "concerns": []
  }},
  "recommendations": []
}}"""


# ============================================================================
# Diagnosis-Specific Templates
# ============================================================================

DIAGNOSIS_TEMPLATES: dict[str, dict] = {
    "stroke": {
        "name": "CVA/Stroke",
        "icd10_patterns": ["I63", "I64", "I69"],
        "common_deficits": [
            "Hemiparesis/hemiplegia", "Balance impairment",
            "Gait deviation", "Decreased activity tolerance",
        ],
        "typical_interventions": [
            "97112 Neuromuscular re-education", "97116 Gait training",
            "97110 Therapeutic exercise", "97530 Balance training",
        ],
        "skilled_language": [
            "Therapist facilitated weight shift to affected side using "
            "tactile and verbal cueing",
            "Skilled neuromuscular re-education to address impaired motor "
            "planning",
        ],
    },
    "parkinsons": {
        "name": "Parkinson's Disease",
        "icd10_patterns": ["G20", "G21"],
        "common_deficits": [
            "Bradykinesia", "Rigidity", "Postural instability",
            "Freezing of gait",
        ],
        "typical_interventions": [
            "97116 Gait training with cueing strategies",
            "97530 Balance training", "97110 Therapeutic exercise",
        ],
        "skilled_language": [
            "Rhythmic auditory cueing provided to address freezing of gait",
            "Visual targets placed on floor to normalize step length",
        ],
    },
    "hip_fracture": {
        "name": "Hip Fracture/ORIF",
        "icd10_patterns": ["S72", "Z96.64"],
        "common_deficits": [
            "Pain with movement", "Decreased ROM", "Weakness",
            "Weight bearing restrictions",
        ],
        "typical_interventions": [
            "97116 Gait training with WB precautions",
            "97110 Therapeutic exercise", "97530 Transfer training",
        ],
        "skilled_language": [
            "Weight bearing status monitored and reinforced",
            "Gait pattern correction provided to prevent compensatory patterns",
        ],
    },
    "tka": {
        "name": "Total Knee Arthroplasty",
        "icd10_patterns": ["Z96.65", "M17"],
        "common_deficits": [
            "Pain and swelling", "Decreased ROM",
            "Quadriceps weakness", "Gait deviation",
        ],
        "typical_interventions": [
            "97110 ROM/strengthening exercises", "97116 Gait training",
            "97140 Manual therapy for ROM",
        ],
        "skilled_language": [
            "PROM/AAROM to address post-operative ROM limitations",
            "Gait training emphasizing terminal knee extension in stance",
        ],
    },
    "debility": {
        "name": "Generalized Weakness/Debility",
        "icd10_patterns": ["R53", "R54", "Z74"],
        "common_deficits": [
            "Generalized weakness", "Decreased activity tolerance",
            "Balance impairment", "Fall risk",
        ],
        "typical_interventions": [
            "97110 Therapeutic exercise", "97116 Gait training",
            "97530 Functional mobility training",
        ],
        "skilled_language": [
            "Progressive resistance exercise to address deconditioning",
            "Activity tolerance monitored with vital signs and perceived exertion",
        ],
    },
    "fall_risk": {
        "name": "Fall Risk/History of Falls",
        "icd10_patterns": ["R29.6", "W19", "Z91.81"],
        "common_deficits": [
            "Balance impairment", "Gait instability",
            "Lower extremity weakness", "Fear of falling",
        ],
        "typical_interventions": [
            "97530 Balance training", "97116 Gait training",
            "97110 Therapeutic exercise",
        ],
        "skilled_language": [
            "Multi-factorial fall risk assessment performed",
            "Perturbation training to improve automatic balance reactions",
        ],
    },
}


# ============================================================================
# Intervention Templates (CPT-Specific Documentation Guidance)
# ============================================================================

INTERVENTION_TEMPLATES: dict[str, dict] = {
    "97110": {
        "name": "Therapeutic Exercise",
        "skilled_documentation": [
            "Therapist instructed patient in {exercise} with {cueing_type} "
            "cueing to address {deficit}",
            "Progressive resistance exercise targeting {muscle_group} "
            "weakness limiting {function}",
        ],
    },
    "97116": {
        "name": "Gait Training",
        "skilled_documentation": [
            "Gait training with {device} targeting {gait_deviation} "
            "limiting safe {environment} ambulation",
            "Therapist facilitated {correction} through {cueing_type} "
            "cueing during {phase} phase",
        ],
    },
    "97112": {
        "name": "Neuromuscular Re-education",
        "skilled_documentation": [
            "Neuromuscular re-education to address {deficit} limiting "
            "{function}",
            "Weight shift training with {feedback_type} feedback for "
            "limits of stability",
        ],
    },
    "97530": {
        "name": "Therapeutic Activities",
        "skilled_documentation": [
            "Therapeutic activities to improve {function} for {adl_goal}",
            "Dynamic balance activities during {task} with graded challenge",
        ],
    },
    "97140": {
        "name": "Manual Therapy",
        "skilled_documentation": [
            "Soft tissue mobilization to {muscle} addressing restriction "
            "limiting {function}",
            "Joint mobilization Grade {grade} to {joint} in {direction} "
            "to improve {rom}",
        ],
    },
    "97535": {
        "name": "Self-Care/Home Management Training",
        "skilled_documentation": [
            "ADL training for {task} with instruction in {technique}",
            "Adaptive equipment training ({equipment}) for {task}",
        ],
    },
}


# ============================================================================
# Public API
# ============================================================================

def get_discipline_codes(discipline: str) -> str:
    """Get appropriate CPT code reference for discipline."""
    codes_map = {"PT": COMMON_CODES_PT, "OT": COMMON_CODES_OT, "SLP": COMMON_CODES_SLP}
    return codes_map.get(discipline.upper(), COMMON_CODES_PT)


def get_soap_system_prompt(
    discipline: str = "PT",
    payer: str = "Medicare",
    note_type: str = "daily_note",
    setting: str = "outpatient",
) -> str:
    """Return the system-level SOAP generation prompt.

    This is the primary entry point for generating SOAP prompts.

    Args:
        discipline: PT, OT, or SLP.
        payer: Medicare, HMO, Commercial, or Medicaid.
        note_type: daily_note, progress_note, initial_evaluation, etc.
        setting: outpatient, homecare, or snf.

    Returns:
        A system prompt string suitable for LLM system message.
    """
    payer_guidance = PAYER_GUIDANCE.get(payer, PAYER_GUIDANCE["Commercial"])
    common_codes = get_discipline_codes(discipline)
    setting_addendum = SETTING_ADDENDUM.get(setting, "")

    note_context = ""
    if note_type in ("initial_evaluation", "evaluation"):
        note_context = (
            "\n\nThis is an EVALUATION note. Include: evaluation findings, "
            "functional baselines, STG/LTG goals, and plan of care."
        )
    elif note_type == "progress_note":
        note_context = (
            "\n\nThis is a PROGRESS NOTE. Include: measurable progress "
            "toward goals, updated medical necessity, prognosis."
        )

    return (
        f"You are a skilled {discipline} clinical documentation specialist "
        f"producing {payer}-compliant SOAP notes for {setting} settings.\n\n"
        f"## Payer Guidelines\n{payer_guidance}\n"
        f"{setting_addendum}\n"
        f"{note_context}\n\n"
        f"## Common {discipline} CPT Codes\n{common_codes}\n\n"
        "## PROHIBITED PHRASES (NEVER use):\n"
        '"tolerated treatment well", "continues to benefit" (without metrics), '
        '"patient did exercises", "good session", "stable" (without context)\n\n'
        "## Intervention Format: WHAT / WHY / HOW\n"
        '"Therapist provided [intervention] with [cueing type] to facilitate '
        '[functional outcome], addressing [deficit/safety risk]"'
    )


def build_generation_prompt(
    discipline: str,
    date_of_service: str,
    payer_type: str,
    raw_soap_text: str,
    total_minutes: int,
    timed_minutes: int,
    interventions: Optional[list[str]] = None,
    setting: str = "outpatient",
    note_type: str = "daily_note",
) -> str:
    """Build complete generation prompt with all variables filled.

    Args:
        discipline: PT, OT, or SLP.
        date_of_service: Date string.
        payer_type: Medicare, HMO, Commercial, Medicaid.
        raw_soap_text: Therapist's raw input.
        total_minutes: Total visit minutes.
        timed_minutes: Timed CPT minutes.
        interventions: List of intervention names.
        setting: outpatient, homecare, or snf.
        note_type: daily_note, progress_note, etc.

    Returns:
        Fully formatted SOAP generation prompt.
    """
    payer_guidance = PAYER_GUIDANCE.get(payer_type, PAYER_GUIDANCE["Commercial"])
    common_codes = get_discipline_codes(discipline)
    setting_addendum = SETTING_ADDENDUM.get(setting, "")
    interventions_text = ", ".join(interventions) if interventions else "Extract from raw input"

    return SOAP_GENERATION_PROMPT.format(
        discipline=discipline,
        date_of_service=date_of_service,
        payer_type=payer_type,
        payer_guidance=payer_guidance,
        setting=setting,
        setting_addendum=setting_addendum,
        raw_soap_text=raw_soap_text,
        total_minutes=total_minutes,
        timed_minutes=timed_minutes,
        interventions=interventions_text,
        common_codes=common_codes,
    )


def build_revision_prompt(
    section: str,
    current_text: str,
    instruction: str,
    payer_type: str = "Medicare",
    discipline: str = "PT",
) -> str:
    """Build section revision prompt."""
    return SECTION_REVISION_PROMPT.format(
        section_upper=section.upper(),
        payer_type=payer_type,
        discipline=discipline,
        current_text=current_text,
        instruction=instruction,
    )


def build_qa_prompt(
    subjective: str,
    objective: str,
    assessment: str,
    plan: str,
    billing_summary: str,
    discipline: str = "PT",
    payer_type: str = "Medicare",
) -> str:
    """Build QA review prompt."""
    return QA_REVIEW_PROMPT.format(
        discipline=discipline,
        payer_type=payer_type,
        subjective=subjective,
        objective=objective,
        assessment=assessment,
        plan=plan,
        billing_summary=billing_summary,
    )


def check_prohibited_phrases(text: str) -> list[str]:
    """Check text for prohibited phrases and return any found."""
    text_lower = text.lower()
    return [phrase for phrase in PROHIBITED_PHRASES if phrase.lower() in text_lower]


def get_skilled_replacement(prohibited_phrase: str) -> str:
    """Get suggested replacement for a prohibited phrase."""
    return SKILLED_REPLACEMENTS.get(
        prohibited_phrase,
        "Replace with skilled language including WHAT/WHY/HOW",
    )


def get_diagnosis_template(diagnosis_text: str) -> Optional[dict]:
    """Match diagnosis text to a diagnosis template.

    Args:
        diagnosis_text: Free text diagnosis or ICD-10 code.

    Returns:
        Matching template dict or None.
    """
    diagnosis_lower = diagnosis_text.lower()

    keyword_map = {
        "stroke": ["stroke", "cva", "cerebrovascular", "hemiparesis"],
        "parkinsons": ["parkinson", "pd", "bradykinesia", "festinating"],
        "hip_fracture": ["hip fracture", "hip fx", "orif hip", "femoral fracture"],
        "tka": ["knee replacement", "tka", "total knee", "knee arthroplasty"],
        "debility": ["debility", "weakness", "deconditioning"],
        "fall_risk": ["fall", "balance", "unsteady", "fall risk"],
    }

    for template_key, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in diagnosis_lower:
                return DIAGNOSIS_TEMPLATES.get(template_key)

    # ICD-10 pattern match
    for template in DIAGNOSIS_TEMPLATES.values():
        for pattern in template.get("icd10_patterns", []):
            if pattern.lower() in diagnosis_lower:
                return template

    return None


def get_intervention_template(cpt_code: str) -> Optional[dict]:
    """Get intervention documentation template for a CPT code."""
    return INTERVENTION_TEMPLATES.get(cpt_code)
