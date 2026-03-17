"""OASIS-E assessment module for homecare settings.

Implements the CMS Outcome and Assessment Information Set (OASIS-E)
used for home health quality reporting, payment determination (PDGM),
and outcome measurement. Covers patient tracking, clinical record,
and functional status sections with skip logic and validation.
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OASISAssessmentType(str, Enum):
    SOC = "start_of_care"
    ROC = "resumption_of_care"
    RECERT = "recertification"
    FOLLOWUP = "follow_up"
    DISCHARGE = "discharge"
    TRANSFER = "transfer"
    DEATH = "death"


# Assessment types that require the full item set
_FULL_ASSESSMENT_TYPES = {
    OASISAssessmentType.SOC,
    OASISAssessmentType.ROC,
    OASISAssessmentType.RECERT,
}

# Assessment types with reduced item requirements
_DISCHARGE_TYPES = {
    OASISAssessmentType.DISCHARGE,
    OASISAssessmentType.TRANSFER,
    OASISAssessmentType.DEATH,
}


@dataclass
class OASISItem:
    """Single OASIS-E data item."""
    item_id: str
    section: str
    label: str
    description: str
    response_type: str  # "numeric", "categorical", "date", "text", "icd10"
    options: list[dict] | None = None
    required: bool = True
    skip_logic: str | None = None  # Condition description for when item applies


# ---------------------------------------------------------------------------
# Core OASIS-E Items (key clinical subset — full OASIS has 176+ items)
# ---------------------------------------------------------------------------

OASIS_ITEMS: dict[str, OASISItem] = {
    # === Patient Tracking (M0010–M0100) ===
    "M0010": OASISItem("M0010", "patient_tracking", "CMS Certification Number",
                        "Agency Medicare provider number", "text"),
    "M0014": OASISItem("M0014", "patient_tracking", "Branch State",
                        "State where branch providing services is located", "text"),
    "M0016": OASISItem("M0016", "patient_tracking", "Branch ID Number",
                        "CMS branch identification number", "text"),
    "M0018": OASISItem("M0018", "patient_tracking", "National Provider Identifier",
                        "Provider NPI number", "text"),
    "M0020": OASISItem("M0020", "patient_tracking", "Patient ID Number",
                        "Agency-assigned patient identifier", "text"),
    "M0030": OASISItem("M0030", "patient_tracking", "Start of Care Date",
                        "Date of first billable visit", "date"),
    "M0040": OASISItem("M0040", "patient_tracking", "Patient Name",
                        "Patient legal name", "text"),
    "M0060": OASISItem("M0060", "patient_tracking", "Patient Address",
                        "Patient home address", "text"),
    "M0063": OASISItem("M0063", "patient_tracking", "Medicare Number",
                        "Medicare Beneficiary Identifier", "text"),
    "M0065": OASISItem("M0065", "patient_tracking", "Medicaid Number",
                        "Medicaid number if applicable", "text", required=False),
    "M0069": OASISItem("M0069", "patient_tracking", "Patient Gender",
                        "Patient gender", "categorical",
                        [{"value": 1, "label": "Male"}, {"value": 2, "label": "Female"}]),
    "M0080": OASISItem("M0080", "patient_tracking", "Discipline of Person Completing Assessment",
                        "Discipline of clinician", "categorical",
                        [{"value": 1, "label": "RN"}, {"value": 2, "label": "PT"},
                         {"value": 3, "label": "SLP/ST"}, {"value": 4, "label": "OT"}]),
    "M0100": OASISItem("M0100", "patient_tracking", "Reason for Assessment",
                        "Purpose of this assessment", "categorical",
                        [{"value": 1, "label": "Start of care"}, {"value": 3, "label": "Resumption of care"},
                         {"value": 4, "label": "Recertification"}, {"value": 6, "label": "Transfer to inpatient"},
                         {"value": 7, "label": "Transfer not to inpatient"}, {"value": 8, "label": "Death at home"},
                         {"value": 9, "label": "Discharge"}]),
    "M0110": OASISItem("M0110", "patient_tracking", "Episode Timing",
                        "Is this assessment an early or late episode?", "categorical",
                        [{"value": 1, "label": "Early"}, {"value": 2, "label": "Late"},
                         {"value": "UK", "label": "Unknown"}]),

    # === Clinical Record (M1021–M1311) ===
    "M1021": OASISItem("M1021", "clinical_record", "Primary Diagnosis",
                        "Primary ICD-10-CM diagnosis", "icd10"),
    "M1023": OASISItem("M1023", "clinical_record", "Other Diagnoses",
                        "Additional ICD-10-CM diagnoses (up to 5)", "icd10", required=False),
    "M1030": OASISItem("M1030", "clinical_record", "Therapies at Home",
                        "IV/Infusion or parenteral/enteral nutrition therapy", "categorical",
                        [{"value": 0, "label": "None"}, {"value": 1, "label": "IV/Infusion"},
                         {"value": 2, "label": "Parenteral"}, {"value": 3, "label": "Enteral"}],
                        required=False),
    "M1033": OASISItem("M1033", "clinical_record", "Risk for Hospitalization",
                        "Indicators for hospitalization risk", "categorical",
                        [{"value": 1, "label": "History of falls (2+ in past 12 months)"},
                         {"value": 2, "label": "Unintentional weight loss"},
                         {"value": 3, "label": "Multiple hospitalizations (2+ in past 12 months)"},
                         {"value": 4, "label": "Multiple ED visits (2+ in past 6 months)"},
                         {"value": 5, "label": "Decline in mental/emotional/behavioral status"},
                         {"value": 6, "label": "Reported/observed difficulty with compliance"},
                         {"value": 7, "label": "Currently taking 5+ medications"},
                         {"value": 8, "label": "Currently reports exhaustion"},
                         {"value": 9, "label": "Other risk(s)"},
                         {"value": 0, "label": "None of the above"}]),
    "M1060": OASISItem("M1060", "clinical_record", "Height and Weight",
                        "Patient height (inches) and weight (pounds)", "text"),
    "M1242": OASISItem("M1242", "clinical_record", "Frequency of Pain Interfering with Activity",
                        "How often does pain interfere with patient activity or movement?", "categorical",
                        [{"value": 0, "label": "No pain"},
                         {"value": 1, "label": "Less often than daily"},
                         {"value": 2, "label": "Daily, but not constantly"},
                         {"value": 3, "label": "All of the time"}]),
    "M1311": OASISItem("M1311", "clinical_record", "Current Number of Unhealed Pressure Ulcers/Injuries",
                        "Number of unhealed pressure ulcers at each stage", "numeric"),
    "M1322": OASISItem("M1322", "clinical_record", "Current Number of Stasis Ulcers",
                        "Stasis ulcer count", "categorical",
                        [{"value": 0, "label": "None"}, {"value": 1, "label": "One"},
                         {"value": 2, "label": "Two"}, {"value": 3, "label": "Three"},
                         {"value": 4, "label": "Four or more"}]),
    "M1324": OASISItem("M1324", "clinical_record", "Stage of Most Problematic Stasis Ulcer",
                        "Stage of most problematic observable stasis ulcer", "categorical",
                        [{"value": 1, "label": "Fully granulating"},
                         {"value": 2, "label": "Early/partial granulation"},
                         {"value": 3, "label": "Not healing"}],
                        required=False, skip_logic="Only if M1322 > 0"),
    "M1340": OASISItem("M1340", "clinical_record", "Surgical Wound Present",
                        "Does this patient have a surgical wound?", "categorical",
                        [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes, observable"},
                         {"value": 2, "label": "Yes, not observable (non-removable dressing)"}]),
    "M1400": OASISItem("M1400", "clinical_record", "Dyspnea",
                        "When is the patient dyspneic or short of breath?", "categorical",
                        [{"value": 0, "label": "Never"},
                         {"value": 1, "label": "With moderate exertion"},
                         {"value": 2, "label": "With minimal exertion"},
                         {"value": 3, "label": "At rest"},
                         {"value": 4, "label": "Patient is non-responsive"}]),

    # === Functional Status (M1800–M1910) ===
    "M1800": OASISItem("M1800", "functional_status", "Grooming",
                        "Current ability to tend to personal hygiene needs", "categorical",
                        [{"value": 0, "label": "Able to groom self unaided"},
                         {"value": 1, "label": "Grooming utensils must be placed within reach"},
                         {"value": 2, "label": "Someone must assist the patient"},
                         {"value": 3, "label": "Patient depends entirely upon someone else"}]),
    "M1810": OASISItem("M1810", "functional_status", "Dressing Upper Body",
                        "Ability to dress upper body", "categorical",
                        [{"value": 0, "label": "Able to get clothes and dress upper body independently"},
                         {"value": 1, "label": "Able if clothing is laid out or handed to patient"},
                         {"value": 2, "label": "Someone must help put on clothing"},
                         {"value": 3, "label": "Patient depends entirely upon another person"}]),
    "M1820": OASISItem("M1820", "functional_status", "Dressing Lower Body",
                        "Ability to dress lower body", "categorical",
                        [{"value": 0, "label": "Able to independently dress lower body"},
                         {"value": 1, "label": "Able if clothing is laid out or handed to patient"},
                         {"value": 2, "label": "Someone must help put on clothing"},
                         {"value": 3, "label": "Patient depends entirely upon another person"}]),
    "M1830": OASISItem("M1830", "functional_status", "Bathing",
                        "Ability to wash entire body safely", "categorical",
                        [{"value": 0, "label": "Able to bathe independently"},
                         {"value": 1, "label": "With the use of devices, able to bathe independently"},
                         {"value": 2, "label": "Able to bathe with intermittent assistance"},
                         {"value": 3, "label": "Requires presence of another person throughout"},
                         {"value": 4, "label": "Unable to use shower/tub, bathed in bed or chair"},
                         {"value": 5, "label": "Unable to effectively participate, total bathing by another"}]),
    "M1840": OASISItem("M1840", "functional_status", "Toilet Transferring",
                        "Ability to get to and from the toilet or bedside commode", "categorical",
                        [{"value": 0, "label": "Able to get to and from toilet independently"},
                         {"value": 1, "label": "When reminded, can get to toilet independently"},
                         {"value": 2, "label": "Someone must assist to get to/from toilet"},
                         {"value": 3, "label": "Unable to get to toilet but can use bedside commode"},
                         {"value": 4, "label": "Unable to get to toilet or commode, uses bedpan/urinal"}]),
    "M1850": OASISItem("M1850", "functional_status", "Transferring",
                        "Ability to move between surfaces", "categorical",
                        [{"value": 0, "label": "Able to independently transfer"},
                         {"value": 1, "label": "Transfers with minimal human assistance or device"},
                         {"value": 2, "label": "Unable to transfer without maximal assistance"},
                         {"value": 3, "label": "Bedfast, unable to transfer"}]),
    "M1860": OASISItem("M1860", "functional_status", "Ambulation/Locomotion",
                        "Ability to safely walk or use wheelchair", "categorical",
                        [{"value": 0, "label": "Able to independently walk on all surfaces"},
                         {"value": 1, "label": "With one-person assist, able to walk on even surfaces"},
                         {"value": 2, "label": "Requires manual wheelchair"},
                         {"value": 3, "label": "Requires motorized wheelchair"},
                         {"value": 4, "label": "Chairfast, unable to ambulate or wheel self"},
                         {"value": 5, "label": "Bedfast, unable to ambulate or be up in chair"}]),
    "M1870": OASISItem("M1870", "functional_status", "Feeding or Eating",
                        "Ability to feed self meals and snacks", "categorical",
                        [{"value": 0, "label": "Able to independently feed self"},
                         {"value": 1, "label": "Able to feed self independently with setup"},
                         {"value": 2, "label": "Requires intermittent assistance"},
                         {"value": 3, "label": "Unable to feed self, must be assisted throughout"},
                         {"value": 4, "label": "Able to take in nutrients orally with IV/TPN"},
                         {"value": 5, "label": "Unable to take in nutrients orally"}]),
    "M1910": OASISItem("M1910", "functional_status", "Fall Risk Assessment",
                        "Has this patient had a multi-factor fall risk assessment?", "categorical",
                        [{"value": 0, "label": "No multi-factor assessment conducted"},
                         {"value": 1, "label": "Yes, and it identified no risk"},
                         {"value": 2, "label": "Yes, and it identified risk"}]),

    # === OASIS-E Additions — GG Items (Section GG: Functional Abilities) ===
    "GG0130": OASISItem("GG0130", "section_gg", "Self-Care",
                        "OASIS-E self-care functional abilities (eating, oral hygiene, "
                        "toileting hygiene, shower/bathe self, upper/lower body dressing, "
                        "putting on/taking off footwear)",
                        "categorical",
                        [{"value": 6, "label": "Independent"},
                         {"value": 5, "label": "Setup or clean-up assistance"},
                         {"value": 4, "label": "Supervision or touching assistance"},
                         {"value": 3, "label": "Partial/moderate assistance"},
                         {"value": 2, "label": "Substantial/maximal assistance"},
                         {"value": 1, "label": "Dependent"},
                         {"value": 7, "label": "Patient refused"},
                         {"value": 9, "label": "Not applicable"},
                         {"value": 88, "label": "Not attempted — safety concerns"},
                         {"value": 10, "label": "Not attempted — not applicable"}]),
    "GG0170": OASISItem("GG0170", "section_gg", "Mobility",
                        "OASIS-E mobility functional abilities (roll, sit to lying, "
                        "lying to sitting, sit to stand, chair/bed-to-chair transfer, "
                        "toilet transfer, walk 10 feet, walk 50 feet, walk 150 feet, "
                        "walking 10 feet on uneven surfaces, 1 step, 4 steps, "
                        "12 steps, picking up object, car transfer, wheel 50 feet, "
                        "wheel 150 feet)",
                        "categorical",
                        [{"value": 6, "label": "Independent"},
                         {"value": 5, "label": "Setup or clean-up assistance"},
                         {"value": 4, "label": "Supervision or touching assistance"},
                         {"value": 3, "label": "Partial/moderate assistance"},
                         {"value": 2, "label": "Substantial/maximal assistance"},
                         {"value": 1, "label": "Dependent"},
                         {"value": 7, "label": "Patient refused"},
                         {"value": 9, "label": "Not applicable"},
                         {"value": 88, "label": "Not attempted — safety concerns"},
                         {"value": 10, "label": "Not attempted — not applicable"}]),
}

# Which items are only required for certain assessment types
_SOC_ROC_ONLY_ITEMS = {"M0030", "M0110"}
_DISCHARGE_SKIP_ITEMS = {"M1030", "M1060", "M1311", "M1322", "M1324", "M1340"}


class OASISValidationError:
    """Single validation issue."""
    def __init__(self, item_id: str, message: str, severity: str = "error"):
        self.item_id = item_id
        self.message = message
        self.severity = severity  # "error" or "warning"

    def to_dict(self) -> dict:
        return {"item_id": self.item_id, "message": self.message, "severity": self.severity}


class OASISAssessment:
    """Complete OASIS-E assessment instance."""

    def __init__(self, assessment_type: OASISAssessmentType, patient_id: str,
                 assessment_id: str | None = None):
        self.assessment_id = assessment_id or str(uuid.uuid4())
        self.assessment_type = assessment_type
        self.patient_id = patient_id
        self.responses: dict[str, Any] = {}
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.submitted = False

    def set_response(self, item_id: str, value: Any) -> None:
        """Set a response for an OASIS item."""
        if item_id not in OASIS_ITEMS:
            raise ValueError(f"Unknown OASIS item: {item_id}")
        item = OASIS_ITEMS[item_id]
        if item.options and item.response_type == "categorical":
            valid_values = {opt["value"] for opt in item.options}
            if value not in valid_values:
                raise ValueError(
                    f"Invalid value {value!r} for {item_id}. "
                    f"Valid: {sorted(str(v) for v in valid_values)}"
                )
        self.responses[item_id] = value
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def get_response(self, item_id: str) -> Any:
        """Get a response for an OASIS item, or None."""
        return self.responses.get(item_id)

    def get_required_items(self) -> list[OASISItem]:
        """Return items required for this assessment type, applying skip logic."""
        required = []
        for item_id, item in OASIS_ITEMS.items():
            if not item.required:
                continue
            # Skip SOC/ROC-only items for non-SOC/ROC assessments
            if item_id in _SOC_ROC_ONLY_ITEMS:
                if self.assessment_type not in (OASISAssessmentType.SOC, OASISAssessmentType.ROC):
                    continue
            # Skip clinical items not needed for discharge/transfer/death
            if self.assessment_type in _DISCHARGE_TYPES:
                if item_id in _DISCHARGE_SKIP_ITEMS:
                    continue
            # Evaluate skip logic for conditional items
            if item.skip_logic and not self._evaluate_skip_logic(item):
                continue
            required.append(item)
        return required

    def _evaluate_skip_logic(self, item: OASISItem) -> bool:
        """Evaluate whether a conditional item should be included."""
        if item.item_id == "M1324":
            m1322 = self.responses.get("M1322", 0)
            return m1322 is not None and m1322 > 0
        return True

    def validate(self) -> list[dict]:
        """Validate assessment completeness and data integrity.

        Returns list of validation issues (errors and warnings).
        """
        issues: list[OASISValidationError] = []
        required_items = self.get_required_items()

        for item in required_items:
            if item.item_id not in self.responses:
                issues.append(OASISValidationError(
                    item.item_id,
                    f"Required item {item.item_id} ({item.label}) is missing",
                ))

        # Cross-field validations
        m0100 = self.responses.get("M0100")
        if m0100 is not None:
            type_map = {1: OASISAssessmentType.SOC, 3: OASISAssessmentType.ROC,
                        4: OASISAssessmentType.RECERT, 9: OASISAssessmentType.DISCHARGE,
                        6: OASISAssessmentType.TRANSFER, 8: OASISAssessmentType.DEATH}
            expected_type = type_map.get(m0100)
            if expected_type and expected_type != self.assessment_type:
                issues.append(OASISValidationError(
                    "M0100",
                    f"M0100 value ({m0100}) does not match assessment type ({self.assessment_type.value})",
                    "warning",
                ))

        return [i.to_dict() for i in issues]

    def get_completion_pct(self) -> float:
        """Return percentage of required items that have responses."""
        required = self.get_required_items()
        if not required:
            return 100.0
        answered = sum(1 for item in required if item.item_id in self.responses)
        return round((answered / len(required)) * 100, 1)

    def to_submission_format(self) -> dict:
        """Export assessment in CMS-ready submission format."""
        return {
            "assessment_id": self.assessment_id,
            "patient_id": self.patient_id,
            "assessment_type": self.assessment_type.value,
            "reason_for_assessment": self.responses.get("M0100"),
            "created_at": self.created_at,
            "submitted_at": datetime.now(timezone.utc).isoformat() if self.submitted else None,
            "completion_pct": self.get_completion_pct(),
            "items": {
                item_id: {
                    "value": value,
                    "label": OASIS_ITEMS[item_id].label if item_id in OASIS_ITEMS else item_id,
                }
                for item_id, value in sorted(self.responses.items())
            },
        }

    def to_dict(self) -> dict:
        """Serialize assessment to dict."""
        return {
            "assessment_id": self.assessment_id,
            "patient_id": self.patient_id,
            "assessment_type": self.assessment_type.value,
            "responses": dict(self.responses),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "submitted": self.submitted,
            "completion_pct": self.get_completion_pct(),
        }


class OASISService:
    """OASIS assessment management service.

    In-memory storage. In production, assessments would be persisted
    to PostgreSQL via a dedicated SQLAlchemy model.
    """

    def __init__(self) -> None:
        self._assessments: dict[str, OASISAssessment] = {}

    def create_assessment(self, assessment_type: OASISAssessmentType,
                          patient_id: str) -> OASISAssessment:
        """Create a new OASIS assessment."""
        assessment = OASISAssessment(assessment_type, patient_id)
        self._assessments[assessment.assessment_id] = assessment
        logger.info("Created OASIS %s assessment %s for patient %s",
                     assessment_type.value, assessment.assessment_id, patient_id)
        return assessment

    def get_assessment(self, assessment_id: str) -> OASISAssessment | None:
        """Retrieve an assessment by ID."""
        return self._assessments.get(assessment_id)

    def save_progress(self, assessment_id: str, responses: dict[str, Any]) -> OASISAssessment:
        """Save partial responses for an in-progress assessment."""
        assessment = self._assessments.get(assessment_id)
        if assessment is None:
            raise ValueError(f"Assessment {assessment_id} not found")
        if assessment.submitted:
            raise ValueError(f"Assessment {assessment_id} is already submitted — cannot modify")
        for item_id, value in responses.items():
            assessment.set_response(item_id, value)
        logger.info("Saved progress for assessment %s (%d items)",
                     assessment_id, len(responses))
        return assessment

    def validate_for_submission(self, assessment_id: str) -> list[dict]:
        """Validate an assessment for CMS submission."""
        assessment = self._assessments.get(assessment_id)
        if assessment is None:
            raise ValueError(f"Assessment {assessment_id} not found")
        return assessment.validate()

    async def auto_populate_from_encounter(self, assessment_id: str,
                                           encounter_data: dict) -> dict:
        """Auto-populate OASIS items from encounter/SOAP data.

        Maps structured encounter data to OASIS items where possible.
        Returns a dict of {item_id: value} that were auto-filled.
        """
        assessment = self._assessments.get(assessment_id)
        if assessment is None:
            raise ValueError(f"Assessment {assessment_id} not found")

        auto_filled: dict[str, Any] = {}

        # Map patient demographics
        if "patient_name" in encounter_data:
            assessment.set_response("M0040", encounter_data["patient_name"])
            auto_filled["M0040"] = encounter_data["patient_name"]
        if "patient_address" in encounter_data:
            assessment.set_response("M0060", encounter_data["patient_address"])
            auto_filled["M0060"] = encounter_data["patient_address"]
        if "medicare_number" in encounter_data:
            assessment.set_response("M0063", encounter_data["medicare_number"])
            auto_filled["M0063"] = encounter_data["medicare_number"]
        if "patient_gender" in encounter_data:
            gender_map = {"male": 1, "m": 1, "female": 2, "f": 2}
            val = gender_map.get(str(encounter_data["patient_gender"]).lower())
            if val:
                assessment.set_response("M0069", val)
                auto_filled["M0069"] = val
        if "primary_diagnosis" in encounter_data:
            assessment.set_response("M1021", encounter_data["primary_diagnosis"])
            auto_filled["M1021"] = encounter_data["primary_diagnosis"]
        if "other_diagnoses" in encounter_data:
            assessment.set_response("M1023", encounter_data["other_diagnoses"])
            auto_filled["M1023"] = encounter_data["other_diagnoses"]
        if "discipline" in encounter_data:
            disc_map = {"rn": 1, "pt": 2, "slp": 3, "st": 3, "ot": 4}
            val = disc_map.get(str(encounter_data["discipline"]).lower())
            if val:
                assessment.set_response("M0080", val)
                auto_filled["M0080"] = val

        logger.info("Auto-populated %d items for assessment %s",
                     len(auto_filled), assessment_id)
        return auto_filled

    def get_recert_due_patients(self, patient_assessments: dict[str, list[str]] | None = None
                                ) -> list[dict]:
        """Identify patients due for 60-day recertification.

        Scans existing assessments and flags patients whose last SOC/RECERT
        is approaching or past the 60-day certification period.
        """
        now = datetime.now(timezone.utc)
        due_list: list[dict] = []

        # Group assessments by patient
        patient_latest: dict[str, OASISAssessment] = {}
        for assessment in self._assessments.values():
            if assessment.assessment_type in (OASISAssessmentType.SOC, OASISAssessmentType.RECERT):
                existing = patient_latest.get(assessment.patient_id)
                if existing is None or assessment.created_at > existing.created_at:
                    patient_latest[assessment.patient_id] = assessment

        for patient_id, assessment in patient_latest.items():
            try:
                created = datetime.fromisoformat(assessment.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                days_elapsed = (now - created).days
                days_until_due = 60 - days_elapsed
                if days_until_due <= 14:  # Due within 2 weeks or overdue
                    due_list.append({
                        "patient_id": patient_id,
                        "last_assessment_id": assessment.assessment_id,
                        "last_assessment_type": assessment.assessment_type.value,
                        "last_assessment_date": assessment.created_at,
                        "days_elapsed": days_elapsed,
                        "days_until_due": max(days_until_due, 0),
                        "overdue": days_until_due < 0,
                    })
            except (ValueError, TypeError):
                continue

        due_list.sort(key=lambda x: x["days_until_due"])
        return due_list

    def list_assessments(self, patient_id: str | None = None) -> list[dict]:
        """List assessments, optionally filtered by patient."""
        results = []
        for a in self._assessments.values():
            if patient_id and a.patient_id != patient_id:
                continue
            results.append(a.to_dict())
        results.sort(key=lambda x: x["created_at"], reverse=True)
        return results
