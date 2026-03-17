"""Care transition management for RehabOS.

Handles patient transitions between clinical settings (e.g., SNF to home,
clinic to telehealth). Generates transfer summaries, tracks pending
transitions, and ensures continuity of care across settings.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from rehab_os.clinical.settings import ClinicalSetting

logger = logging.getLogger(__name__)


@dataclass
class CareTransition:
    """Represents a patient care transition between settings."""

    transition_id: str
    patient_id: str
    from_setting: ClinicalSetting
    to_setting: ClinicalSetting
    transition_date: str  # ISO date string
    reason: str  # "discharge_home", "hospital_readmit", "snf_to_home", "clinic_to_telehealth"
    status: str = "pending"  # pending, in_progress, completed, cancelled

    # Provider info
    referring_provider_npi: str = ""
    receiving_provider_npi: str = ""

    # Clinical carry-forward
    clinical_summary: str = ""  # Auto-generated from last encounter
    pending_orders: list[str] = field(default_factory=list)
    active_goals: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    precautions: list[str] = field(default_factory=list)

    # Timestamps
    created_at: str = ""
    completed_at: Optional[str] = None


# Allowed transition paths (from_setting -> [valid to_settings])
VALID_TRANSITIONS: dict[ClinicalSetting, list[ClinicalSetting]] = {
    ClinicalSetting.SNF: [
        ClinicalSetting.HOMECARE, ClinicalSetting.OUTPATIENT,
        ClinicalSetting.IRF, ClinicalSetting.ALF, ClinicalSetting.TELEHEALTH,
    ],
    ClinicalSetting.IRF: [
        ClinicalSetting.SNF, ClinicalSetting.HOMECARE,
        ClinicalSetting.OUTPATIENT, ClinicalSetting.TELEHEALTH,
    ],
    ClinicalSetting.HOMECARE: [
        ClinicalSetting.OUTPATIENT, ClinicalSetting.TELEHEALTH,
        ClinicalSetting.SNF,
    ],
    ClinicalSetting.OUTPATIENT: [
        ClinicalSetting.TELEHEALTH, ClinicalSetting.HOMECARE,
        ClinicalSetting.SNF,
    ],
    ClinicalSetting.ALF: [
        ClinicalSetting.SNF, ClinicalSetting.HOMECARE,
        ClinicalSetting.OUTPATIENT, ClinicalSetting.TELEHEALTH,
    ],
    ClinicalSetting.SCHOOL: [
        ClinicalSetting.OUTPATIENT, ClinicalSetting.TELEHEALTH,
    ],
    ClinicalSetting.TELEHEALTH: [
        ClinicalSetting.OUTPATIENT, ClinicalSetting.HOMECARE,
    ],
}


class TransitionManager:
    """Manage care transitions between clinical settings.

    In-memory storage for now. Production should use CareTransitionDB
    from rehab_os.core.models.
    """

    def __init__(self) -> None:
        self._transitions: dict[str, CareTransition] = {}

    def initiate_transition(
        self,
        patient_id: str,
        from_setting: ClinicalSetting,
        to_setting: ClinicalSetting,
        reason: str,
        referring_provider_npi: str = "",
        receiving_provider_npi: str = "",
        pending_orders: Optional[list[str]] = None,
        active_goals: Optional[list[str]] = None,
        medications: Optional[list[str]] = None,
        precautions: Optional[list[str]] = None,
    ) -> CareTransition:
        """Create a new care transition request.

        Validates the transition path and creates a pending transition.
        """
        # Validate transition path
        valid_targets = VALID_TRANSITIONS.get(from_setting, [])
        if to_setting not in valid_targets:
            raise ValueError(
                f"Invalid transition: {from_setting.value} -> {to_setting.value}. "
                f"Valid targets: {[s.value for s in valid_targets]}"
            )

        now = datetime.now(timezone.utc).isoformat()
        transition = CareTransition(
            transition_id=str(uuid.uuid4()),
            patient_id=patient_id,
            from_setting=from_setting,
            to_setting=to_setting,
            transition_date=now,
            reason=reason,
            status="pending",
            referring_provider_npi=referring_provider_npi,
            receiving_provider_npi=receiving_provider_npi,
            pending_orders=pending_orders or [],
            active_goals=active_goals or [],
            medications=medications or [],
            precautions=precautions or [],
            created_at=now,
        )

        self._transitions[transition.transition_id] = transition
        logger.info(
            "Transition initiated: %s -> %s for patient %s (id=%s)",
            from_setting.value, to_setting.value, patient_id, transition.transition_id,
        )
        return transition

    def generate_transfer_summary(self, patient_id: str) -> str:
        """Generate a clinical transfer summary for a patient.

        In production, this would call the LLM to summarize the patient's
        recent encounters, goals, and progress into a narrative. For now,
        returns a structured placeholder.
        """
        # Collect the most recent transition for this patient
        patient_transitions = [
            t for t in self._transitions.values()
            if t.patient_id == patient_id
        ]
        if not patient_transitions:
            return f"No active transitions for patient {patient_id}."

        latest = max(patient_transitions, key=lambda t: t.created_at)

        parts = [
            f"TRANSFER SUMMARY - Patient: {patient_id}",
            f"From: {latest.from_setting.value} -> To: {latest.to_setting.value}",
            f"Reason: {latest.reason}",
            f"Date: {latest.transition_date}",
        ]
        if latest.active_goals:
            parts.append(f"Active Goals: {'; '.join(latest.active_goals)}")
        if latest.medications:
            parts.append(f"Medications: {'; '.join(latest.medications)}")
        if latest.precautions:
            parts.append(f"Precautions: {'; '.join(latest.precautions)}")
        if latest.pending_orders:
            parts.append(f"Pending Orders: {'; '.join(latest.pending_orders)}")

        # TODO: Replace with LLM-generated narrative summary from
        # recent encounter notes and outcome scores.

        return "\n".join(parts)

    def get_pending_transitions(self, patient_id: Optional[str] = None) -> list[CareTransition]:
        """Get all pending/in-progress transitions, optionally filtered by patient."""
        results = [
            t for t in self._transitions.values()
            if t.status in ("pending", "in_progress")
        ]
        if patient_id:
            results = [t for t in results if t.patient_id == patient_id]
        return sorted(results, key=lambda t: t.created_at, reverse=True)

    def complete_transition(
        self,
        transition_id: str,
        clinical_summary: str = "",
    ) -> CareTransition:
        """Mark a transition as completed."""
        transition = self._transitions.get(transition_id)
        if not transition:
            raise ValueError(f"Transition not found: {transition_id}")
        if transition.status == "completed":
            raise ValueError(f"Transition already completed: {transition_id}")

        transition.status = "completed"
        transition.completed_at = datetime.now(timezone.utc).isoformat()
        if clinical_summary:
            transition.clinical_summary = clinical_summary

        logger.info("Transition completed: %s", transition_id)
        return transition

    def cancel_transition(self, transition_id: str, reason: str = "") -> CareTransition:
        """Cancel a pending transition."""
        transition = self._transitions.get(transition_id)
        if not transition:
            raise ValueError(f"Transition not found: {transition_id}")
        if transition.status == "completed":
            raise ValueError(f"Cannot cancel completed transition: {transition_id}")

        transition.status = "cancelled"
        if reason:
            transition.clinical_summary += f"\nCancellation reason: {reason}"

        logger.info("Transition cancelled: %s (%s)", transition_id, reason)
        return transition

    def get_transition(self, transition_id: str) -> Optional[CareTransition]:
        """Get a specific transition by ID."""
        return self._transitions.get(transition_id)
