"""Longitudinal outcome tracking for RehabOS.

Defines standardized rehab outcome measures with MCID/MDC thresholds
and an in-memory tracker for recording and analyzing patient progress.

TODO: Replace in-memory storage with database-backed persistence
      using OutcomeScoreDB from rehab_os.core.models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class OutcomeMeasure:
    """A standardized outcome measure with MCID/MDC thresholds."""

    name: str           # e.g., "LEFS", "ODI", "Berg"
    scale_min: float
    scale_max: float
    direction: str      # "up" = higher is better, "down" = lower is better
    mcid: float         # Minimum Clinically Important Difference
    mdc: float          # Minimum Detectable Change
    discipline: str     # "PT", "OT", "SLP", "ALL"


# --- 22 common rehab outcome measures ---

OUTCOME_MEASURES: dict[str, OutcomeMeasure] = {
    # PT - Orthopedic
    "LEFS": OutcomeMeasure("Lower Extremity Functional Scale", 0, 80, "up", 9.0, 9.0, "PT"),
    "ODI": OutcomeMeasure("Oswestry Disability Index", 0, 100, "down", 10.0, 10.0, "PT"),
    "NDI": OutcomeMeasure("Neck Disability Index", 0, 50, "down", 7.5, 5.5, "PT"),
    "DASH": OutcomeMeasure("Disabilities of Arm/Shoulder/Hand", 0, 100, "down", 10.2, 12.8, "PT"),
    "SPADI": OutcomeMeasure("Shoulder Pain and Disability Index", 0, 100, "down", 13.2, 18.0, "PT"),
    "KOOS": OutcomeMeasure("Knee Injury and OA Outcome Score", 0, 100, "up", 10.0, 12.0, "PT"),
    # PT - Neuro/Balance
    "Berg": OutcomeMeasure("Berg Balance Scale", 0, 56, "up", 4.0, 5.0, "PT"),
    "TUG": OutcomeMeasure("Timed Up and Go", 0, 60, "down", 3.4, 2.9, "PT"),
    "6MWT": OutcomeMeasure("6-Minute Walk Test", 0, 800, "up", 50.0, 45.0, "PT"),
    "DGI": OutcomeMeasure("Dynamic Gait Index", 0, 24, "up", 2.0, 2.9, "PT"),
    "FGA": OutcomeMeasure("Functional Gait Assessment", 0, 30, "up", 4.0, 4.2, "PT"),
    "ABC": OutcomeMeasure("Activities-Specific Balance Confidence", 0, 100, "up", 13.0, 13.0, "PT"),
    # PT - Pain
    "NPRS": OutcomeMeasure("Numeric Pain Rating Scale", 0, 10, "down", 2.0, 2.0, "PT"),
    "PSFS": OutcomeMeasure("Patient-Specific Functional Scale", 0, 10, "up", 2.0, 2.0, "PT"),
    # OT
    "QuickDASH": OutcomeMeasure("Quick DASH", 0, 100, "down", 8.0, 11.0, "OT"),
    "FIM": OutcomeMeasure("Functional Independence Measure", 18, 126, "up", 22.0, 17.0, "ALL"),
    "Barthel": OutcomeMeasure("Barthel Index", 0, 100, "up", 4.0, 5.0, "ALL"),
    "COPM": OutcomeMeasure("Canadian Occupational Performance Measure", 1, 10, "up", 2.0, 1.9, "OT"),
    # SLP
    "FOIS": OutcomeMeasure("Functional Oral Intake Scale", 1, 7, "up", 1.0, 1.0, "SLP"),
    "ASHA_NOMS": OutcomeMeasure("ASHA NOMS FCM", 1, 7, "up", 1.0, 1.0, "SLP"),
    "VHI": OutcomeMeasure("Voice Handicap Index", 0, 120, "down", 18.0, 20.0, "SLP"),
    # Global
    "GRC": OutcomeMeasure("Global Rating of Change", -7, 7, "up", 2.0, 2.0, "ALL"),
}


@dataclass
class ScoreRecord:
    """A single recorded score for a patient on a measure."""

    patient_id: str
    episode_id: Optional[str]
    measure_name: str
    score: float
    recorded_at: datetime
    recorded_by: Optional[str] = None


# Diagnosis-to-measure recommendations (simplified)
_DIAGNOSIS_MEASURES: dict[str, list[str]] = {
    "low_back_pain": ["NPRS", "ODI", "PSFS", "LEFS"],
    "neck_pain": ["NPRS", "NDI", "PSFS"],
    "shoulder": ["NPRS", "DASH", "SPADI", "PSFS"],
    "knee": ["NPRS", "LEFS", "KOOS", "PSFS"],
    "hip": ["NPRS", "LEFS", "PSFS"],
    "stroke": ["Berg", "FIM", "6MWT", "TUG", "FGA"],
    "tbi": ["Berg", "FIM", "DGI", "FGA"],
    "balance": ["Berg", "TUG", "DGI", "ABC", "FGA"],
    "fall_risk": ["Berg", "TUG", "ABC", "6MWT"],
    "hand_wrist": ["QuickDASH", "NPRS", "PSFS"],
    "dysphagia": ["FOIS", "ASHA_NOMS"],
    "voice": ["VHI", "ASHA_NOMS"],
    "general_deconditioning": ["FIM", "Barthel", "6MWT", "TUG"],
}


class OutcomeTracker:
    """Track patient outcomes longitudinally across episodes.

    Currently in-memory (dict-based). Should be migrated to
    database-backed persistence via OutcomeScoreDB for production.
    """

    def __init__(self) -> None:
        # {patient_id: {measure_name: [ScoreRecord, ...]}}
        self._scores: dict[str, dict[str, list[ScoreRecord]]] = {}

    def record_score(
        self,
        patient_id: str,
        measure_name: str,
        score: float,
        recorded_at: Optional[datetime] = None,
        episode_id: Optional[str] = None,
        recorded_by: Optional[str] = None,
    ) -> ScoreRecord:
        """Record a new outcome score for a patient."""
        measure = OUTCOME_MEASURES.get(measure_name)
        if not measure:
            raise ValueError(f"Unknown measure: {measure_name}. Valid: {list(OUTCOME_MEASURES)}")

        # Clamp to valid range
        clamped = max(measure.scale_min, min(measure.scale_max, score))
        if clamped != score:
            logger.warning(
                "Score %.1f clamped to %.1f for %s (range %.0f-%.0f)",
                score, clamped, measure_name, measure.scale_min, measure.scale_max,
            )

        record = ScoreRecord(
            patient_id=patient_id,
            episode_id=episode_id,
            measure_name=measure_name,
            score=clamped,
            recorded_at=recorded_at or datetime.now(timezone.utc),
            recorded_by=recorded_by,
        )
        self._scores.setdefault(patient_id, {}).setdefault(measure_name, []).append(record)
        # Keep sorted by date
        self._scores[patient_id][measure_name].sort(key=lambda r: r.recorded_at)
        return record

    def get_progress(self, patient_id: str, measure_name: str) -> list[ScoreRecord]:
        """Get all recorded scores for a patient on a given measure."""
        return self._scores.get(patient_id, {}).get(measure_name, [])

    def check_mcid(self, patient_id: str, measure_name: str) -> dict[str, Any]:
        """Check whether a patient has met MCID/MDC for a measure.

        Returns dict with: met_mcid, met_mdc, delta, baseline, latest, direction.
        """
        measure = OUTCOME_MEASURES.get(measure_name)
        if not measure:
            return {"error": f"Unknown measure: {measure_name}"}

        scores = self.get_progress(patient_id, measure_name)
        if len(scores) < 2:
            return {
                "met_mcid": False,
                "met_mdc": False,
                "delta": 0.0,
                "baseline": scores[0].score if scores else None,
                "latest": scores[-1].score if scores else None,
                "direction": measure.direction,
                "scores_recorded": len(scores),
            }

        baseline = scores[0].score
        latest = scores[-1].score

        if measure.direction == "up":
            delta = latest - baseline
        else:
            delta = baseline - latest  # positive delta = improvement for "down" measures

        return {
            "met_mcid": delta >= measure.mcid,
            "met_mdc": delta >= measure.mdc,
            "delta": round(delta, 2),
            "baseline": baseline,
            "latest": latest,
            "mcid_threshold": measure.mcid,
            "mdc_threshold": measure.mdc,
            "direction": measure.direction,
            "scores_recorded": len(scores),
        }

    def get_functional_summary(self, patient_id: str) -> dict[str, dict[str, Any]]:
        """Get summary of all measures for a patient with trend analysis."""
        patient_scores = self._scores.get(patient_id, {})
        summary: dict[str, dict[str, Any]] = {}

        for measure_name, records in patient_scores.items():
            measure = OUTCOME_MEASURES.get(measure_name)
            if not measure or not records:
                continue

            latest = records[-1]
            baseline = records[0]

            if measure.direction == "up":
                delta = latest.score - baseline.score
            else:
                delta = baseline.score - latest.score

            if len(records) >= 2:
                if delta > 0:
                    trend = "improving"
                elif delta < 0:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "baseline_only"

            summary[measure_name] = {
                "full_name": measure.name,
                "baseline": baseline.score,
                "latest": latest.score,
                "delta": round(delta, 2),
                "trend": trend,
                "met_mcid": delta >= measure.mcid if len(records) >= 2 else False,
                "scores_recorded": len(records),
                "last_recorded": latest.recorded_at.isoformat(),
            }

        return summary

    def suggest_measures(
        self,
        discipline: str,
        diagnosis: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Recommend appropriate outcome measures for a discipline/diagnosis.

        Args:
            discipline: "PT", "OT", or "SLP"
            diagnosis: Optional diagnosis keyword for targeted suggestions

        Returns:
            List of dicts with measure name, full_name, and reason.
        """
        suggestions: list[dict[str, Any]] = []

        # Diagnosis-specific suggestions first
        if diagnosis:
            diagnosis_lower = diagnosis.lower().replace(" ", "_")
            for key, measures in _DIAGNOSIS_MEASURES.items():
                if key in diagnosis_lower:
                    for m in measures:
                        if m in OUTCOME_MEASURES:
                            suggestions.append({
                                "measure": m,
                                "full_name": OUTCOME_MEASURES[m].name,
                                "reason": f"Recommended for {key.replace('_', ' ')}",
                            })

        # Add discipline-level defaults if nothing matched
        if not suggestions:
            for key, measure in OUTCOME_MEASURES.items():
                if measure.discipline in (discipline.upper(), "ALL"):
                    suggestions.append({
                        "measure": key,
                        "full_name": measure.name,
                        "reason": f"Standard {measure.discipline} measure",
                    })

        # Deduplicate by measure name
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for s in suggestions:
            if s["measure"] not in seen:
                seen.add(s["measure"])
                unique.append(s)

        return unique
