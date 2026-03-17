"""Longitudinal flow sheet tracking across encounters for RehabOS.

Tracks clinical data points (ROM, strength, functional status, pain, etc.)
across encounters for a patient, enabling trend analysis and charting.

TODO: Replace in-memory storage with database-backed persistence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class FlowSheetColumn:
    """Definition of a single trackable clinical measure."""

    key: str
    label: str
    unit: str
    category: str  # "rom", "strength", "functional", "pain", "vital", "outcome"
    body_region: str | None = None  # "lumbar", "shoulder_r", "knee_l", etc.


@dataclass
class FlowSheetEntry:
    """A single row of flow sheet data tied to an encounter."""

    encounter_id: str
    encounter_date: str  # ISO date string
    provider_id: str
    data: dict[str, Any]  # column_key -> value
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Predefined column sets by discipline
# ---------------------------------------------------------------------------

PT_COLUMNS: list[FlowSheetColumn] = [
    # ROM
    FlowSheetColumn("rom_flex", "Flexion ROM", "degrees", "rom"),
    FlowSheetColumn("rom_ext", "Extension ROM", "degrees", "rom"),
    FlowSheetColumn("rom_abd", "Abduction ROM", "degrees", "rom"),
    FlowSheetColumn("rom_add", "Adduction ROM", "degrees", "rom"),
    FlowSheetColumn("rom_ir", "Internal Rotation ROM", "degrees", "rom"),
    FlowSheetColumn("rom_er", "External Rotation ROM", "degrees", "rom"),
    # Strength
    FlowSheetColumn("mmt", "MMT Grade", "0-5", "strength"),
    FlowSheetColumn("mmt_flex", "MMT Flexion", "0-5", "strength"),
    FlowSheetColumn("mmt_ext", "MMT Extension", "0-5", "strength"),
    FlowSheetColumn("mmt_abd", "MMT Abduction", "0-5", "strength"),
    # Pain
    FlowSheetColumn("pain_current", "Pain Now", "0-10", "pain"),
    FlowSheetColumn("pain_best", "Pain Best", "0-10", "pain"),
    FlowSheetColumn("pain_worst", "Pain Worst", "0-10", "pain"),
    # Functional
    FlowSheetColumn("gait_distance", "Gait Distance", "feet", "functional"),
    FlowSheetColumn("gait_device", "Assistive Device", "", "functional"),
    FlowSheetColumn("gait_speed", "Gait Speed", "m/s", "functional"),
    FlowSheetColumn("transfers", "Transfer Level", "min-max-mod-SBA-CGA-I", "functional"),
    FlowSheetColumn("balance_static", "Static Balance", "seconds", "functional"),
    FlowSheetColumn("balance_dynamic", "Dynamic Balance", "good/fair/poor", "functional"),
    FlowSheetColumn("tug", "Timed Up & Go", "seconds", "functional"),
    FlowSheetColumn("sit_to_stand", "5x Sit to Stand", "seconds", "functional"),
    # Vitals
    FlowSheetColumn("bp_systolic", "BP Systolic", "mmHg", "vital"),
    FlowSheetColumn("bp_diastolic", "BP Diastolic", "mmHg", "vital"),
    FlowSheetColumn("hr", "Heart Rate", "bpm", "vital"),
    FlowSheetColumn("spo2", "SpO2", "%", "vital"),
]

OT_COLUMNS: list[FlowSheetColumn] = [
    # Strength
    FlowSheetColumn("grip_strength_r", "Grip Strength (R)", "lbs", "strength"),
    FlowSheetColumn("grip_strength_l", "Grip Strength (L)", "lbs", "strength"),
    FlowSheetColumn("pinch_strength_r", "Pinch Strength (R)", "lbs", "strength"),
    FlowSheetColumn("pinch_strength_l", "Pinch Strength (L)", "lbs", "strength"),
    # ROM
    FlowSheetColumn("rom_wrist_flex", "Wrist Flexion ROM", "degrees", "rom"),
    FlowSheetColumn("rom_wrist_ext", "Wrist Extension ROM", "degrees", "rom"),
    FlowSheetColumn("rom_digit_flex", "Digit Flexion ROM", "degrees", "rom"),
    # ADLs
    FlowSheetColumn("adl_feeding", "Feeding", "I/SBA/CGA/Mod/Max/Dep", "functional"),
    FlowSheetColumn("adl_dressing_ue", "Dressing UE", "I/SBA/CGA/Mod/Max/Dep", "functional"),
    FlowSheetColumn("adl_dressing_le", "Dressing LE", "I/SBA/CGA/Mod/Max/Dep", "functional"),
    FlowSheetColumn("adl_grooming", "Grooming", "I/SBA/CGA/Mod/Max/Dep", "functional"),
    FlowSheetColumn("adl_bathing", "Bathing", "I/SBA/CGA/Mod/Max/Dep", "functional"),
    FlowSheetColumn("adl_toileting", "Toileting", "I/SBA/CGA/Mod/Max/Dep", "functional"),
    FlowSheetColumn("adl_toilet_transfer", "Toilet Transfer", "I/SBA/CGA/Mod/Max/Dep", "functional"),
    FlowSheetColumn("adl_tub_transfer", "Tub/Shower Transfer", "I/SBA/CGA/Mod/Max/Dep", "functional"),
    # Pain
    FlowSheetColumn("pain_current", "Pain Now", "0-10", "pain"),
    FlowSheetColumn("pain_worst", "Pain Worst", "0-10", "pain"),
    # Cognition
    FlowSheetColumn("cognition_orientation", "Orientation", "x0-x4", "functional"),
    FlowSheetColumn("cognition_attention", "Attention", "WNL/mild/mod/severe", "functional"),
    FlowSheetColumn("cognition_memory", "Short-term Memory", "WNL/mild/mod/severe", "functional"),
]

SLP_COLUMNS: list[FlowSheetColumn] = [
    # Swallowing
    FlowSheetColumn("oral_motor", "Oral Motor Function", "WNL/mild/mod/severe", "functional"),
    FlowSheetColumn("swallow_liquid", "Swallow - Liquids", "regular/nectar/honey/pudding/NPO", "functional"),
    FlowSheetColumn("swallow_solid", "Swallow - Solids", "regular/mechanical/pureed/NPO", "functional"),
    FlowSheetColumn("aspiration_risk", "Aspiration Risk", "low/moderate/high", "functional"),
    FlowSheetColumn("diet_level", "Diet Level (IDDSI)", "0-7", "functional"),
    # Voice
    FlowSheetColumn("voice_quality", "Voice Quality", "WNL/breathy/hoarse/strained", "functional"),
    FlowSheetColumn("voice_volume", "Voice Volume", "WNL/reduced/aphonic", "functional"),
    FlowSheetColumn("mpt", "Max Phonation Time", "seconds", "functional"),
    # Language
    FlowSheetColumn("intelligibility", "Intelligibility", "percent", "functional"),
    FlowSheetColumn("auditory_comp", "Auditory Comprehension", "WNL/mild/mod/severe", "functional"),
    FlowSheetColumn("verbal_expression", "Verbal Expression", "WNL/mild/mod/severe", "functional"),
    FlowSheetColumn("reading_comp", "Reading Comprehension", "WNL/mild/mod/severe", "functional"),
    FlowSheetColumn("written_expression", "Written Expression", "WNL/mild/mod/severe", "functional"),
    # Cognition
    FlowSheetColumn("cognition_orientation", "Orientation", "x0-x4", "functional"),
    FlowSheetColumn("cognition_attention", "Attention", "WNL/mild/mod/severe", "functional"),
    FlowSheetColumn("cognition_memory", "Short-term Memory", "WNL/mild/mod/severe", "functional"),
    FlowSheetColumn("cognition_problem_solving", "Problem Solving", "WNL/mild/mod/severe", "functional"),
]

_DISCIPLINE_COLUMNS: dict[str, list[FlowSheetColumn]] = {
    "PT": PT_COLUMNS,
    "OT": OT_COLUMNS,
    "SLP": SLP_COLUMNS,
}


class FlowSheetService:
    """Track clinical data points across encounters for a patient.

    Currently in-memory. TODO: migrate to database persistence.
    """

    def __init__(self) -> None:
        # {patient_id: [FlowSheetEntry, ...]}
        self._entries: dict[str, list[FlowSheetEntry]] = {}

    def get_columns(self, discipline: str) -> list[FlowSheetColumn]:
        """Get the predefined columns for a discipline (PT/OT/SLP)."""
        key = discipline.upper()
        if key not in _DISCIPLINE_COLUMNS:
            raise ValueError(f"Unknown discipline: {discipline}. Valid: PT, OT, SLP")
        return _DISCIPLINE_COLUMNS[key]

    def record_entry(
        self,
        patient_id: str,
        encounter_id: str,
        encounter_date: str,
        provider_id: str,
        data: dict[str, Any],
    ) -> FlowSheetEntry:
        """Record a new flow sheet entry for a patient encounter.

        Args:
            patient_id: Patient UUID string.
            encounter_id: Encounter UUID string.
            encounter_date: ISO date string (YYYY-MM-DD).
            provider_id: Provider UUID string.
            data: Dict mapping column keys to values.

        Returns:
            The created FlowSheetEntry.
        """
        entry = FlowSheetEntry(
            encounter_id=encounter_id,
            encounter_date=encounter_date,
            provider_id=provider_id,
            data=data,
        )
        self._entries.setdefault(patient_id, []).append(entry)
        # Keep sorted by date
        self._entries[patient_id].sort(key=lambda e: e.encounter_date)
        logger.info(
            "Flow sheet entry recorded for patient %s, encounter %s (%d data points)",
            patient_id,
            encounter_id,
            len(data),
        )
        return entry

    def get_flow_sheet(
        self,
        patient_id: str,
        discipline: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[FlowSheetEntry]:
        """Get flow sheet entries for a patient.

        Args:
            patient_id: Patient UUID string.
            discipline: If provided, filter data keys to this discipline's columns.
            date_from: Optional start date filter (inclusive, ISO string).
            date_to: Optional end date filter (inclusive, ISO string).

        Returns:
            List of FlowSheetEntry, sorted by encounter_date ascending.
        """
        entries = self._entries.get(patient_id, [])

        # Date range filter
        if date_from:
            entries = [e for e in entries if e.encounter_date >= date_from]
        if date_to:
            entries = [e for e in entries if e.encounter_date <= date_to]

        # Discipline column filter
        if discipline:
            valid_keys = {col.key for col in self.get_columns(discipline)}
            filtered = []
            for entry in entries:
                filtered_data = {k: v for k, v in entry.data.items() if k in valid_keys}
                filtered.append(FlowSheetEntry(
                    encounter_id=entry.encounter_id,
                    encounter_date=entry.encounter_date,
                    provider_id=entry.provider_id,
                    data=filtered_data,
                    recorded_at=entry.recorded_at,
                ))
            entries = filtered

        return entries

    def get_trending_data(
        self,
        patient_id: str,
        column_key: str,
    ) -> list[dict[str, Any]]:
        """Get trend data for a specific column across all encounters.

        Returns a list of {date, value, encounter_id} dicts for charting.
        """
        entries = self._entries.get(patient_id, [])
        trend: list[dict[str, Any]] = []

        for entry in entries:
            if column_key in entry.data:
                value = entry.data[column_key]
                trend.append({
                    "date": entry.encounter_date,
                    "value": value,
                    "encounter_id": entry.encounter_id,
                    "provider_id": entry.provider_id,
                })

        return trend

    def get_summary(self, patient_id: str, discipline: str) -> dict[str, Any]:
        """Get summary statistics for a patient's flow sheet.

        Returns column-level summaries with first/last values and trends.
        """
        entries = self.get_flow_sheet(patient_id, discipline)
        if not entries:
            return {"total_entries": 0, "columns": {}}

        columns = self.get_columns(discipline)
        summary: dict[str, Any] = {
            "total_entries": len(entries),
            "date_range": {
                "first": entries[0].encounter_date,
                "last": entries[-1].encounter_date,
            },
            "columns": {},
        }

        for col in columns:
            values = []
            for entry in entries:
                if col.key in entry.data and entry.data[col.key] is not None:
                    values.append({
                        "date": entry.encounter_date,
                        "value": entry.data[col.key],
                    })

            if values:
                first_val = values[0]["value"]
                last_val = values[-1]["value"]

                # Determine trend for numeric values
                trend = "stable"
                try:
                    first_num = float(first_val)
                    last_num = float(last_val)
                    delta = last_num - first_num
                    if abs(delta) > 0.01:
                        trend = "improving" if delta > 0 else "declining"
                        # Invert for pain (lower is better)
                        if col.category == "pain":
                            trend = "improving" if delta < 0 else "declining"
                except (ValueError, TypeError):
                    trend = "n/a"

                summary["columns"][col.key] = {
                    "label": col.label,
                    "unit": col.unit,
                    "category": col.category,
                    "first_value": first_val,
                    "last_value": last_val,
                    "data_points": len(values),
                    "trend": trend,
                }

        return summary


# Module-level singleton
_flow_sheet_service: FlowSheetService | None = None


def get_flow_sheet_service() -> FlowSheetService:
    """Get or create the singleton FlowSheetService instance."""
    global _flow_sheet_service
    if _flow_sheet_service is None:
        _flow_sheet_service = FlowSheetService()
    return _flow_sheet_service
