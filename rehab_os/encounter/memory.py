"""Memory Agent — loads patient context from memU and Patient-Core."""
from __future__ import annotations

import logging
from typing import Any, Optional

from rehab_os.encounter.state import PatientHistory

logger = logging.getLogger(__name__)


async def load_patient_context(
    patient_id: str,
    session_memory: Any = None,
    db_session: Any = None,
) -> PatientHistory:
    """Load comprehensive patient context for the encounter Brain.

    Sources:
    1. Patient-Core database (demographics, diagnosis, encounters)
    2. memU session memory (clinical memory, cross-namespace)
    """
    history = PatientHistory()

    # 1. Patient-Core database
    if db_session:
        try:
            import uuid as _uuid

            from rehab_os.core.repository import ClinicalNoteRepository, PatientRepository

            pid = _uuid.UUID(patient_id) if isinstance(patient_id, str) else patient_id

            patient_repo = PatientRepository(db_session)
            patient = await patient_repo.get_by_id(pid)

            if patient:
                note_repo = ClinicalNoteRepository(db_session)
                notes = await note_repo.list_by_patient(pid, limit=3)

                for note in notes:
                    enc: dict[str, Any] = {
                        "date": note.note_date.isoformat() if note.note_date else "",
                        "note_type": note.note_type,
                        "summary": (note.soap_subjective or "")[:100],
                    }
                    if note.structured_data:
                        sd = note.structured_data
                        vitals = sd.get("vitals") or {}
                        if vitals.get("pain_level") is not None:
                            enc["pain_level"] = vitals["pain_level"]
                        if sd.get("rom"):
                            enc["rom"] = sd["rom"]
                    history.last_encounters.append(enc)
        except Exception as e:
            logger.warning("Failed to load from Patient-Core: %s", e)

    # 2. memU session memory
    if session_memory:
        try:
            consultations = session_memory.get_consultation_history(
                patient_id=patient_id, limit=5
            )
            for c in consultations or []:
                c_date = c.get("timestamp", "")[:10]
                if not any(e.get("date") == c_date for e in history.last_encounters):
                    history.last_encounters.append(
                        {
                            "date": c_date,
                            "summary": c.get("summary", "")[:100],
                            "pain_level": c.get("pain_level"),
                        }
                    )
        except Exception as e:
            logger.warning("Failed to load from memU: %s", e)

    # Sort by date descending
    history.last_encounters.sort(key=lambda x: x.get("date", ""), reverse=True)

    return history
