"""Encounter Brain — intelligent clinical documentation orchestrator."""

from rehab_os.encounter.state import EncounterState, EncounterPhase
from rehab_os.encounter.brain import EncounterBrain
from rehab_os.encounter.memory import load_patient_context

__all__ = ["EncounterState", "EncounterPhase", "EncounterBrain", "load_patient_context"]
