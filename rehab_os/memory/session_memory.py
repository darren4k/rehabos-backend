"""
RehabOS — Session Memory Service

Persists patient consultation history via memU (PostgreSQL + pgvector).
Falls back gracefully to an in-memory dict when memU is unavailable.

All memU user IDs are prefixed with ``rehab:`` to namespace from other
applications sharing the same memU database.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Optional

from rehab_os.models.output import ConsultationResponse

logger = logging.getLogger(__name__)

_NAMESPACE_PREFIX = "rehab:"

# Thread pool for sync→async bridging
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="memu")


def _run_async(coro):
    """Run an async coroutine from synchronous context."""
    try:
        asyncio.get_running_loop()
        # Already in async — offload to thread
        return _executor.submit(asyncio.run, coro).result(timeout=30)
    except RuntimeError:
        return asyncio.run(coro)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionMemoryService:
    """memU-backed patient session memory with in-memory fallback."""

    def __init__(self, enabled: bool = True) -> None:
        self._memu_service: Any = None
        self._memu_available: bool = False
        self._cache: dict[str, list[dict]] = {}  # patient_id -> consultations
        self._outcomes: dict[str, list[dict]] = {}  # patient_id -> outcomes

        if enabled:
            self._init_memu()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_memu(self) -> None:
        """Try to initialize memU. Fall back silently on failure."""
        try:
            from memu.app import MemoryService
            from rehab_os.memory.memu_config import build_memu_service_kwargs

            kwargs = build_memu_service_kwargs()
            self._memu_service = MemoryService(**kwargs)
            self._memu_available = True
            logger.info("memU session memory backend initialized")
        except Exception as e:
            logger.warning("memU unavailable, using in-memory fallback: %s", e)
            self._memu_available = False

    @property
    def is_memu_available(self) -> bool:
        return self._memu_available

    # ------------------------------------------------------------------
    # Scoped user ID
    # ------------------------------------------------------------------

    @staticmethod
    def _scoped_id(patient_id: str) -> str:
        """Prefix patient_id with namespace to avoid collisions."""
        if patient_id.startswith(_NAMESPACE_PREFIX):
            return patient_id
        return f"{_NAMESPACE_PREFIX}{patient_id}"

    # ------------------------------------------------------------------
    # Store consultation
    # ------------------------------------------------------------------

    def store_consultation(
        self,
        patient_id: str,
        consultation: ConsultationResponse,
    ) -> None:
        """Persist key data from a completed consultation."""
        record: dict[str, Any] = {
            "timestamp": _utcnow_iso(),
            "patient_id": patient_id,
        }

        # Extract key fields
        if consultation.diagnosis:
            record["primary_diagnosis"] = consultation.diagnosis.primary_diagnosis
            record["icd_codes"] = consultation.diagnosis.icd_codes
            record["confidence"] = consultation.diagnosis.confidence
            record["differential"] = consultation.diagnosis.differential_diagnoses

        if consultation.plan:
            record["goals"] = [g.description for g in consultation.plan.smart_goals] if consultation.plan.smart_goals else []
            record["interventions"] = [i.name for i in consultation.plan.interventions] if consultation.plan.interventions else []

        if consultation.safety:
            record["safety_flags"] = [rf.finding for rf in consultation.safety.red_flags]
            record["urgency"] = consultation.safety.urgency_level.value

        if consultation.outcomes:
            record["outcome_measures"] = [
                m.name for m in consultation.outcomes.primary_measures
            ]

        if consultation.qa_review:
            record["qa_score"] = consultation.qa_review.overall_quality

        record["processing_notes"] = consultation.processing_notes

        # In-memory cache
        self._cache.setdefault(patient_id, []).append(record)

        # Persist to memU
        if self._memu_available:
            self._persist_record(patient_id, record, "consultation_history")

    # ------------------------------------------------------------------
    # Retrieve patient history
    # ------------------------------------------------------------------

    def get_patient_history(self, patient_id: str) -> list[dict]:
        """Return prior consultations for a patient."""
        # Check memU first
        if self._memu_available:
            history = self._retrieve_history(patient_id)
            if history:
                return history

        # Fall back to in-memory
        return list(self._cache.get(patient_id, []))

    # ------------------------------------------------------------------
    # Longitudinal context (formatted for LLM injection)
    # ------------------------------------------------------------------

    def get_longitudinal_context(self, patient_id: str) -> str:
        """Return a formatted summary of the patient's history for LLM context."""
        history = self.get_patient_history(patient_id)
        if not history:
            return ""

        lines: list[str] = [
            f"=== Patient Longitudinal Context ({len(history)} prior visit(s)) ===",
        ]

        for i, visit in enumerate(history[-10:], 1):  # Last 10 visits
            ts = visit.get("timestamp", "unknown date")
            dx = visit.get("primary_diagnosis", "N/A")
            confidence = visit.get("confidence")
            conf_str = f" (confidence: {confidence:.0%})" if confidence else ""
            urgency = visit.get("urgency", "")
            flags = visit.get("safety_flags", [])

            lines.append(f"\n--- Visit {i} ({ts}) ---")
            lines.append(f"Diagnosis: {dx}{conf_str}")
            if urgency:
                lines.append(f"Urgency: {urgency}")
            if flags:
                lines.append(f"Safety flags: {', '.join(flags)}")

            goals = visit.get("goals", [])
            if goals:
                lines.append(f"Goals: {'; '.join(goals[:5])}")

            interventions = visit.get("interventions", [])
            if interventions:
                lines.append(f"Interventions: {'; '.join(interventions[:5])}")

            measures = visit.get("outcome_measures", [])
            if measures:
                lines.append(f"Outcome measures: {', '.join(measures)}")

        # Add outcome trends if available
        trends = self.get_outcome_trends(patient_id)
        if trends:
            lines.append("\n--- Outcome Trends ---")
            for measure, values in trends.items():
                if len(values) >= 2:
                    change = values[-1].get("value", 0) - values[0].get("value", 0)
                    direction = "improved" if change > 0 else "declined" if change < 0 else "stable"
                    lines.append(f"{measure}: {direction} ({values[0].get('value')} → {values[-1].get('value')})")

        lines.append("\n=== End Longitudinal Context ===")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Outcome tracking
    # ------------------------------------------------------------------

    def store_outcome(self, patient_id: str, outcome_data: dict) -> None:
        """Store outcome measure results."""
        record = {
            "timestamp": _utcnow_iso(),
            "patient_id": patient_id,
            **outcome_data,
        }

        self._outcomes.setdefault(patient_id, []).append(record)

        if self._memu_available:
            self._persist_record(patient_id, record, "treatment_outcomes")

    def get_outcome_trends(self, patient_id: str) -> dict[str, list[dict]]:
        """Return outcome measure trends grouped by measure name."""
        outcomes = list(self._outcomes.get(patient_id, []))

        # Group by measure name
        trends: dict[str, list[dict]] = {}
        for o in outcomes:
            name = o.get("measure_name", o.get("name", "unknown"))
            trends.setdefault(name, []).append(o)

        # Sort each by timestamp
        for name in trends:
            trends[name].sort(key=lambda x: x.get("timestamp", ""))

        return trends

    # ------------------------------------------------------------------
    # memU persistence helpers
    # ------------------------------------------------------------------

    def _persist_record(
        self,
        patient_id: str,
        record: dict,
        category: str,
    ) -> None:
        """Persist a record to memU asynchronously."""
        if not self._memu_available:
            return

        try:
            scoped_id = self._scoped_id(patient_id)
            conversation = [
                {
                    "role": "system",
                    "content": f"RehabOS {category} for patient {patient_id}",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(record, default=str),
                },
            ]

            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="rehab_memu_"
            )
            json.dump(conversation, tmp, default=str)
            tmp.close()

            _run_async(
                self._memu_service.memorize(
                    resource_url=tmp.name,
                    modality="conversation",
                    user={"user_id": scoped_id},
                )
            )

            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        except Exception as e:
            logger.warning("memU persist failed for %s: %s", patient_id, e)

    def _retrieve_history(self, patient_id: str) -> list[dict]:
        """Retrieve consultation history from memU."""
        try:
            scoped_id = self._scoped_id(patient_id)
            result = _run_async(
                self._memu_service.retrieve(
                    queries=[
                        {
                            "role": "user",
                            "content": {"text": "consultation history and clinical visits"},
                        }
                    ],
                    where={"user_id": scoped_id},
                    method="rag",
                )
            )

            items = result.get("items", [])
            records: list[dict] = []
            for item in items:
                summary = item.get("summary", "")
                try:
                    if "{" in summary:
                        start = summary.index("{")
                        end = summary.rindex("}") + 1
                        data = json.loads(summary[start:end])
                        if "patient_id" in data:
                            records.append(data)
                except (json.JSONDecodeError, ValueError):
                    continue

            return records

        except Exception as e:
            logger.warning("memU retrieve failed for %s: %s", patient_id, e)
            return []
