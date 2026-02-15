"""Cross-namespace memU queries for longitudinal patient context.

Enables RehabOS to pull encounter history from multiple services (e.g.
DocPilot) that share the same memU database but use different namespace
prefixes for their ``user_id`` values.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional, Sequence

from rehab_os.memory.session_memory import SessionMemoryService, _run_async

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACES: list[str] = ["rehab", "docpilot"]


@dataclass
class EncounterRecord:
    """A single encounter from any namespace."""

    namespace: str
    timestamp: str
    patient_id: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_timestamp(ts: str) -> datetime:
    """Best-effort ISO timestamp parse for sorting."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return datetime.min


def get_patient_history(
    memory_service: SessionMemoryService,
    patient_id: str,
    namespaces: Sequence[str] = DEFAULT_NAMESPACES,
) -> list[EncounterRecord]:
    """Query memU across multiple namespace prefixes and return a unified timeline.

    Parameters
    ----------
    memory_service:
        An initialised ``SessionMemoryService`` (provides access to the
        underlying ``_memu_service``).
    patient_id:
        The *bare* patient identifier (no namespace prefix).
    namespaces:
        Namespace prefixes to query (default: ``["rehab", "docpilot"]``).

    Returns
    -------
    list[EncounterRecord]
        Encounters sorted by timestamp (oldest first), each tagged with its
        source namespace.
    """
    records: list[EncounterRecord] = []

    for ns in namespaces:
        scoped_id = f"{ns}:{patient_id}"
        ns_records = _query_namespace(memory_service, scoped_id, ns, patient_id)
        records.extend(ns_records)

    # Sort by timestamp
    records.sort(key=lambda r: _parse_timestamp(r.timestamp))
    return records


def _query_namespace(
    memory_service: SessionMemoryService,
    scoped_id: str,
    namespace: str,
    patient_id: str,
) -> list[EncounterRecord]:
    """Query a single namespace in memU."""
    if not memory_service.is_memu_available or memory_service._memu_service is None:
        # Fall back to in-memory cache for the rehab namespace
        if namespace == "rehab":
            cached = memory_service._cache.get(patient_id, [])
            return [
                EncounterRecord(
                    namespace="rehab",
                    timestamp=r.get("timestamp", ""),
                    patient_id=patient_id,
                    data=r,
                )
                for r in cached
            ]
        return []

    try:
        result = _run_async(
            memory_service._memu_service.retrieve(
                queries=[
                    {
                        "role": "user",
                        "content": {"text": "clinical encounters and consultation history"},
                    }
                ],
                where={"user_id": scoped_id},
                method="rag",
            )
        )

        items = result.get("items", [])
        records: list[EncounterRecord] = []
        for item in items:
            summary = item.get("summary", "")
            data = _try_parse_json(summary)
            ts = data.get("timestamp", item.get("created_at", ""))
            records.append(
                EncounterRecord(
                    namespace=namespace,
                    timestamp=ts,
                    patient_id=patient_id,
                    data=data if data else {"summary": summary},
                )
            )
        return records

    except Exception as e:
        logger.warning("Cross-namespace query failed for %s:%s: %s", namespace, patient_id, e)
        return []


def _try_parse_json(text: str) -> dict:
    """Try to extract a JSON object from text."""
    try:
        if "{" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


def format_cross_namespace_context(records: list[EncounterRecord], max_records: int = 10) -> str:
    """Format encounter records into an LLM-injectable context block."""
    if not records:
        return ""

    recent = records[-max_records:]
    lines = [
        f"=== Cross-Service Patient History ({len(records)} encounter(s), showing last {len(recent)}) ===",
    ]

    for i, rec in enumerate(recent, 1):
        source = rec.namespace.upper()
        ts = rec.timestamp or "unknown date"
        lines.append(f"\n--- Encounter {i} [{source}] ({ts}) ---")

        data = rec.data
        if "primary_diagnosis" in data:
            lines.append(f"Diagnosis: {data['primary_diagnosis']}")
        if "confidence" in data:
            lines.append(f"Confidence: {data['confidence']:.0%}")
        if "goals" in data and data["goals"]:
            lines.append(f"Goals: {'; '.join(data['goals'][:5])}")
        if "interventions" in data and data["interventions"]:
            lines.append(f"Interventions: {'; '.join(data['interventions'][:5])}")
        if "safety_flags" in data and data["safety_flags"]:
            lines.append(f"Safety flags: {', '.join(data['safety_flags'])}")
        if "summary" in data and not any(k in data for k in ("primary_diagnosis", "goals")):
            lines.append(f"Summary: {data['summary'][:300]}")

    lines.append("\n=== End Cross-Service History ===")
    return "\n".join(lines)
