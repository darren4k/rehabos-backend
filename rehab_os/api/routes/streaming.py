"""WebSocket streaming for real-time consultation updates."""

import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streaming"])


class StreamingConsultRequest(BaseModel):
    """Request for streaming consultation."""

    query: str
    discipline: str = "PT"
    setting: str = "outpatient"
    patient: Optional[dict[str, Any]] = None
    session_id: Optional[str] = None
    skip_qa: bool = False


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")

    async def send_json(self, client_id: str, data: dict[str, Any]):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(data)

    async def send_text(self, client_id: str, text: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(text)


manager = ConnectionManager()


@router.websocket("/ws/consult/{client_id}")
async def websocket_consult(
    websocket: WebSocket,
    client_id: str,
):
    """WebSocket endpoint for streaming consultation results.

    Protocol:
    1. Client connects with unique client_id
    2. Client sends JSON request: {query, discipline, setting, patient?, session_id?}
    3. Server streams progress updates as JSON messages:
       - {type: "status", step: "safety", message: "Running safety check..."}
       - {type: "result", step: "safety", data: {...}}
       - {type: "status", step: "diagnosis", message: "Analyzing..."}
       - {type: "result", step: "diagnosis", data: {...}}
       - ...
       - {type: "complete", data: {full response}}
    4. On error: {type: "error", message: "..."}
    """
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Wait for consultation request
            data = await websocket.receive_text()

            try:
                request = json.loads(data)
                await process_streaming_consult(client_id, request)
            except json.JSONDecodeError:
                await manager.send_json(client_id, {
                    "type": "error",
                    "message": "Invalid JSON request",
                })
            except Exception as e:
                logger.error(f"Streaming consult error: {e}")
                await manager.send_json(client_id, {
                    "type": "error",
                    "message": str(e),
                })

    except WebSocketDisconnect:
        manager.disconnect(client_id)


async def process_streaming_consult(client_id: str, request: dict[str, Any]):
    """Process consultation with streaming updates."""
    from rehab_os.models.output import ClinicalRequest
    from rehab_os.models.patient import PatientContext, Discipline, CareSetting
    from rehab_os.observability import get_observability_logger

    obs = get_observability_logger()
    obs.set_session_id(request.get("session_id"))

    # Send initial status
    await manager.send_json(client_id, {
        "type": "status",
        "step": "init",
        "message": "Starting consultation...",
    })

    try:
        # Build patient context
        patient_data = request.get("patient", {})
        patient = PatientContext(
            age=patient_data.get("age", 50),
            sex=patient_data.get("sex", "unknown"),
            chief_complaint=patient_data.get("chief_complaint", request.get("query", "")),
            discipline=Discipline(request.get("discipline", "PT")),
            setting=CareSetting(request.get("setting", "outpatient")),
            diagnosis=patient_data.get("diagnosis", []),
            comorbidities=patient_data.get("comorbidities", []),
            medications=patient_data.get("medications", []),
        )

        # Create clinical request
        clinical_request = ClinicalRequest(
            query=request.get("query", ""),
            patient=patient,
            discipline=Discipline(request.get("discipline", "PT")),
            setting=CareSetting(request.get("setting", "outpatient")),
        )

        # Get orchestrator from app state (simplified - in production use dependency injection)
        from rehab_os.llm import create_router_from_settings
        from rehab_os.agents import Orchestrator

        llm = create_router_from_settings()
        orchestrator = Orchestrator(llm=llm)

        # Stream each step
        steps = [
            ("safety", "Running safety screening...", orchestrator._run_safety_check),
        ]

        # Step 1: Safety
        await manager.send_json(client_id, {
            "type": "status",
            "step": "safety",
            "message": "Checking for red flags...",
        })

        from rehab_os.agents.base import AgentContext
        context = AgentContext(
            discipline=request.get("discipline", "PT"),
            setting=request.get("setting", "outpatient"),
        )

        safety = await orchestrator._run_safety_check(clinical_request, context)

        await manager.send_json(client_id, {
            "type": "result",
            "step": "safety",
            "data": {
                "is_safe_to_treat": safety.is_safe_to_treat,
                "urgency_level": safety.urgency_level.value,
                "red_flags_count": len(safety.red_flags),
                "summary": safety.summary,
            },
        })

        # Check for critical findings
        if safety.has_critical_findings:
            await manager.send_json(client_id, {
                "type": "complete",
                "data": {
                    "safety": safety.model_dump(),
                    "is_emergency": True,
                    "message": "Critical red flags detected - consultation stopped",
                },
            })
            return

        # Step 2: Diagnosis & Evidence (parallel)
        await manager.send_json(client_id, {
            "type": "status",
            "step": "diagnosis",
            "message": "Analyzing clinical findings...",
        })

        diagnosis, evidence = await orchestrator._run_diagnosis_and_evidence(clinical_request, context)

        await manager.send_json(client_id, {
            "type": "result",
            "step": "diagnosis",
            "data": {
                "primary_diagnosis": diagnosis.primary_diagnosis,
                "icd_codes": diagnosis.icd_codes,
                "confidence": diagnosis.confidence,
                "rationale": diagnosis.rationale[:200] if diagnosis.rationale else None,
            },
        })

        await manager.send_json(client_id, {
            "type": "result",
            "step": "evidence",
            "data": {
                "total_sources": evidence.total_sources,
                "synthesis": evidence.synthesis[:300] if evidence.synthesis else None,
            },
        })

        # Step 3: Planning
        await manager.send_json(client_id, {
            "type": "status",
            "step": "plan",
            "message": "Generating treatment plan...",
        })

        plan = await orchestrator._run_planning(clinical_request, diagnosis, evidence, context)

        await manager.send_json(client_id, {
            "type": "result",
            "step": "plan",
            "data": {
                "clinical_summary": plan.clinical_summary,
                "prognosis": plan.prognosis,
                "goals_count": len(plan.smart_goals),
                "interventions_count": len(plan.interventions),
                "visit_frequency": plan.visit_frequency,
            },
        })

        # Step 4: Outcomes
        await manager.send_json(client_id, {
            "type": "status",
            "step": "outcomes",
            "message": "Selecting outcome measures...",
        })

        outcomes = await orchestrator._run_outcome_selection(clinical_request, diagnosis, context)

        await manager.send_json(client_id, {
            "type": "result",
            "step": "outcomes",
            "data": {
                "primary_measures": [m.name for m in outcomes.primary_measures],
                "secondary_measures": [m.name for m in outcomes.secondary_measures],
            },
        })

        # Step 5: QA (optional)
        qa_result = None
        if not request.get("skip_qa", False):
            await manager.send_json(client_id, {
                "type": "status",
                "step": "qa",
                "message": "Running quality review...",
            })

            qa_result = await orchestrator._run_qa_review(
                clinical_request, safety, diagnosis, plan, evidence, context
            )

            await manager.send_json(client_id, {
                "type": "result",
                "step": "qa",
                "data": {
                    "overall_quality": qa_result.overall_quality,
                    "strengths": qa_result.strengths[:3] if qa_result.strengths else [],
                    "suggestions": qa_result.suggestions[:3] if qa_result.suggestions else [],
                },
            })

        # Complete response
        await manager.send_json(client_id, {
            "type": "complete",
            "data": {
                "safety": safety.model_dump(),
                "diagnosis": diagnosis.model_dump(),
                "evidence": {
                    "total_sources": evidence.total_sources,
                    "synthesis": evidence.synthesis,
                    "confidence": evidence.confidence,
                },
                "plan": plan.model_dump(),
                "outcomes": outcomes.model_dump(),
                "qa_review": qa_result.model_dump() if qa_result else None,
            },
        })

    except Exception as e:
        logger.error(f"Streaming consult failed: {e}")
        await manager.send_json(client_id, {
            "type": "error",
            "message": f"Consultation failed: {str(e)}",
        })


@router.get("/ws/status")
async def websocket_status():
    """Get WebSocket connection status."""
    return {
        "active_connections": len(manager.active_connections),
        "connection_ids": list(manager.active_connections.keys()),
    }
