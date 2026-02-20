"""Direct agent access endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from rehab_os.agents.base import AgentContext

logger = logging.getLogger(__name__)

router = APIRouter()


class AgentRequest(BaseModel):
    """Request for direct agent invocation."""

    inputs: dict[str, Any] = Field(..., description="Agent-specific input data")
    discipline: str = Field(default="PT")
    setting: str = Field(default="outpatient")


class EvidenceSearchRequest(BaseModel):
    """Request for evidence search."""

    condition: str = Field(..., description="Clinical condition")
    clinical_question: str = Field(..., description="Specific clinical question")
    patient_population: str | None = None
    intervention_focus: str | None = None
    discipline: str = Field(default="PT")


@router.post("/{agent_name}")
async def run_agent(
    request: Request,
    agent_name: str,
    agent_request: AgentRequest,
) -> dict:
    """Run a single agent directly.

    Available agents:
    - red_flag: Safety screening
    - diagnosis: Clinical classification
    - evidence: Evidence retrieval
    - plan: Treatment planning
    - outcome: Outcome measure selection
    - documentation: Note generation
    - qa: Quality assurance review
    """
    orchestrator = request.app.state.orchestrator

    context = AgentContext(
        discipline=agent_request.discipline,
        setting=agent_request.setting,
    )

    try:
        result = await orchestrator.run_single_agent(
            agent_name=agent_name,
            inputs=agent_request.inputs,
            context=context,
        )
        return {"agent": agent_name, "result": result.model_dump()}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Agent {agent_name} error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal processing error",
        )


@router.post("/evidence/search")
async def search_evidence(
    request: Request,
    search_request: EvidenceSearchRequest,
) -> dict:
    """Search for clinical evidence.

    Searches both the local knowledge base and optionally PubMed
    for relevant clinical evidence.
    """
    from rehab_os.agents import EvidenceAgent
    from rehab_os.agents.evidence import EvidenceInput

    llm = request.app.state.llm_router
    knowledge_base = request.app.state.vector_store

    agent = EvidenceAgent(llm=llm, knowledge_base=knowledge_base)

    evidence_input = EvidenceInput(
        condition=search_request.condition,
        clinical_question=search_request.clinical_question,
        patient_population=search_request.patient_population,
        intervention_focus=search_request.intervention_focus,
        setting=search_request.discipline,
    )

    context = AgentContext(discipline=search_request.discipline)

    try:
        result = await agent.run(evidence_input, context)
        return {
            "query": result.query,
            "total_sources": result.total_sources,
            "synthesis": result.synthesis,
            "evidence_items": [e.model_dump() for e in result.evidence_items],
            "guideline_recommendations": [
                r.model_dump() for r in result.guideline_recommendations
            ],
            "confidence": result.confidence,
        }

    except Exception as e:
        logger.exception(f"Evidence search error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal processing error",
        )


@router.get("/list")
async def list_agents() -> dict:
    """List available agents and their descriptions."""
    return {
        "agents": [
            {
                "name": "red_flag",
                "description": "Safety screening and red flag identification",
                "input_schema": "RedFlagInput",
            },
            {
                "name": "diagnosis",
                "description": "Clinical diagnosis and classification",
                "input_schema": "DiagnosisInput",
            },
            {
                "name": "evidence",
                "description": "Evidence retrieval and synthesis",
                "input_schema": "EvidenceInput",
            },
            {
                "name": "plan",
                "description": "Treatment planning and goal setting",
                "input_schema": "PlanInput",
            },
            {
                "name": "outcome",
                "description": "Outcome measure selection",
                "input_schema": "OutcomeInput",
            },
            {
                "name": "documentation",
                "description": "Clinical documentation generation",
                "input_schema": "DocumentationInput",
            },
            {
                "name": "qa",
                "description": "Quality assurance and review",
                "input_schema": "QAInput",
            },
        ]
    }
