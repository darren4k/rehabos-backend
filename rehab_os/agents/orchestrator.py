"""Orchestrator for coordinating clinical reasoning agents."""

import asyncio
import logging
from typing import Any, Optional

from rehab_os.agents.base import AgentContext
from rehab_os.agents.diagnosis import DiagnosisAgent, DiagnosisInput
from rehab_os.agents.documentation import DocumentationAgent, DocumentationInput
from rehab_os.agents.evidence import EvidenceAgent, EvidenceInput
from rehab_os.agents.outcome import OutcomeAgent, OutcomeInput
from rehab_os.agents.plan import PlanAgent, PlanInput
from rehab_os.agents.qa_learning import QAInput, QALearningAgent
from rehab_os.agents.red_flag import RedFlagAgent, RedFlagInput
from rehab_os.llm import LLMRouter
from rehab_os.models.output import (
    ClinicalRequest,
    ConsultationResponse,
    DocumentationType,
    SafetyAssessment,
    UrgencyLevel,
)
from rehab_os.observability import get_observability_logger

logger = logging.getLogger(__name__)


class Orchestrator:
    """Central coordinator for the multi-agent clinical reasoning pipeline.

    Manages the flow of information between agents:
    1. RedFlagAgent (safety gate) - always runs first
    2. DiagnosisAgent + EvidenceAgent (parallel)
    3. PlanAgent (uses diagnosis + evidence)
    4. OutcomeAgent (uses diagnosis)
    5. DocumentationAgent (optional, uses all above)
    6. QALearningAgent (final review)
    """

    def __init__(
        self,
        llm: LLMRouter,
        knowledge_base: Optional[Any] = None,
        session_memory: Optional[Any] = None,
    ):
        """Initialize orchestrator with all agents.

        Args:
            llm: LLM router for model access
            knowledge_base: Optional knowledge base for evidence retrieval
            session_memory: Optional SessionMemoryService for longitudinal context
        """
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.session_memory = session_memory

        # Initialize agents
        self.red_flag_agent = RedFlagAgent(llm)
        self.diagnosis_agent = DiagnosisAgent(llm)
        self.evidence_agent = EvidenceAgent(llm, knowledge_base)
        self.plan_agent = PlanAgent(llm)
        self.outcome_agent = OutcomeAgent(llm)
        self.documentation_agent = DocumentationAgent(llm)
        self.qa_agent = QALearningAgent(llm)

        logger.info("Orchestrator initialized with all agents")

    async def process(
        self,
        request: ClinicalRequest,
        skip_qa: bool = False,
    ) -> ConsultationResponse:
        """Process a clinical consultation request through the agent pipeline.

        Args:
            request: The clinical consultation request
            skip_qa: Skip QA review for faster response

        Returns:
            Complete consultation response
        """
        obs = get_observability_logger()
        query_summary = request.query[:100] if request.query else None

        with obs.orchestrator_run(
            discipline=request.discipline.value,
            setting=request.setting.value,
            query_summary=query_summary,
        ) as event:
            logger.info(f"Processing consultation: {request.task_type}")

            # Create shared context
            context = AgentContext(
                discipline=request.discipline.value,
                setting=request.setting.value,
            )

            # Inject longitudinal patient context if available
            if self.session_memory and request.patient_id:
                try:
                    longitudinal = self.session_memory.get_longitudinal_context(
                        request.patient_id
                    )
                    if longitudinal:
                        context.metadata["longitudinal_context"] = longitudinal
                        logger.info(
                            "Injected longitudinal context for patient %s",
                            request.patient_id,
                        )
                except Exception as e:
                    logger.warning("Failed to retrieve longitudinal context: %s", e)

            processing_notes: list[str] = []
            agents_called: list[str] = []

            # Step 1: Safety check (always first, blocking)
            logger.info("Step 1: Running safety screening")
            safety = await self._run_safety_check(request, context)
            agents_called.append("red_flag")
            processing_notes.append(f"Safety screening: {safety.urgency_level.value}")

            # Check for critical findings
            if safety.has_critical_findings:
                logger.warning("Critical red flags detected - returning emergency response")
                event.has_red_flags = True
                event.is_emergency = True
                event.agents_called = agents_called
                return self._create_emergency_response(safety, request)

            event.has_red_flags = len(safety.red_flags) > 0

            # Route based on task type
            if request.task_type == "safety_only":
                event.agents_called = agents_called
                return ConsultationResponse(safety=safety, processing_notes=processing_notes)

            # Step 2: Diagnosis and Evidence (parallel when possible)
            logger.info("Step 2: Running diagnosis and evidence retrieval")
            diagnosis, evidence = await self._run_diagnosis_and_evidence(request, context)
            agents_called.extend(["diagnosis", "evidence"])
            processing_notes.append(f"Diagnosis confidence: {diagnosis.confidence:.2f}")
            processing_notes.append(f"Evidence sources: {evidence.total_sources}")
            event.diagnosis_confidence = diagnosis.confidence

            if request.task_type == "diagnosis_only":
                event.agents_called = agents_called
                return ConsultationResponse(
                    safety=safety,
                    diagnosis=diagnosis,
                    evidence=evidence,
                    processing_notes=processing_notes,
                )

            # Step 3: Treatment planning
            logger.info("Step 3: Generating treatment plan")
            plan = await self._run_planning(request, diagnosis, evidence, context)
            agents_called.append("plan")
            processing_notes.append(f"Goals: {len(plan.smart_goals)}")
            processing_notes.append(f"Interventions: {len(plan.interventions)}")

            # Step 4: Outcome measures
            logger.info("Step 4: Recommending outcome measures")
            outcomes = await self._run_outcome_selection(request, diagnosis, context)
            agents_called.append("outcome")
            processing_notes.append(
                f"Outcome measures: {len(outcomes.primary_measures) + len(outcomes.secondary_measures)}"
            )

            if request.task_type == "plan_only":
                event.agents_called = agents_called
                return ConsultationResponse(
                    safety=safety,
                    diagnosis=diagnosis,
                    plan=plan,
                    outcomes=outcomes,
                    processing_notes=processing_notes,
                )

            # Step 5: Documentation (optional)
            documentation = None
            if request.include_documentation:
                logger.info("Step 5: Generating documentation")
                documentation = await self._run_documentation(
                    request, safety, diagnosis, plan, outcomes, evidence, context
                )
                agents_called.append("documentation")
                processing_notes.append(f"Documentation: {documentation.document_type.value}")

            # Step 6: QA Review
            qa_result = None
            if not skip_qa:
                logger.info("Step 6: Running QA review")
                qa_result = await self._run_qa_review(
                    request, safety, diagnosis, plan, evidence, context
                )
                agents_called.append("qa")
                processing_notes.append(f"QA score: {qa_result.overall_quality:.2f}")
                event.qa_score = qa_result.overall_quality

            # Aggregate citations
            all_citations = []
            if evidence and evidence.evidence_items:
                for item in evidence.evidence_items:
                    if item.citation:
                        all_citations.append(item.citation)

            event.agents_called = agents_called
            event.total_llm_calls = len(agents_called)

            response = ConsultationResponse(
                safety=safety,
                diagnosis=diagnosis,
                evidence=evidence,
                plan=plan,
                outcomes=outcomes,
                documentation=documentation,
                qa_review=qa_result,
                citations=all_citations,
                processing_notes=processing_notes,
            )

            # Persist consultation to session memory
            if self.session_memory and request.patient_id:
                try:
                    self.session_memory.store_consultation(
                        request.patient_id, response
                    )
                    logger.info(
                        "Stored consultation for patient %s", request.patient_id
                    )
                except Exception as e:
                    logger.warning("Failed to store consultation: %s", e)

            return response

    async def _run_safety_check(
        self,
        request: ClinicalRequest,
        context: AgentContext,
    ) -> SafetyAssessment:
        """Run red flag screening."""
        if not request.patient:
            # Minimal safety response if no patient context
            return SafetyAssessment(
                is_safe_to_treat=True,
                urgency_level=UrgencyLevel.ROUTINE,
                summary="No patient context provided - unable to perform full safety screening",
            )

        red_flag_input = RedFlagInput(
            patient=request.patient,
            chief_complaint=request.patient.chief_complaint,
            subjective_report=request.patient.subjective_notes,
            objective_findings=request.patient.objective_findings,
        )

        return await self.red_flag_agent.run(red_flag_input, context)

    async def _run_diagnosis_and_evidence(
        self,
        request: ClinicalRequest,
        context: AgentContext,
    ):
        """Run diagnosis and evidence retrieval in parallel."""
        if not request.patient:
            raise ValueError("Patient context required for diagnosis")

        # Prepare inputs
        diagnosis_input = DiagnosisInput(
            patient=request.patient,
            subjective=request.patient.subjective_notes or request.query,
            objective=request.patient.objective_findings or "",
        )

        evidence_input = EvidenceInput(
            condition=request.patient.chief_complaint,
            clinical_question=request.query,
            patient_population=f"{request.patient.age}yo {request.patient.sex}",
            setting=request.setting.value,
        )

        # Run in parallel
        diagnosis, evidence = await asyncio.gather(
            self.diagnosis_agent.run(diagnosis_input, context),
            self.evidence_agent.run(evidence_input, context),
        )

        return diagnosis, evidence

    async def _run_planning(
        self,
        request: ClinicalRequest,
        diagnosis,
        evidence,
        context: AgentContext,
    ):
        """Run treatment planning agent."""
        plan_input = PlanInput(
            patient=request.patient,
            diagnosis=diagnosis,
            evidence=evidence,
        )

        return await self.plan_agent.run(plan_input, context)

    async def _run_outcome_selection(
        self,
        request: ClinicalRequest,
        diagnosis,
        context: AgentContext,
    ):
        """Run outcome measure selection."""
        outcome_input = OutcomeInput(
            patient=request.patient,
            diagnosis=diagnosis,
        )

        return await self.outcome_agent.run(outcome_input, context)

    async def _run_documentation(
        self,
        request: ClinicalRequest,
        safety,
        diagnosis,
        plan,
        outcomes,
        evidence,
        context: AgentContext,
    ):
        """Run documentation generation."""
        doc_type = request.documentation_type or DocumentationType.INITIAL_EVAL

        doc_input = DocumentationInput(
            document_type=doc_type,
            patient=request.patient,
            safety=safety,
            diagnosis=diagnosis,
            plan=plan,
            outcomes=outcomes,
            evidence=evidence,
        )

        return await self.documentation_agent.run(doc_input, context)

    async def _run_qa_review(
        self,
        request: ClinicalRequest,
        safety,
        diagnosis,
        plan,
        evidence,
        context: AgentContext,
    ):
        """Run QA review."""
        qa_input = QAInput(
            patient=request.patient,
            safety=safety,
            diagnosis=diagnosis,
            plan=plan,
            evidence=evidence,
        )

        return await self.qa_agent.run(qa_input, context)

    def _create_emergency_response(
        self,
        safety: SafetyAssessment,
        request: ClinicalRequest,
    ) -> ConsultationResponse:
        """Create response for emergency/critical situations."""
        return ConsultationResponse(
            safety=safety,
            processing_notes=[
                "CRITICAL RED FLAGS DETECTED",
                "Treatment planning suspended",
                f"Referral recommended: {safety.referral_to or 'Emergency services'}",
            ],
            disclaimer=(
                "URGENT: Critical safety concerns identified. "
                "This patient requires immediate medical evaluation. "
                "Do not proceed with rehabilitation until cleared by appropriate provider."
            ),
        )

    async def run_single_agent(
        self,
        agent_name: str,
        inputs: dict,
        context: Optional[AgentContext] = None,
    ):
        """Run a single agent directly for testing or specialized queries.

        Args:
            agent_name: Name of agent to run
            inputs: Dict of inputs to convert to agent input model
            context: Optional agent context

        Returns:
            Agent output
        """
        context = context or AgentContext()

        agents = {
            "red_flag": (self.red_flag_agent, RedFlagInput),
            "diagnosis": (self.diagnosis_agent, DiagnosisInput),
            "evidence": (self.evidence_agent, EvidenceInput),
            "plan": (self.plan_agent, PlanInput),
            "outcome": (self.outcome_agent, OutcomeInput),
            "documentation": (self.documentation_agent, DocumentationInput),
            "qa": (self.qa_agent, QAInput),
        }

        if agent_name not in agents:
            raise ValueError(f"Unknown agent: {agent_name}. Available: {list(agents.keys())}")

        agent, input_model = agents[agent_name]
        typed_input = input_model.model_validate(inputs)
        return await agent.run(typed_input, context)
