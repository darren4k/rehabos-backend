"""Base agent class for clinical reasoning agents."""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel

from rehab_os.llm import LLMRouter, Message, MessageRole
from rehab_os.observability import get_observability_logger

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ModelTier(str, Enum):
    """Model complexity tiers for agent tasks.

    Guides model selection based on task requirements:
    - FAST: Quick, simple tasks (safety screening rules, simple lookups)
    - STANDARD: Most clinical reasoning tasks (diagnosis, planning)
    - COMPLEX: Detailed synthesis requiring high capability (QA review, documentation)
    """

    FAST = "fast"  # Use smaller/faster model when available
    STANDARD = "standard"  # Default clinical reasoning model
    COMPLEX = "complex"  # Use most capable model


class AgentContext(BaseModel):
    """Context passed between agents."""

    session_id: Optional[str] = None
    discipline: str = "PT"
    setting: str = "outpatient"
    metadata: dict[str, Any] = {}
    prefer_fast_model: bool = False  # Hint to use faster model if available


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Abstract base class for clinical reasoning agents.

    Each agent follows a consistent pattern:
    1. Prepare context and system prompt
    2. Call LLM with structured output
    3. Validate and return typed result
    """

    def __init__(
        self,
        llm: LLMRouter,
        name: str,
        description: str,
    ):
        """Initialize agent.

        Args:
            llm: LLM router for model access
            name: Agent name for logging
            description: Brief description of agent's role
        """
        self.llm = llm
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"rehab_os.agents.{name}")

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> type[OutputT]:
        """Get the Pydantic schema for agent output."""
        pass

    @abstractmethod
    def format_input(self, inputs: InputT, context: AgentContext) -> str:
        """Format agent-specific input as user message.

        Args:
            inputs: Typed input data
            context: Shared agent context

        Returns:
            Formatted string for LLM user message
        """
        pass

    async def run(
        self,
        inputs: InputT,
        context: Optional[AgentContext] = None,
        request_id: Optional[str] = None,
    ) -> OutputT:
        """Execute the agent's reasoning task.

        Args:
            inputs: Agent-specific input data
            context: Optional shared context
            request_id: Optional request ID for observability

        Returns:
            Typed output matching output_schema
        """
        context = context or AgentContext()
        obs = get_observability_logger()
        request_id = request_id or obs.generate_request_id()

        # Generate input summary for observability (not full content)
        input_summary = self._generate_input_summary(inputs)

        with obs.agent_run(
            agent_name=self.name,
            agent_description=self.description,
            model_tier=self.model_tier.value,
            input_type=type(inputs).__name__,
            input_summary=input_summary,
            request_id=request_id,
        ) as event:
            self.logger.info(f"Running {self.name} agent")

            # Build messages
            messages = [
                Message(role=MessageRole.SYSTEM, content=self.system_prompt),
                Message(role=MessageRole.USER, content=self.format_input(inputs, context)),
            ]

            # Call LLM with structured output
            result = await self.llm.complete_structured(
                messages,
                self.output_schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Generate output summary for observability
            event.output_type = type(result).__name__
            event.output_summary = self._generate_output_summary(result)
            event.llm_calls = 1

            # Extract confidence and quality metrics from output
            confidence_info = self._extract_confidence_metrics(result)
            if confidence_info.get("confidence_score") is not None:
                event.confidence_score = confidence_info["confidence_score"]
            if confidence_info.get("evidence_quality") is not None:
                event.evidence_quality = confidence_info["evidence_quality"]
            if confidence_info.get("guideline_alignment") is not None:
                event.guideline_alignment = confidence_info["guideline_alignment"]
            if confidence_info.get("uncertainty_flags"):
                event.uncertainty_flags = confidence_info["uncertainty_flags"]

            self.logger.info(f"{self.name} agent completed successfully")
            return result

    def _generate_input_summary(self, inputs: InputT) -> str:
        """Generate a summary of inputs for observability."""
        # Override in subclasses for better summaries
        if hasattr(inputs, "patient"):
            patient = inputs.patient
            return f"{patient.age}yo {patient.sex}, {patient.chief_complaint[:50]}"
        return type(inputs).__name__

    def _generate_output_summary(self, output: OutputT) -> str:
        """Generate a summary of output for observability."""
        # Override in subclasses for better summaries
        if hasattr(output, "primary_diagnosis"):
            return f"Dx: {output.primary_diagnosis}"
        if hasattr(output, "is_safe_to_treat"):
            return f"Safe: {output.is_safe_to_treat}, Flags: {len(getattr(output, 'red_flags', []))}"
        if hasattr(output, "clinical_summary"):
            return output.clinical_summary[:100]
        return type(output).__name__

    def _extract_confidence_metrics(self, output: OutputT) -> dict[str, Any]:
        """Extract confidence and quality metrics from agent output.

        Override in subclasses for agent-specific extraction.

        Returns:
            Dict with optional keys: confidence_score, evidence_quality,
            guideline_alignment, uncertainty_flags
        """
        metrics: dict[str, Any] = {}

        # DiagnosisResult has confidence field
        if hasattr(output, "confidence"):
            metrics["confidence_score"] = output.confidence

        # QAResult has overall_quality
        if hasattr(output, "overall_quality"):
            metrics["confidence_score"] = output.overall_quality

        # EvidenceSummary has confidence
        if hasattr(output, "confidence") and hasattr(output, "total_sources"):
            metrics["evidence_quality"] = output.confidence

        # Extract uncertainties if present
        if hasattr(output, "uncertainties") and output.uncertainties:
            metrics["uncertainty_flags"] = output.uncertainties[:5]  # Limit to 5

        # SafetyAssessment can indicate uncertainty
        if hasattr(output, "red_flags") and hasattr(output, "is_safe_to_treat"):
            # More red flags with safe=True indicates uncertainty
            if output.is_safe_to_treat and len(output.red_flags) > 0:
                metrics["uncertainty_flags"] = [
                    f"Red flag: {rf.finding}" for rf in output.red_flags[:3]
                ]

        return metrics

    @property
    def temperature(self) -> float:
        """LLM temperature for this agent (override in subclass)."""
        return 0.3  # Lower temperature for clinical reasoning

    @property
    def max_tokens(self) -> int:
        """Max tokens for this agent (override in subclass)."""
        return 4096

    @property
    def model_tier(self) -> ModelTier:
        """Model complexity tier for this agent.

        Override in subclass to specify model requirements.
        Used by orchestrator to potentially route to different models.

        Returns:
            ModelTier indicating complexity requirements
        """
        return ModelTier.STANDARD

    async def validate_output(self, output: OutputT) -> bool:
        """Optional validation of agent output.

        Override in subclass for custom validation.
        """
        return True


class RuleBasedMixin:
    """Mixin for agents with rule-based pre-processing."""

    def apply_rules(self, inputs: Any) -> dict[str, Any]:
        """Apply deterministic rules before LLM call.

        Returns dict of findings from rules.
        Override in subclass.
        """
        return {}


class KnowledgeAwareMixin:
    """Mixin for agents that use knowledge base."""

    knowledge_base: Any  # Type hint for knowledge base

    async def retrieve_evidence(self, query: str, top_k: int = 5) -> list[Any]:
        """Retrieve relevant evidence from knowledge base."""
        if hasattr(self, "knowledge_base") and self.knowledge_base:
            return await self.knowledge_base.search(query, top_k=top_k)
        return []
