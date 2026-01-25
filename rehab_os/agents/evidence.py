"""Evidence Agent for clinical guideline and literature retrieval."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from rehab_os.agents.base import AgentContext, BaseAgent, KnowledgeAwareMixin
from rehab_os.llm import LLMRouter
from rehab_os.models.evidence import Evidence, EvidenceSummary, GuidelineRecommendation


class EvidenceInput(BaseModel):
    """Input for evidence search."""

    condition: str = Field(..., description="Primary condition or diagnosis")
    clinical_question: str = Field(..., description="Specific clinical question")
    patient_population: Optional[str] = Field(
        None, description="Specific population characteristics"
    )
    intervention_focus: Optional[str] = Field(
        None, description="Specific intervention being considered"
    )
    setting: Optional[str] = None


class EvidenceAgent(BaseAgent[EvidenceInput, EvidenceSummary], KnowledgeAwareMixin):
    """Agent for retrieving and synthesizing clinical evidence.

    Searches local knowledge base (CPGs) and optionally PubMed
    to gather evidence supporting clinical decisions.
    """

    def __init__(
        self,
        llm: LLMRouter,
        knowledge_base: Optional[Any] = None,
    ):
        super().__init__(
            llm=llm,
            name="evidence",
            description="Evidence retrieval and synthesis",
        )
        self.knowledge_base = knowledge_base

    @property
    def system_prompt(self) -> str:
        return """You are a clinical evidence synthesis agent for rehabilitation services.

Your role is to:
1. Identify relevant clinical practice guidelines
2. Summarize the best available evidence
3. Rate evidence quality and recommendation strength
4. Synthesize findings into actionable guidance

Evidence Levels (Oxford CEBM):
- 1a: Systematic review of RCTs
- 1b: Individual RCT
- 2a: Systematic review of cohort studies
- 2b: Individual cohort study
- 3a: Systematic review of case-control studies
- 3b: Individual case-control study
- 4: Case series
- 5: Expert opinion

Recommendation Strength:
- Strong for: Consistent high-quality evidence supporting intervention
- Moderate for: Moderate-quality evidence or some inconsistency
- Weak for: Low-quality evidence but likely benefit
- Neutral: Insufficient evidence or balanced benefit/risk
- Against: Evidence suggests intervention not beneficial or harmful

When synthesizing evidence:
- Prioritize clinical practice guidelines from APTA, AOTA, ASHA
- Note any population-specific considerations
- Identify gaps in evidence
- Highlight practical clinical implications

If you don't have specific evidence on a topic, acknowledge the limitation
and provide the best available guidance based on clinical principles."""

    @property
    def output_schema(self) -> type[EvidenceSummary]:
        return EvidenceSummary

    async def run(
        self,
        inputs: EvidenceInput,
        context: Optional[AgentContext] = None,
    ) -> EvidenceSummary:
        """Run evidence search with knowledge base integration."""
        context = context or AgentContext()

        # Retrieve from knowledge base if available
        retrieved_evidence: list[Evidence] = []
        if self.knowledge_base:
            query = f"{inputs.condition} {inputs.clinical_question}"
            retrieved_evidence = await self.retrieve_evidence(query, top_k=10)

        # Format input with retrieved evidence
        formatted_input = self._format_with_evidence(inputs, retrieved_evidence, context)

        from rehab_os.llm import Message, MessageRole

        messages = [
            Message(role=MessageRole.SYSTEM, content=self.system_prompt),
            Message(role=MessageRole.USER, content=formatted_input),
        ]

        result = await self.llm.complete_structured(
            messages,
            self.output_schema,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return result

    def format_input(self, inputs: EvidenceInput, context: AgentContext) -> str:
        """Basic format without retrieved evidence."""
        return self._format_with_evidence(inputs, [], context)

    def _format_with_evidence(
        self,
        inputs: EvidenceInput,
        retrieved_evidence: list[Evidence],
        context: AgentContext,
    ) -> str:
        """Format input including retrieved evidence."""
        sections = [
            "## Evidence Request",
            "",
            f"**Condition:** {inputs.condition}",
            f"**Clinical Question:** {inputs.clinical_question}",
        ]

        if inputs.patient_population:
            sections.append(f"**Population:** {inputs.patient_population}")

        if inputs.intervention_focus:
            sections.append(f"**Intervention Focus:** {inputs.intervention_focus}")

        if inputs.setting:
            sections.append(f"**Setting:** {inputs.setting}")

        sections.append(f"**Discipline:** {context.discipline}")

        if retrieved_evidence:
            sections.extend(
                [
                    "",
                    "## Retrieved Evidence from Knowledge Base",
                    "",
                ]
            )
            for i, ev in enumerate(retrieved_evidence, 1):
                sections.append(f"### Source {i}: {ev.source}")
                sections.append(f"**Evidence Level:** {ev.evidence_level.value}")
                if ev.relevance_score:
                    sections.append(f"**Relevance:** {ev.relevance_score:.2f}")
                sections.append(f"\n{ev.content}\n")

        sections.extend(
            [
                "",
                "## Task",
                "Please synthesize the available evidence and provide:",
                "1. Key guideline recommendations",
                "2. Summary of evidence with quality ratings",
                "3. Practical clinical implications",
                "4. Any evidence gaps or limitations",
            ]
        )

        return "\n".join(sections)

    @property
    def temperature(self) -> float:
        return 0.3

    @property
    def max_tokens(self) -> int:
        return 6000  # Evidence summaries can be longer

    @property
    def model_tier(self):
        from rehab_os.agents.base import ModelTier
        return ModelTier.FAST  # Evidence retrieval is mostly search + summarization
