"""Evidence Agent for clinical guideline and literature retrieval."""

import logging
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from rehab_os.agents.base import AgentContext, BaseAgent, KnowledgeAwareMixin
from rehab_os.llm import LLMRouter
from rehab_os.models.evidence import Evidence, EvidenceSummary, GuidelineRecommendation

logger = logging.getLogger(__name__)

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


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


async def _search_pubmed(query: str, max_results: int = 10) -> list[dict]:
    """Search PubMed for rehabilitation-relevant articles.

    Returns a list of dicts with keys: pmid, title, abstract, authors, journal, year.
    """
    results: list[dict] = []
    rehab_query = (
        f"({query}) AND (rehabilitation[MeSH] OR physical therapy[MeSH] "
        "OR occupational therapy[MeSH] OR speech therapy[MeSH]) "
        "AND (Clinical Trial[pt] OR Systematic Review[pt] OR Practice Guideline[pt])"
    )

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # Step 1: search for PMIDs
            search_resp = await client.get(
                f"{PUBMED_BASE_URL}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": rehab_query,
                    "retmax": max_results,
                    "retmode": "json",
                    "sort": "relevance",
                },
            )
            search_resp.raise_for_status()
            id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return results

            # Step 2: fetch summaries via esummary (lighter than efetch XML)
            summary_resp = await client.get(
                f"{PUBMED_BASE_URL}/esummary.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "json",
                },
            )
            summary_resp.raise_for_status()
            summaries = summary_resp.json().get("result", {})

            for pmid in id_list:
                info = summaries.get(pmid, {})
                if not isinstance(info, dict):
                    continue
                authors = [a.get("name", "") for a in info.get("authors", [])]
                results.append(
                    {
                        "pmid": pmid,
                        "title": info.get("title", f"Article {pmid}"),
                        "authors": authors,
                        "journal": info.get("fulljournalname", info.get("source", "")),
                        "year": info.get("pubdate", "")[:4],
                    }
                )
    except Exception as exc:
        logger.warning("PubMed search failed for query %r: %s", query, exc)

    return results


class EvidenceAgent(BaseAgent[EvidenceInput, EvidenceSummary], KnowledgeAwareMixin):
    """Agent for retrieving and synthesizing clinical evidence.

    Combines two evidence sources:
    1. **Local knowledge base** – Clinical Practice Guidelines stored in the
       vector store (high-quality, curated).
    2. **PubMed** – Recent literature via the NCBI E-Utilities API.

    Results are de-duplicated before being sent to the LLM for synthesis.
    """

    def __init__(
        self,
        llm: LLMRouter,
        knowledge_base: Optional[Any] = None,
        enable_pubmed: bool = True,
    ):
        super().__init__(
            llm=llm,
            name="evidence",
            description="Evidence retrieval and synthesis",
        )
        self.knowledge_base = knowledge_base
        self.enable_pubmed = enable_pubmed

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
        """Run evidence search with knowledge base + PubMed integration."""
        context = context or AgentContext()

        query = f"{inputs.condition} {inputs.clinical_question}"

        # --- Source 1: local vector-store (CPGs / guidelines) ---
        retrieved_evidence: list[Evidence] = []
        if self.knowledge_base:
            retrieved_evidence = await self.retrieve_evidence(query, top_k=10)

        # --- Source 2: PubMed literature ---
        pubmed_results: list[dict] = []
        if self.enable_pubmed:
            pubmed_results = await _search_pubmed(query, max_results=10)

        # --- De-duplicate across sources ---
        seen_titles: set[str] = set()
        deduped_evidence: list[Evidence] = []
        for ev in retrieved_evidence:
            norm = ev.source.strip().lower()
            if norm not in seen_titles:
                seen_titles.add(norm)
                deduped_evidence.append(ev)

        deduped_pubmed: list[dict] = []
        for pm in pubmed_results:
            norm = pm.get("title", "").strip().lower()
            if norm not in seen_titles:
                seen_titles.add(norm)
                deduped_pubmed.append(pm)

        # Format input with both sources
        formatted_input = self._format_with_evidence(
            inputs, deduped_evidence, context, pubmed_articles=deduped_pubmed
        )

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
        pubmed_articles: list[dict] | None = None,
    ) -> str:
        """Format input including retrieved evidence and PubMed articles."""
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

        # --- Local knowledge base evidence ---
        if retrieved_evidence:
            sections.extend(
                [
                    "",
                    "## Retrieved Evidence from Knowledge Base (Clinical Practice Guidelines)",
                    "",
                ]
            )
            for i, ev in enumerate(retrieved_evidence, 1):
                sections.append(f"### Source {i}: {ev.source}")
                sections.append(f"**Evidence Level:** {ev.evidence_level.value}")
                if ev.relevance_score:
                    sections.append(f"**Relevance:** {ev.relevance_score:.2f}")
                sections.append(f"\n{ev.content}\n")

        # --- PubMed literature ---
        if pubmed_articles:
            sections.extend(
                [
                    "",
                    "## Recent PubMed Literature",
                    "",
                ]
            )
            for i, article in enumerate(pubmed_articles, 1):
                authors_str = ", ".join(article.get("authors", [])[:3])
                if len(article.get("authors", [])) > 3:
                    authors_str += " et al."
                sections.append(
                    f"{i}. **{article.get('title', 'Untitled')}** — "
                    f"{authors_str} ({article.get('journal', 'N/A')}, "
                    f"{article.get('year', 'N/A')}) "
                    f"[PMID: {article.get('pmid', '')}]"
                )
            sections.append("")

        sections.extend(
            [
                "",
                "## Task",
                "Please synthesize ALL the available evidence (both local guidelines and PubMed literature) and provide:",
                "1. Key guideline recommendations (from CPGs)",
                "2. Supporting literature findings (from PubMed)",
                "3. Summary of evidence with quality ratings (Oxford CEBM levels)",
                "4. Practical clinical implications",
                "5. Any evidence gaps or limitations",
                "6. Note where guideline recommendations and recent literature agree or conflict",
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
