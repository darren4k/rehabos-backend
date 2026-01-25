"""PubMed/NCBI E-utilities client for literature search."""

import logging
from typing import Optional
from xml.etree import ElementTree

import httpx

from rehab_os.models.evidence import Citation, Evidence, EvidenceLevel

logger = logging.getLogger(__name__)

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedClient:
    """Client for searching PubMed via NCBI E-utilities.

    Provides access to medical literature for evidence retrieval.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize PubMed client.

        Args:
            api_key: NCBI API key (increases rate limit)
            email: Email for NCBI (required by their terms)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.email = email
        self.timeout = timeout

        self._client = httpx.AsyncClient(timeout=timeout)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        sort: str = "relevance",
    ) -> list[Evidence]:
        """Search PubMed for relevant articles.

        Args:
            query: Search query (supports PubMed syntax)
            max_results: Maximum number of results
            sort: Sort order ('relevance' or 'date')

        Returns:
            List of Evidence objects from PubMed
        """
        # Step 1: Search for PMIDs
        pmids = await self._search_pmids(query, max_results, sort)

        if not pmids:
            logger.info(f"No PubMed results for: {query}")
            return []

        # Step 2: Fetch article details
        articles = await self._fetch_articles(pmids)

        return articles

    async def _search_pmids(
        self,
        query: str,
        max_results: int,
        sort: str,
    ) -> list[str]:
        """Search PubMed and return PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort,
            "retmode": "json",
        }

        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email

        try:
            response = await self._client.get(
                f"{PUBMED_BASE_URL}/esearch.fcgi",
                params=params,
            )
            response.raise_for_status()

            data = response.json()
            return data.get("esearchresult", {}).get("idlist", [])

        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    async def _fetch_articles(self, pmids: list[str]) -> list[Evidence]:
        """Fetch article details for given PMIDs."""
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }

        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email

        try:
            response = await self._client.get(
                f"{PUBMED_BASE_URL}/efetch.fcgi",
                params=params,
            )
            response.raise_for_status()

            return self._parse_articles(response.text)

        except Exception as e:
            logger.error(f"PubMed fetch error: {e}")
            return []

    def _parse_articles(self, xml_text: str) -> list[Evidence]:
        """Parse PubMed XML response into Evidence objects."""
        evidence_list = []

        try:
            root = ElementTree.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                evidence = self._parse_single_article(article)
                if evidence:
                    evidence_list.append(evidence)

        except ElementTree.ParseError as e:
            logger.error(f"XML parse error: {e}")

        return evidence_list

    def _parse_single_article(self, article) -> Optional[Evidence]:
        """Parse a single PubMed article into Evidence."""
        try:
            # Get article metadata
            medline = article.find(".//MedlineCitation")
            if medline is None:
                return None

            pmid = medline.findtext("PMID", "")
            article_elem = medline.find(".//Article")

            if article_elem is None:
                return None

            # Title
            title = article_elem.findtext(".//ArticleTitle", "Untitled")

            # Authors
            authors = []
            for author in article_elem.findall(".//Author"):
                last = author.findtext("LastName", "")
                first = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{last} {first[0]}" if first else last)

            # Journal info
            journal = article_elem.find(".//Journal")
            journal_title = journal.findtext(".//Title", "") if journal else ""
            pub_date = journal.find(".//PubDate") if journal else None
            year = int(pub_date.findtext("Year", "0")) if pub_date is not None else None

            # Abstract
            abstract_elem = article_elem.find(".//Abstract")
            abstract = ""
            if abstract_elem is not None:
                abstract_parts = []
                for text in abstract_elem.findall(".//AbstractText"):
                    label = text.get("Label", "")
                    content = text.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {content}")
                    else:
                        abstract_parts.append(content)
                abstract = " ".join(abstract_parts)

            # Publication type for evidence level
            pub_types = [pt.text for pt in article_elem.findall(".//PublicationType")]
            evidence_level = self._determine_evidence_level(pub_types)

            # Create citation
            citation = Citation(
                title=title,
                authors=authors[:5],  # Limit authors
                journal=journal_title,
                year=year,
                pmid=pmid,
            )

            return Evidence(
                content=abstract or title,
                source=f"PubMed: {journal_title}",
                citation=citation,
                evidence_level=evidence_level,
            )

        except Exception as e:
            logger.error(f"Error parsing article: {e}")
            return None

    def _determine_evidence_level(self, pub_types: list[str]) -> EvidenceLevel:
        """Determine evidence level from publication types."""
        pub_types_lower = [pt.lower() for pt in pub_types]

        if any("systematic review" in pt or "meta-analysis" in pt for pt in pub_types_lower):
            return EvidenceLevel.LEVEL_1A

        if any("randomized controlled trial" in pt for pt in pub_types_lower):
            return EvidenceLevel.LEVEL_1B

        if any("cohort" in pt for pt in pub_types_lower):
            return EvidenceLevel.LEVEL_2B

        if any("case-control" in pt for pt in pub_types_lower):
            return EvidenceLevel.LEVEL_3B

        if any("case report" in pt or "case series" in pt for pt in pub_types_lower):
            return EvidenceLevel.LEVEL_4

        if any("practice guideline" in pt or "guideline" in pt for pt in pub_types_lower):
            return EvidenceLevel.CPG

        return EvidenceLevel.UNKNOWN

    async def search_rehabilitation(
        self,
        condition: str,
        discipline: str = "PT",
        max_results: int = 5,
    ) -> list[Evidence]:
        """Search for rehabilitation-specific evidence.

        Args:
            condition: Clinical condition
            discipline: PT, OT, or SLP
            max_results: Maximum results

        Returns:
            List of Evidence objects
        """
        discipline_terms = {
            "PT": "physical therapy OR physiotherapy",
            "OT": "occupational therapy",
            "SLP": "speech therapy OR speech-language pathology OR dysphagia",
        }

        query = f"({condition}) AND ({discipline_terms.get(discipline, 'rehabilitation')})"

        return await self.search(query, max_results)

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
