"""ChromaDB vector store for clinical practice guidelines."""

import logging
import time
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from rehab_os.models.evidence import Evidence, EvidenceLevel
from rehab_os.observability import get_observability_logger

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store for clinical guidelines and evidence.

    Stores and retrieves clinical practice guidelines, research summaries,
    and other evidence documents using semantic similarity search.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = "clinical_guidelines",
    ):
        """Initialize vector store.

        Args:
            persist_directory: Directory for persistent storage (None for in-memory)
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB client
        if persist_directory:
            persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Clinical practice guidelines and evidence"},
        )

        logger.info(
            f"VectorStore initialized: {collection_name} "
            f"({self._collection.count()} documents)"
        )

    async def add_documents(
        self,
        documents: list[dict],
        ids: Optional[list[str]] = None,
    ) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents with 'content' and optional metadata
            ids: Optional list of IDs (auto-generated if not provided)
        """
        if not documents:
            return

        contents = [doc["content"] for doc in documents]
        metadatas = [
            {k: v for k, v in doc.items() if k != "content"} for doc in documents
        ]

        if ids is None:
            ids = [f"doc_{i}_{hash(contents[i])}" for i in range(len(documents))]

        self._collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"Added {len(documents)} documents to vector store")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[Evidence]:
        """Search for relevant documents.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of Evidence objects
        """
        start_time = time.time()

        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata,
        )

        evidence_list = []
        top_score = None
        top_source = None

        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else None

                # Convert distance to similarity score (ChromaDB uses L2 distance)
                relevance_score = 1 / (1 + distance) if distance is not None else None

                # Track top result for observability
                if i == 0:
                    top_score = relevance_score
                    top_source = metadata.get("source", "Unknown")

                evidence = Evidence(
                    content=doc,
                    source=metadata.get("source", "Unknown"),
                    evidence_level=EvidenceLevel(
                        metadata.get("evidence_level", "unknown")
                    ),
                    condition=metadata.get("condition"),
                    intervention=metadata.get("intervention"),
                    relevance_score=relevance_score,
                    chunk_id=results["ids"][0][i] if results["ids"] else None,
                )
                evidence_list.append(evidence)

        # Log search for observability
        duration_ms = (time.time() - start_time) * 1000
        obs = get_observability_logger()
        obs.log_knowledge_search(
            query=query,
            top_k=top_k,
            results_count=len(evidence_list),
            source_type="vector_store",
            top_result_score=top_score,
            top_result_source=top_source,
            duration_ms=duration_ms,
        )

        return evidence_list

    async def delete_collection(self) -> None:
        """Delete the entire collection."""
        self._client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")

    @property
    def count(self) -> int:
        """Get number of documents in the store."""
        return self._collection.count()

    async def get_by_condition(
        self,
        condition: str,
        top_k: int = 10,
    ) -> list[Evidence]:
        """Get evidence filtered by condition.

        Args:
            condition: Condition name to filter by
            top_k: Maximum results

        Returns:
            List of Evidence objects
        """
        return await self.search(
            query=condition,
            top_k=top_k,
            filter_metadata={"condition": condition},
        )

    async def get_by_discipline(
        self,
        discipline: str,
        query: str,
        top_k: int = 5,
    ) -> list[Evidence]:
        """Get evidence filtered by discipline.

        Args:
            discipline: PT, OT, or SLP
            query: Search query
            top_k: Maximum results

        Returns:
            List of Evidence objects
        """
        return await self.search(
            query=query,
            top_k=top_k,
            filter_metadata={"discipline": discipline},
        )
