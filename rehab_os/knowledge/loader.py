"""Loader for ingesting clinical guidelines into vector store."""

import logging
from pathlib import Path
from typing import Optional

from rehab_os.knowledge.guidelines import (
    ClinicalPracticeGuideline,
    GuidelineRepository,
    SAMPLE_GUIDELINES,
)
from rehab_os.knowledge.vector_store import VectorStore

logger = logging.getLogger(__name__)


class GuidelineLoader:
    """Loader for ingesting clinical practice guidelines into the vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize loader.

        Args:
            vector_store: VectorStore instance
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def load_guideline(self, guideline: ClinicalPracticeGuideline) -> int:
        """Load a single guideline into the vector store.

        Args:
            guideline: Clinical practice guideline to load

        Returns:
            Number of chunks added
        """
        documents = []

        # Add summary as a document
        if guideline.summary:
            documents.append(
                {
                    "content": f"{guideline.title}\n\n{guideline.summary}",
                    "source": f"{guideline.source.abbreviation}: {guideline.title}",
                    "evidence_level": "cpg",
                    "condition": ",".join(guideline.conditions),
                    "discipline": ",".join(guideline.disciplines),
                    "type": "summary",
                }
            )

        # Add each recommendation as a document
        for rec in guideline.recommendations:
            content = f"""Recommendation from {guideline.source.abbreviation} {guideline.title}:

{rec.text}

Evidence Level: {rec.evidence_level.value}
Recommendation Strength: {rec.strength.value}
"""
            if rec.intervention:
                content += f"Intervention: {rec.intervention}\n"
            if rec.population:
                content += f"Population: {rec.population}\n"
            if rec.notes:
                content += f"Notes: {rec.notes}\n"

            documents.append(
                {
                    "content": content,
                    "source": f"{guideline.source.abbreviation}: {guideline.title}",
                    "evidence_level": rec.evidence_level.value,
                    "recommendation_strength": rec.strength.value,
                    "condition": ",".join(guideline.conditions),
                    "discipline": ",".join(guideline.disciplines),
                    "type": "recommendation",
                    "recommendation_id": rec.recommendation_id,
                }
            )

        if documents:
            ids = [
                f"{guideline.source.abbreviation}_{i}"
                for i in range(len(documents))
            ]
            await self.vector_store.add_documents(documents, ids)

        logger.info(
            f"Loaded {len(documents)} chunks from: {guideline.title}"
        )
        return len(documents)

    async def load_sample_guidelines(self) -> int:
        """Load the sample guidelines into the vector store.

        Returns:
            Total number of chunks added
        """
        total = 0
        for guideline in SAMPLE_GUIDELINES:
            count = await self.load_guideline(guideline)
            total += count

        logger.info(f"Loaded {total} total chunks from sample guidelines")
        return total

    async def load_from_directory(
        self,
        directory: Path,
        file_pattern: str = "*.md",
    ) -> int:
        """Load guidelines from markdown files in a directory.

        Expected format:
        - First line: Title
        - Second line: Source (e.g., "APTA" or "JOSPT")
        - Remaining: Content with recommendations

        Args:
            directory: Path to directory containing guideline files
            file_pattern: Glob pattern for files

        Returns:
            Total chunks added
        """
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return 0

        total = 0
        for file_path in directory.glob(file_pattern):
            try:
                count = await self._load_markdown_file(file_path)
                total += count
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        return total

    async def _load_markdown_file(self, file_path: Path) -> int:
        """Load a single markdown file as a guideline."""
        content = file_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        if len(lines) < 3:
            logger.warning(f"File too short: {file_path}")
            return 0

        title = lines[0].lstrip("# ").strip()
        source = lines[1].strip()
        body = "\n".join(lines[2:])

        # Chunk the content
        chunks = self._chunk_text(body)

        documents = []
        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "content": f"{title}\n\n{chunk}",
                    "source": f"{source}: {title}",
                    "evidence_level": "cpg",
                    "file": str(file_path.name),
                    "chunk_index": i,
                }
            )

        if documents:
            ids = [f"{file_path.stem}_{i}" for i in range(len(documents))]
            await self.vector_store.add_documents(documents, ids)

        logger.info(f"Loaded {len(documents)} chunks from: {file_path.name}")
        return len(documents)

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at paragraph or sentence boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.chunk_size // 2:
                    end = para_break

                # Otherwise look for sentence break
                elif (sent_break := text.rfind(". ", start, end)) > start + self.chunk_size // 2:
                    end = sent_break + 1

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return chunks


async def initialize_knowledge_base(
    persist_dir: Optional[Path] = None,
    load_samples: bool = True,
) -> tuple[VectorStore, GuidelineRepository]:
    """Initialize the knowledge base with vector store and guidelines.

    Args:
        persist_dir: Directory for vector store persistence
        load_samples: Whether to load sample guidelines

    Returns:
        Tuple of (VectorStore, GuidelineRepository)
    """
    # Initialize vector store
    vector_store = VectorStore(persist_directory=persist_dir)

    # Initialize guideline repository
    guideline_repo = GuidelineRepository()

    # Load samples if store is empty
    if load_samples and vector_store.count == 0:
        loader = GuidelineLoader(vector_store)
        await loader.load_sample_guidelines()

    return vector_store, guideline_repo
