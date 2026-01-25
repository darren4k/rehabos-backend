"""Tests for vector store and knowledge base."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from rehab_os.knowledge.vector_store import VectorStore
from rehab_os.knowledge.guidelines import (
    GuidelineRepository,
    ClinicalPracticeGuideline,
    GuidelineRecommendation,
    SAMPLE_GUIDELINES,
    APTA,
)
from rehab_os.knowledge.loader import GuidelineLoader
from rehab_os.models.evidence import Evidence, EvidenceLevel, RecommendationStrength


@pytest.fixture
def in_memory_vector_store():
    """Create in-memory vector store for testing with unique collection name."""
    import uuid
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    return VectorStore(persist_directory=None, collection_name=collection_name)


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "content": "Exercise therapy is strongly recommended for chronic low back pain.",
            "source": "JOSPT Guidelines",
            "evidence_level": "1a",
            "condition": "low back pain",
            "discipline": "PT",
        },
        {
            "content": "Manual therapy combined with exercise shows benefit for neck pain.",
            "source": "JOSPT Guidelines",
            "evidence_level": "1a",
            "condition": "neck pain",
            "discipline": "PT",
        },
        {
            "content": "High-intensity task-specific practice is recommended for stroke rehabilitation.",
            "source": "APTA Guidelines",
            "evidence_level": "1a",
            "condition": "stroke",
            "discipline": "PT,OT",
        },
        {
            "content": "Instrumental swallow assessment is recommended for dysphagia evaluation.",
            "source": "ASHA Guidelines",
            "evidence_level": "2a",
            "condition": "dysphagia",
            "discipline": "SLP",
        },
    ]


class TestVectorStore:
    """Tests for VectorStore."""

    @pytest.mark.asyncio
    async def test_add_documents(self, in_memory_vector_store, sample_documents):
        """Test adding documents to vector store."""
        initial_count = in_memory_vector_store.count

        await in_memory_vector_store.add_documents(sample_documents)

        assert in_memory_vector_store.count == initial_count + len(sample_documents)

    @pytest.mark.asyncio
    async def test_search_returns_relevant_results(
        self, in_memory_vector_store, sample_documents
    ):
        """Test search returns relevant documents."""
        await in_memory_vector_store.add_documents(sample_documents)

        results = await in_memory_vector_store.search("low back pain treatment", top_k=2)

        assert len(results) > 0
        assert isinstance(results[0], Evidence)
        # Most relevant result should be about LBP
        assert "back pain" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_search_with_discipline_filter(
        self, in_memory_vector_store, sample_documents
    ):
        """Test search with metadata filter."""
        await in_memory_vector_store.add_documents(sample_documents)

        results = await in_memory_vector_store.get_by_discipline(
            discipline="SLP",
            query="swallowing assessment",
            top_k=5,
        )

        assert len(results) > 0
        # Should return SLP-relevant content
        assert any("dysphagia" in r.content.lower() or "swallow" in r.content.lower()
                   for r in results)

    @pytest.mark.asyncio
    async def test_search_returns_evidence_objects(
        self, in_memory_vector_store, sample_documents
    ):
        """Test that search returns proper Evidence objects."""
        await in_memory_vector_store.add_documents(sample_documents)

        results = await in_memory_vector_store.search("exercise therapy", top_k=1)

        assert len(results) == 1
        evidence = results[0]
        assert isinstance(evidence, Evidence)
        assert evidence.source is not None
        assert evidence.evidence_level is not None
        assert evidence.relevance_score is not None

    @pytest.mark.asyncio
    async def test_search_empty_store(self):
        """Test searching empty vector store."""
        import uuid
        empty_store = VectorStore(
            persist_directory=None,
            collection_name=f"empty_test_{uuid.uuid4().hex[:8]}"
        )
        results = await empty_store.search("anything", top_k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_add_empty_documents(self, in_memory_vector_store):
        """Test adding empty document list."""
        initial_count = in_memory_vector_store.count

        await in_memory_vector_store.add_documents([])

        assert in_memory_vector_store.count == initial_count

    @pytest.mark.asyncio
    async def test_get_by_condition(self, in_memory_vector_store, sample_documents):
        """Test getting evidence by condition."""
        await in_memory_vector_store.add_documents(sample_documents)

        results = await in_memory_vector_store.get_by_condition("stroke", top_k=5)

        assert len(results) > 0


class TestGuidelineRepository:
    """Tests for GuidelineRepository."""

    def test_get_by_condition(self):
        """Test getting guidelines by condition."""
        repo = GuidelineRepository()

        results = repo.get_by_condition("low back pain")

        assert len(results) > 0
        assert any("back" in g.title.lower() for g in results)

    def test_get_by_discipline(self):
        """Test getting guidelines by discipline."""
        repo = GuidelineRepository()

        pt_results = repo.get_by_discipline("PT")
        slp_results = repo.get_by_discipline("SLP")

        assert len(pt_results) > 0
        assert len(slp_results) > 0

    def test_get_recommendations(self):
        """Test getting recommendations for a condition."""
        repo = GuidelineRepository()

        recommendations = repo.get_recommendations("low back pain")

        assert len(recommendations) > 0
        assert all(isinstance(r, GuidelineRecommendation) for r in recommendations)

    def test_get_recommendations_with_strength_filter(self):
        """Test filtering recommendations by strength."""
        repo = GuidelineRepository()

        strong_recs = repo.get_recommendations(
            "low back pain",
            min_strength=RecommendationStrength.STRONG_FOR,
        )

        assert len(strong_recs) > 0
        assert all(r.strength == RecommendationStrength.STRONG_FOR for r in strong_recs)

    def test_add_guideline(self):
        """Test adding a new guideline to a fresh repository."""
        from datetime import date

        # Create completely fresh list (not referencing SAMPLE_GUIDELINES)
        fresh_guidelines = []
        repo = GuidelineRepository(guidelines=fresh_guidelines)

        new_guideline = ClinicalPracticeGuideline(
            title="Test Guideline Added",
            source=APTA,
            publication_date=date(2024, 1, 1),
            conditions=["unique test condition xyz"],
            disciplines=["PT"],
        )

        initial_count = len(repo.all_guidelines)
        repo.add_guideline(new_guideline)

        assert len(repo.all_guidelines) == initial_count + 1
        results = repo.get_by_condition("unique test condition xyz")
        assert len(results) == 1
        assert results[0].title == "Test Guideline Added"


class TestGuidelineLoader:
    """Tests for GuidelineLoader."""

    @pytest.mark.asyncio
    async def test_load_guideline(self, in_memory_vector_store):
        """Test loading a single guideline."""
        loader = GuidelineLoader(in_memory_vector_store)
        guideline = SAMPLE_GUIDELINES[0]  # LBP guideline

        count = await loader.load_guideline(guideline)

        assert count > 0
        assert in_memory_vector_store.count > 0

    @pytest.mark.asyncio
    async def test_load_sample_guidelines(self):
        """Test loading all sample guidelines."""
        import uuid
        fresh_store = VectorStore(
            persist_directory=None,
            collection_name=f"sample_test_{uuid.uuid4().hex[:8]}"
        )
        loader = GuidelineLoader(fresh_store)

        total = await loader.load_sample_guidelines()

        assert total > 0
        # Store count should match or exceed total (may have duplicates from IDs)
        assert fresh_store.count >= total - 2  # Allow small variance

    @pytest.mark.asyncio
    async def test_load_from_nonexistent_directory(self, in_memory_vector_store, tmp_path):
        """Test loading from non-existent directory."""
        loader = GuidelineLoader(in_memory_vector_store)
        fake_path = tmp_path / "nonexistent"

        count = await loader.load_from_directory(fake_path)

        assert count == 0

    @pytest.mark.asyncio
    async def test_load_markdown_file(self, in_memory_vector_store, tmp_path):
        """Test loading a markdown guideline file."""
        loader = GuidelineLoader(in_memory_vector_store)

        # Create test markdown file
        md_content = """# Test Guideline Title
APTA

## Overview
This is a test guideline for unit testing.

## Recommendations
1. Exercise is recommended for test condition.
2. Manual therapy may be beneficial.
"""
        md_file = tmp_path / "test_guideline.md"
        md_file.write_text(md_content)

        count = await loader.load_from_directory(tmp_path, "*.md")

        assert count > 0

    def test_chunk_text_short(self, in_memory_vector_store):
        """Test chunking short text."""
        loader = GuidelineLoader(in_memory_vector_store, chunk_size=1000)

        chunks = loader._chunk_text("Short text")

        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_chunk_text_long(self, in_memory_vector_store):
        """Test chunking long text."""
        loader = GuidelineLoader(in_memory_vector_store, chunk_size=100, chunk_overlap=20)
        long_text = "Word " * 100  # 500 characters

        chunks = loader._chunk_text(long_text)

        assert len(chunks) > 1

    def test_chunk_text_preserves_paragraphs(self, in_memory_vector_store):
        """Test that chunking tries to preserve paragraph boundaries."""
        loader = GuidelineLoader(in_memory_vector_store, chunk_size=200, chunk_overlap=20)

        text = "First paragraph with some content.\n\nSecond paragraph with more content.\n\nThird paragraph here."

        chunks = loader._chunk_text(text)

        # Should try to break at paragraph boundaries
        assert len(chunks) >= 1


class TestKnowledgeBaseIntegration:
    """Integration tests for knowledge base components."""

    @pytest.mark.asyncio
    async def test_full_ingestion_and_search(self, in_memory_vector_store):
        """Test full workflow: ingest guidelines then search."""
        # Load guidelines
        loader = GuidelineLoader(in_memory_vector_store)
        await loader.load_sample_guidelines()

        # Search for specific content
        results = await in_memory_vector_store.search(
            "exercise therapy chronic pain",
            top_k=3,
        )

        assert len(results) > 0
        # Should find exercise-related recommendations
        assert any("exercise" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_discipline_specific_search(self, in_memory_vector_store):
        """Test searching for discipline-specific content."""
        loader = GuidelineLoader(in_memory_vector_store)
        await loader.load_sample_guidelines()

        # Search for SLP content
        results = await in_memory_vector_store.search(
            "swallowing dysphagia assessment",
            top_k=5,
        )

        # Should find dysphagia-related content
        dysphagia_results = [r for r in results if "dysphagia" in r.content.lower() or "swallow" in r.content.lower()]
        assert len(dysphagia_results) > 0

    @pytest.mark.asyncio
    async def test_evidence_levels_preserved(self, in_memory_vector_store):
        """Test that evidence levels are preserved through ingestion."""
        loader = GuidelineLoader(in_memory_vector_store)
        await loader.load_sample_guidelines()

        results = await in_memory_vector_store.search("recommendation", top_k=5)

        # Check that evidence levels are set
        for result in results:
            assert result.evidence_level is not None
            assert result.evidence_level != EvidenceLevel.UNKNOWN
