"""Full intake pipeline: document → text → structured patient profile."""

import logging
from typing import Optional

from rehab_os.intake.agent import IntakeAgent, IntakeInput, IntakeResult
from rehab_os.intake.extractor import extract_text
from rehab_os.llm.router import LLMRouter
from rehab_os.memory.session_memory import SessionMemoryService

logger = logging.getLogger(__name__)


class IntakePipeline:
    """End-to-end pipeline for processing referral documents."""

    def __init__(
        self,
        llm: LLMRouter,
        session_memory: Optional[SessionMemoryService] = None,
    ):
        self.llm = llm
        self.session_memory = session_memory
        self.agent = IntakeAgent(llm)

    async def process_referral(
        self,
        file_bytes: bytes,
        content_type: str,
        metadata: dict = {},
    ) -> IntakeResult:
        """Process a referral document file through the full pipeline.

        Args:
            file_bytes: Raw file bytes (PDF or image).
            content_type: MIME type of the file.
            metadata: Optional metadata (referring_provider, received_date, etc.).

        Returns:
            Structured IntakeResult with patient profile.
        """
        # Step 1: Extract text
        logger.info("Extracting text from %s document", content_type)
        raw_text = extract_text(file_bytes, content_type)

        # Step 2: Run through agent
        result = await self._run_agent(
            raw_text=raw_text,
            source_type=metadata.get("source_type", "referral"),
            referring_provider=metadata.get("referring_provider"),
            received_date=metadata.get("received_date"),
        )

        # Step 3: Store in session memory if available
        self._store_patient(result)

        return result

    async def process_raw_text(
        self,
        text: str,
        source_type: str = "referral",
    ) -> IntakeResult:
        """Process already-extracted text through the intake agent.

        Args:
            text: Pre-extracted document text.
            source_type: Type of source document.

        Returns:
            Structured IntakeResult with patient profile.
        """
        result = await self._run_agent(raw_text=text, source_type=source_type)
        self._store_patient(result)
        return result

    async def _run_agent(
        self,
        raw_text: str,
        source_type: str = "referral",
        referring_provider: Optional[str] = None,
        received_date=None,
    ) -> IntakeResult:
        """Run the intake agent on extracted text."""
        intake_input = IntakeInput(
            raw_text=raw_text,
            source_type=source_type,
            referring_provider=referring_provider,
            received_date=received_date,
        )

        logger.info("Running intake agent on %s text (%d chars)", source_type, len(raw_text))
        return await self.agent.run(intake_input)

    def _store_patient(self, result: IntakeResult) -> None:
        """Store patient profile in session memory if available."""
        if not self.session_memory:
            return

        try:
            # Use a deterministic ID from patient demographics if possible
            patient_id = f"intake-{hash(result.raw_text_snippet) % 100000:05d}"
            record = {
                "timestamp": None,  # Will be set by session memory
                "source": "intake",
                "referral_summary": result.referral_summary,
                "confidence": result.extraction_confidence,
                "missing_fields": result.missing_fields,
            }
            self.session_memory._cache.setdefault(patient_id, []).append(record)
            logger.info("Stored intake result for patient %s", patient_id)
        except Exception as e:
            logger.warning("Failed to store intake result: %s", e)
