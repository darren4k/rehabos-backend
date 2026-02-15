"""Full intake pipeline: document → text → structured patient profile."""

import logging
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.core.repository import PatientRepository
from rehab_os.core.schemas import PatientCreate
from rehab_os.intake.agent import IntakeAgent, IntakeInput, IntakeResult
from rehab_os.intake.extractor import extract_text
from rehab_os.llm.router import LLMRouter
from rehab_os.memory.session_memory import SessionMemoryService
from rehab_os.models.patient import PatientContext

logger = logging.getLogger(__name__)


def _estimate_dob(age: int) -> "date":
    """Estimate a date of birth from age (uses Jan 1 of birth year)."""
    from datetime import date

    return date(date.today().year - age, 1, 1)


class IntakePipeline:
    """End-to-end pipeline for processing referral documents."""

    def __init__(
        self,
        llm: LLMRouter,
        session_memory: Optional[SessionMemoryService] = None,
        db_session: Optional[AsyncSession] = None,
    ):
        self.llm = llm
        self.session_memory = session_memory
        self.db_session = db_session
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

        # Step 4: Create patient record in Patient-Core if db available
        patient_id = await self._create_patient_record(result.patient)
        if patient_id:
            result.patient_id = str(patient_id)

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

        patient_id = await self._create_patient_record(result.patient)
        if patient_id:
            result.patient_id = str(patient_id)

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

    async def _create_patient_record(self, patient: PatientContext) -> Optional[uuid.UUID]:
        """Create or find a patient record in Patient-Core.

        Maps PatientContext fields to PatientCreate schema and persists via
        PatientRepository. If a patient with matching name+dob already exists,
        returns the existing patient ID instead.

        Returns:
            The patient UUID, or None if no db session or insufficient data.
        """
        if not self.db_session:
            return None

        if not patient.name:
            logger.info("No patient name in context; skipping patient record creation")
            return None

        # Split name into first/last
        name_parts = patient.name.strip().split(None, 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""

        repo = PatientRepository(self.db_session)

        # Check for duplicates by name (and dob if available)
        try:
            existing = await repo.search_by_name(first_name, limit=50)
            for p in existing:
                if (
                    p.first_name.lower() == first_name.lower()
                    and p.last_name.lower() == last_name.lower()
                    and (patient.date_of_birth is None or p.dob == patient.date_of_birth)
                ):
                    logger.info("Found existing patient %s, reusing", p.id)
                    return p.id
        except Exception as e:
            logger.warning("Duplicate check failed: %s", e)

        # Map sex value
        sex = patient.sex if patient.sex in ("male", "female", "other") else "other"

        # Build create payload
        create_data = PatientCreate(
            first_name=first_name,
            last_name=last_name,
            dob=patient.date_of_birth or _estimate_dob(patient.age),
            sex=sex,
        )

        try:
            record = await repo.create(**create_data.model_dump())
            await self.db_session.flush()
            logger.info("Created patient record %s", record.id)
            return record.id
        except Exception as e:
            logger.error("Failed to create patient record: %s", e)
            return None

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
