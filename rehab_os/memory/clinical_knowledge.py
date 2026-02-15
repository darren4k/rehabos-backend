"""
Clinical Knowledge Store via memU
=================================
Stores Medicare guidelines, skilled intervention phrases, and facility-specific
documentation templates as searchable memories in memU. When generating notes,
the LLM queries memU to retrieve relevant phrases and compliance requirements.

Namespace: rehab:knowledge
User ID pattern: rehab:knowledge:<category>

This separates clinical knowledge from patient data while keeping everything
in the same memU infrastructure.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Knowledge categories stored in memU
KNOWLEDGE_NAMESPACE = "rehab:knowledge"

CATEGORIES = {
    "medicare_guidelines": "Medicare coverage criteria, maintenance programs, medical necessity",
    "skilled_phrases": "Skilled intervention phrases organized by CPT code",
    "progress_indicators": "Progress/visit improvement indicator phrases",
    "billing_justification": "Skilled care justification language patterns",
}


async def seed_clinical_knowledge(memu_client) -> int:
    """
    Seed memU with clinical knowledge base.
    Idempotent — checks if already seeded before writing.

    Returns number of memories created.
    """
    from rehab_os.memory.clinical_phrases_data import (
        MEDICARE_GUIDELINES,
        SKILLED_INTERVENTION_PHRASES,
        PROGRESS_INDICATORS,
        BILLING_JUSTIFICATION,
    )

    count = 0
    user_id = f"{KNOWLEDGE_NAMESPACE}:reference"

    # Check if already seeded
    existing = await memu_client.recall(
        user_id=user_id,
        query="Medicare coverage criteria for rehabilitation therapy",
        top_k=1,
    )
    if existing and len(existing) > 0:
        logger.info("Clinical knowledge already seeded in memU, skipping")
        return 0

    # Seed Medicare guidelines
    for i, chunk in enumerate(MEDICARE_GUIDELINES):
        await memu_client.memorize(
            user_id=user_id,
            content=chunk,
            metadata={"category": "medicare_guidelines", "chunk": i},
        )
        count += 1

    # Seed skilled phrases by CPT
    for cpt_code, phrases in SKILLED_INTERVENTION_PHRASES.items():
        for subcategory, phrase_list in phrases.items():
            content = f"CPT {cpt_code} — {subcategory}:\n" + "\n".join(
                f"• {p}" for p in phrase_list
            )
            await memu_client.memorize(
                user_id=user_id,
                content=content,
                metadata={
                    "category": "skilled_phrases",
                    "cpt_code": cpt_code,
                    "subcategory": subcategory,
                },
            )
            count += 1

    # Seed progress indicators
    await memu_client.memorize(
        user_id=user_id,
        content="Progress/Visit Improvement Indicators:\n"
        + "\n".join(f"• {p}" for p in PROGRESS_INDICATORS),
        metadata={"category": "progress_indicators"},
    )
    count += 1

    # Seed billing justification patterns
    await memu_client.memorize(
        user_id=user_id,
        content=BILLING_JUSTIFICATION,
        metadata={"category": "billing_justification"},
    )
    count += 1

    logger.info(f"Seeded {count} clinical knowledge memories into memU")
    return count


async def recall_relevant_phrases(
    memu_client,
    transcript: str,
    top_k: int = 10,
    category: Optional[str] = None,
) -> list[dict]:
    """
    Query memU for clinical phrases relevant to a therapist's transcript.
    Returns ranked list of phrase chunks the LLM can use for note generation.
    """
    user_id = f"{KNOWLEDGE_NAMESPACE}:reference"

    results = await memu_client.recall(
        user_id=user_id,
        query=transcript,
        top_k=top_k,
    )

    if category:
        results = [
            r for r in results if r.get("metadata", {}).get("category") == category
        ]

    return results


async def recall_medicare_guidance(
    memu_client,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Query memU specifically for Medicare coverage guidance.
    Used when generating assessments and justifying skilled care.
    """
    return await recall_relevant_phrases(
        memu_client, query, top_k=top_k, category="medicare_guidelines"
    )
