"""Main entry point for RehabOS."""

import asyncio
import logging
import sys

from rehab_os.config import get_settings


def setup_logging():
    """Configure logging based on settings."""
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """Main entry point - runs CLI."""
    setup_logging()

    from rehab_os.cli.commands import app

    app()


async def run_consultation(query: str, **kwargs):
    """Programmatic API for running consultations.

    Example:
        import asyncio
        from rehab_os.main import run_consultation

        result = asyncio.run(run_consultation(
            "68yo male s/p L TKA POD 2, evaluate for PT",
            discipline="PT",
            setting="inpatient",
        ))
    """
    from rehab_os.llm import create_router_from_settings
    from rehab_os.agents import Orchestrator
    from rehab_os.knowledge import VectorStore, initialize_knowledge_base
    from rehab_os.models.output import ClinicalRequest
    from rehab_os.models.patient import PatientContext, Discipline, CareSetting

    setup_logging()
    settings = get_settings()

    # Initialize components
    llm = create_router_from_settings()
    vector_store, _ = await initialize_knowledge_base(
        persist_dir=settings.chroma_persist_dir,
    )
    orchestrator = Orchestrator(llm=llm, knowledge_base=vector_store)

    # Parse options
    discipline = Discipline(kwargs.get("discipline", "PT"))
    setting = CareSetting(kwargs.get("setting", "outpatient"))

    # Create patient context
    patient = PatientContext(
        age=kwargs.get("age", 50),
        sex=kwargs.get("sex", "other"),
        chief_complaint=query,
        discipline=discipline,
        setting=setting,
    )

    # Create request
    request = ClinicalRequest(
        query=query,
        patient=patient,
        discipline=discipline,
        setting=setting,
        task_type=kwargs.get("task_type", "full_consult"),
        include_documentation=kwargs.get("include_documentation", False),
    )

    # Run consultation
    return await orchestrator.process(request)


if __name__ == "__main__":
    main()
