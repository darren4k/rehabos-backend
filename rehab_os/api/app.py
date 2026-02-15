"""FastAPI application for RehabOS."""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rehab_os import __version__
from rehab_os.api.middleware import APIKeyMiddleware, RequestLoggingMiddleware
from rehab_os.api.routes import consult, agents, health, feedback, sessions, streaming, mobile, knowledge, analyze, compliance, programs, scholar, chat, extract, notes, voice, intake, scheduling, patients
from rehab_os.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting RehabOS API")

    settings = get_settings()

    # Initialize components and store in app state
    from rehab_os.llm import create_router_from_settings
    from rehab_os.agents import Orchestrator
    from rehab_os.knowledge import VectorStore, initialize_knowledge_base

    # Initialize LLM router
    llm_router = create_router_from_settings()

    # Initialize knowledge base
    vector_store, guideline_repo = await initialize_knowledge_base(
        persist_dir=settings.chroma_persist_dir,
        load_samples=True,
    )

    # Initialize session memory
    from rehab_os.memory import SessionMemoryService

    session_memory = SessionMemoryService(enabled=settings.memu_enabled)

    # Initialize Patient-Core database
    from rehab_os.core.database import init_db

    try:
        await init_db()
        logger.info("Patient-Core database initialized")
    except Exception as e:
        logger.warning("Patient-Core database init failed (non-fatal): %s", e)

    # Initialize orchestrator
    orchestrator = Orchestrator(
        llm=llm_router,
        knowledge_base=vector_store,
        session_memory=session_memory,
    )

    # Store in app state
    app.state.llm_router = llm_router
    app.state.orchestrator = orchestrator
    app.state.vector_store = vector_store
    app.state.guideline_repo = guideline_repo
    app.state.session_memory = session_memory

    logger.info("RehabOS API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down RehabOS API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="RehabOS API",
        description="Multi-agent clinical reasoning system for PT/OT/SLP",
        version=__version__,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.add_middleware(RequestLoggingMiddleware)

    if settings.api_key:
        app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)

    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(consult.router, prefix="/api/v1", tags=["consult"])
    app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
    app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
    app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
    app.include_router(streaming.router, prefix="/api/v1", tags=["streaming"])
    app.include_router(mobile.router, prefix="/api/v1", tags=["mobile"])
    app.include_router(knowledge.router, prefix="/api/v1", tags=["knowledge"])
    app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])
    app.include_router(compliance.router, prefix="/api/v1", tags=["compliance"])
    app.include_router(programs.router, prefix="/api/v1", tags=["programs"])
    app.include_router(scholar.router, prefix="/api/v1", tags=["scholar"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    app.include_router(extract.router, prefix="/api/v1", tags=["extract"])
    app.include_router(notes.router, prefix="/api/v1", tags=["documentation"])
    app.include_router(voice.router, prefix="/api/v1", tags=["voice"])
    app.include_router(intake.router, prefix="/api/v1", tags=["intake"])
    app.include_router(scheduling.router, prefix="/api/v1", tags=["scheduling"])
    app.include_router(patients.router, prefix="/api/v1", tags=["patients"])

    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.debug_mode else None,
            },
        )

    return app


# Create app instance for uvicorn
app = create_app()
