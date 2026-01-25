"""Health check endpoints."""

from fastapi import APIRouter, Request
from rehab_os import __version__

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> dict:
    """Basic health check."""
    return {
        "status": "healthy",
        "service": "rehab-os",
    }


@router.get("/health/ready")
async def readiness_check(request: Request) -> dict:
    """Readiness check - verifies all dependencies are available."""
    errors = []

    # Check LLM
    try:
        llm_health = await request.app.state.llm_router.health_check()
        if not any(llm_health.values()):
            errors.append("No LLM available")
    except Exception as e:
        errors.append(f"LLM check failed: {e}")

    # Check vector store
    try:
        doc_count = request.app.state.vector_store.count
        if doc_count == 0:
            errors.append("Vector store empty")
    except Exception as e:
        errors.append(f"Vector store check failed: {e}")

    if errors:
        return {
            "status": "not_ready",
            "errors": errors,
        }

    return {
        "status": "ready",
        "llm": llm_health,
        "knowledge_base_docs": doc_count,
    }


@router.get("/health/live")
async def liveness_check() -> dict:
    """Liveness check - basic process health."""
    return {"status": "alive"}


@router.get("/api-info")
async def api_info(request: Request) -> dict:
    """API information for frontend integration.

    Returns API metadata useful for configuring Lovable or other
    no-code platforms.
    """
    base_url = str(request.base_url).rstrip("/")

    return {
        "name": "RehabOS API",
        "version": __version__,
        "description": "Multi-agent clinical reasoning system for PT/OT/SLP",
        "base_url": base_url,
        "openapi_url": f"{base_url}/openapi.json",
        "docs_url": f"{base_url}/docs",
        "endpoints": {
            "consult": {
                "url": "/api/v1/consult",
                "method": "POST",
                "description": "Full clinical consultation",
            },
            "quick_consult": {
                "url": "/api/v1/mobile/quick-consult",
                "method": "POST",
                "description": "Quick mobile consultation",
            },
            "safety_check": {
                "url": "/api/v1/mobile/safety-check",
                "method": "POST",
                "description": "Safety screening endpoint",
            },
            "hep": {
                "url": "/api/v1/mobile/hep",
                "method": "POST",
                "description": "Home exercise program generation",
            },
            "sessions": {
                "create": "/api/v1/sessions/create",
                "get": "/api/v1/sessions/{session_id}",
                "history": "/api/v1/sessions/{session_id}/history",
            },
            "streaming": {
                "websocket": "/api/v1/ws/consult/{client_id}",
                "description": "WebSocket for real-time consultation updates",
            },
            "disciplines": "/api/v1/mobile/disciplines",
            "settings": "/api/v1/mobile/settings",
        },
        "authentication": {
            "type": "api_key",
            "header": "X-API-Key",
            "required": False,  # Will be True in production
        },
        "cors": {
            "enabled": True,
            "origins": ["*"],
        },
    }
