"""API middleware for authentication and logging."""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from rehab_os.core.auth import ACCESS_COOKIE, decode_token

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"client={request.client.host if request.client else 'unknown'}"
        )

        # Process request
        response = await call_next(request)

        # Log response
        duration = time.time() - start_time
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"status={response.status_code} duration={duration:.3f}s"
        )

        # Add timing header
        response.headers["X-Process-Time"] = f"{duration:.3f}"

        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for health endpoints and auth routes
        skip_paths = ["/health", "/health/ready", "/health/live", "/docs", "/openapi.json"]
        path = request.url.path
        if path in skip_paths or path.startswith("/api/v1/auth/"):
            return await call_next(request)

        # If a valid JWT access cookie is present, pass through (no API key needed)
        token = request.cookies.get(ACCESS_COOKIE)
        if token:
            claims = decode_token(token)
            if claims and claims.get("type") == "access":
                return await call_next(request)

        # Check API key
        auth_header = request.headers.get("Authorization")
        api_key_header = request.headers.get("X-API-Key")

        provided_key = None
        if auth_header and auth_header.startswith("Bearer "):
            provided_key = auth_header[7:]
        elif api_key_header:
            provided_key = api_key_header

        if not provided_key or provided_key != self.api_key:
            logger.warning(
                f"Unauthorized request: {request.method} {request.url.path} "
                f"client={request.client.host if request.client else 'unknown'}"
            )
            return Response(
                content='{"error": "Unauthorized", "detail": "Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


class GuardrailMiddleware(BaseHTTPMiddleware):
    """Middleware for clinical guardrails and safety checks."""

    # Keywords that might indicate inappropriate use
    BLOCKED_PATTERNS = [
        "prescribe",
        "prescription",
        "drug dosage",
        "medication dose",
    ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only check POST requests to consult endpoints
        if request.method == "POST" and "/consult" in request.url.path:
            try:
                body = await request.body()
                body_text = body.decode("utf-8").lower()

                for pattern in self.BLOCKED_PATTERNS:
                    if pattern in body_text:
                        logger.warning(
                            f"Guardrail triggered: pattern='{pattern}' "
                            f"path={request.url.path}"
                        )
                        return Response(
                            content='{"error": "Request blocked", '
                            '"detail": "This system cannot provide medication prescriptions or dosages. '
                            'Please consult a physician for medication-related questions."}',
                            status_code=400,
                            media_type="application/json",
                        )

                # Re-inject the consumed body so downstream handlers can read it
                async def receive():
                    return {"type": "http.request", "body": body}

                request._receive = receive

            except Exception as e:
                logger.error(f"Guardrail check error: {e}")

        return await call_next(request)
