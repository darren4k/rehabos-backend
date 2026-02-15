"""In-memory sliding-window rate limiter for FastAPI."""

import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request


class SlidingWindowRateLimiter:
    """Sliding-window rate limiter keyed by API key (or client IP as fallback).

    Usage as a FastAPI dependency::

        limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=60)

        @router.post("/consult")
        async def consult(request: Request, _=Depends(limiter)):
            ...
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # key -> list of timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _client_key(self, request: Request) -> str:
        """Extract a rate-limit key from the request (API key or IP)."""
        # Prefer the API key header used by the auth middleware
        api_key: Optional[str] = request.headers.get("x-api-key")
        if api_key:
            return f"key:{api_key}"
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"

    def _prune(self, key: str, now: float) -> None:
        """Remove timestamps outside the current window."""
        cutoff = now - self.window_seconds
        timestamps = self._requests[key]
        # Find first index within window (timestamps are in order)
        i = 0
        while i < len(timestamps) and timestamps[i] < cutoff:
            i += 1
        if i:
            self._requests[key] = timestamps[i:]

    async def __call__(self, request: Request) -> None:
        key = self._client_key(request)
        now = time.monotonic()
        self._prune(key, now)

        if len(self._requests[key]) >= self.max_requests:
            retry_after = int(
                self.window_seconds
                - (now - self._requests[key][0])
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s.",
                headers={"Retry-After": str(max(retry_after, 1))},
            )

        self._requests[key].append(now)
