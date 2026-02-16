"""LLM router with local-first routing and cloud fallback."""

import logging
from typing import Any, Optional, TypeVar

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rehab_os.llm.base import (
    BaseLLM,
    LLMConnectionError,
    LLMError,
    LLMOverloadError,
    LLMResponse,
    LLMTimeoutError,
    Message,
)
from rehab_os.observability import get_observability_logger

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMRouter:
    """Smart LLM router with local-first strategy and cloud fallback.

    Routes requests to local LLM first, falling back to cloud LLM
    on connection errors, timeouts, or overload conditions.
    """

    def __init__(
        self,
        primary: BaseLLM,
        fallback: Optional[BaseLLM] = None,
        max_retries: int = 3,
        always_try_primary: bool = True,
    ):
        """Initialize LLM router.

        Args:
            primary: Primary LLM (typically local)
            fallback: Fallback LLM (typically cloud/Anthropic)
            max_retries: Maximum retries per LLM
            always_try_primary: Always try primary first even after failures
        """
        self.primary = primary
        self.fallback = fallback
        self.max_retries = max_retries
        self.always_try_primary = always_try_primary

        self._primary_healthy = True
        self._consecutive_failures = 0
        self._failure_threshold = 3

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: Optional[list[str]] = None,
        prefer_fallback: bool = False,
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion with automatic failover.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            stop: Stop sequences
            prefer_fallback: Skip primary and use fallback directly
            request_id: Optional request ID for observability
            **kwargs: Additional arguments

        Returns:
            LLMResponse from whichever LLM succeeds
        """
        obs = get_observability_logger()
        request_id = request_id or obs.generate_request_id()

        if prefer_fallback and self.fallback:
            return await self._complete_with_observability(
                self.fallback, messages, temperature, max_tokens, stop, request_id, **kwargs
            )

        # Try primary first
        if self._should_try_primary():
            try:
                response = await self._complete_with_observability(
                    self.primary, messages, temperature, max_tokens, stop, request_id, **kwargs
                )
                self._record_success()
                return response
            except LLMError as e:
                self._record_failure()
                logger.warning(
                    f"Primary LLM ({self.primary.provider}) failed: {e}. "
                    f"{'Trying fallback...' if self.fallback else 'No fallback available.'}"
                )
                obs.log_llm_fallback(
                    from_provider=self.primary.provider,
                    to_provider=self.fallback.provider if self.fallback else "none",
                    reason=str(e),
                    request_id=request_id,
                )
                if not self.fallback:
                    raise

        # Try fallback
        if self.fallback:
            try:
                response = await self._complete_with_observability(
                    self.fallback, messages, temperature, max_tokens, stop, request_id,
                    is_fallback=True, **kwargs
                )
                logger.info(f"Fallback LLM ({self.fallback.provider}) succeeded")
                return response
            except LLMError as e:
                logger.error(f"Fallback LLM also failed: {e}")
                raise

        raise LLMError("No healthy LLM available")

    async def _complete_with_observability(
        self,
        llm: BaseLLM,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        stop: Optional[list[str]],
        request_id: str,
        is_fallback: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """Complete with observability logging."""
        obs = get_observability_logger()
        msg_dicts = [{"role": m.role.value, "content": m.content} for m in messages]

        with obs.llm_call(
            provider=llm.provider,
            model=llm.model_name,
            messages=msg_dicts,
            temperature=temperature,
            max_tokens=max_tokens,
            request_id=request_id,
        ) as event:
            event.is_fallback = is_fallback
            response = await self._try_with_retry(
                llm, messages, temperature, max_tokens, stop, **kwargs
            )
            event.response_content = response.content
            if response.usage:
                event.input_tokens = response.input_tokens
                event.output_tokens = response.output_tokens
                event.total_tokens = response.input_tokens + response.output_tokens
            return response

    async def _structured_with_observability(
        self,
        llm: BaseLLM,
        messages: list[Message],
        schema: type[T],
        temperature: float,
        max_tokens: int,
        request_id: str,
        is_fallback: bool = False,
        **kwargs: Any,
    ) -> T:
        """Complete structured call with observability logging."""
        obs = get_observability_logger()
        msg_dicts = [{"role": m.role.value, "content": m.content} for m in messages]

        with obs.llm_call(
            provider=llm.provider,
            model=llm.model_name,
            messages=msg_dicts,
            temperature=temperature,
            max_tokens=max_tokens,
            request_id=request_id,
        ) as event:
            event.is_fallback = is_fallback
            event.structured_schema = schema.__name__
            result = await llm.complete_structured(
                messages, schema, temperature=temperature, max_tokens=max_tokens, **kwargs
            )
            event.response_content = result.model_dump_json()
            return result

    async def complete_structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        prefer_fallback: bool = False,
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """Generate structured output with automatic failover."""
        obs = get_observability_logger()
        request_id = request_id or obs.generate_request_id()

        if prefer_fallback and self.fallback:
            return await self._structured_with_observability(
                self.fallback, messages, schema, temperature, max_tokens, request_id, **kwargs
            )

        # Try primary first
        if self._should_try_primary():
            try:
                result = await self._structured_with_observability(
                    self.primary, messages, schema, temperature, max_tokens, request_id, **kwargs
                )
                self._record_success()
                return result
            except LLMError as e:
                self._record_failure()
                logger.warning(
                    f"Primary LLM structured call failed: {e}. "
                    f"{'Trying fallback...' if self.fallback else 'No fallback.'}"
                )
                obs.log_llm_fallback(
                    from_provider=self.primary.provider,
                    to_provider=self.fallback.provider if self.fallback else "none",
                    reason=str(e),
                    request_id=request_id,
                )
                if not self.fallback:
                    raise

        # Try fallback
        if self.fallback:
            return await self._structured_with_observability(
                self.fallback, messages, schema, temperature, max_tokens, request_id,
                is_fallback=True, **kwargs
            )

        raise LLMError("No healthy LLM available for structured output")

    @retry(
        retry=retry_if_exception_type((LLMTimeoutError, LLMOverloadError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _try_with_retry(
        self,
        llm: BaseLLM,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        stop: Optional[list[str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Try LLM with retries for transient errors."""
        return await llm.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        )

    def _should_try_primary(self) -> bool:
        """Determine if primary should be attempted."""
        if self.always_try_primary:
            return True
        return self._primary_healthy

    def _record_success(self) -> None:
        """Record successful primary call."""
        self._primary_healthy = True
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        """Record failed primary call."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._primary_healthy = False
            logger.warning(
                f"Primary LLM marked unhealthy after {self._consecutive_failures} failures"
            )

    async def health_check(self) -> dict[str, bool]:
        """Check health of all LLMs."""
        result = {"primary": await self.primary.health_check()}
        if self.fallback:
            result["fallback"] = await self.fallback.health_check()
        return result

    @property
    def active_provider(self) -> str:
        """Get the currently preferred provider."""
        if self._primary_healthy:
            return self.primary.provider
        elif self.fallback:
            return self.fallback.provider
        return "none"


def create_router_from_settings() -> LLMRouter:
    """Create LLM router from application settings."""
    from rehab_os.config import get_settings
    from rehab_os.llm.local import LocalLLM
    from rehab_os.llm.anthropic_llm import AnthropicLLM

    settings = get_settings()

    # Create primary (local) LLM
    primary = LocalLLM(
        base_url=settings.local_llm_base_url,
        model=settings.local_llm_model,
        timeout=settings.local_llm_timeout,
    )

    # Create fallback (Anthropic) if configured
    fallback = None
    if settings.has_anthropic_key:
        fallback = AnthropicLLM(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )

    return LLMRouter(
        primary=primary,
        fallback=fallback,
        max_retries=settings.max_retries,
    )
