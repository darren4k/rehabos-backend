"""Local LLM implementation using OpenAI-compatible API."""

import json
import logging
from typing import Any, Optional, TypeVar

import httpx
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel, ValidationError

from rehab_os.llm.base import (
    BaseLLM,
    LLMConnectionError,
    LLMOverloadError,
    LLMResponse,
    LLMTimeoutError,
    LLMValidationError,
    Message,
    parse_structured_response,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LocalLLM(BaseLLM):
    """Local LLM via OpenAI-compatible API (vLLM, Ollama, TGI, etc.)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 60,
        api_key: str = "not-needed",  # Many local servers don't require a key
    ):
        """Initialize local LLM client.

        Args:
            base_url: OpenAI-compatible API endpoint (e.g., http://localhost:8000/v1)
            model: Model name/path
            timeout: Request timeout in seconds
            api_key: API key (often not required for local)
        """
        self._base_url = base_url
        self._model = model
        self._timeout = timeout

        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "local"

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion using local LLM."""
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs,
            )

            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                finish_reason=choice.finish_reason,
                raw_response=response,
            )

        except APIConnectionError as e:
            logger.error(f"Local LLM connection error: {e}")
            raise LLMConnectionError(f"Failed to connect to local LLM at {self._base_url}") from e
        except APITimeoutError as e:
            logger.error(f"Local LLM timeout: {e}")
            raise LLMTimeoutError(f"Local LLM request timed out after {self._timeout}s") from e
        except RateLimitError as e:
            logger.warning(f"Local LLM overloaded: {e}")
            raise LLMOverloadError("Local LLM is overloaded") from e

    async def complete_structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> T:
        """Generate structured output from local LLM."""
        # Add JSON schema instruction to system message
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        structured_messages = list(messages)

        # Append schema instruction
        schema_instruction = Message(
            role=messages[-1].role,
            content=f"{messages[-1].content}\n\nRespond with valid JSON matching this schema:\n```json\n{schema_json}\n```\n\nRespond ONLY with the JSON object, no other text.",
        )
        structured_messages[-1] = schema_instruction

        response = await self.complete(
            structured_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return parse_structured_response(response.content, schema)

    async def health_check(self) -> bool:
        """Check if local LLM is available."""
        try:
            # Try to list models or make a simple request
            await self._client.models.list()
            return True
        except Exception as e:
            logger.debug(f"Local LLM health check failed: {e}")
            return False
