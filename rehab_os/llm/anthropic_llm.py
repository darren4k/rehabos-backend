"""Anthropic Claude LLM implementation."""

import json
import logging
from typing import Any, Optional, TypeVar

from anthropic import AsyncAnthropic, APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel, ValidationError

from rehab_os.llm.base import (
    BaseLLM,
    LLMConnectionError,
    LLMOverloadError,
    LLMResponse,
    LLMTimeoutError,
    LLMValidationError,
    Message,
    MessageRole,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        timeout: int = 120,
    ):
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model name (e.g., claude-sonnet-4-20250514)
            timeout: Request timeout in seconds
        """
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "anthropic"

    def _prepare_messages(
        self, messages: list[Message]
    ) -> tuple[Optional[str], list[dict[str, str]]]:
        """Separate system message from conversation messages.

        Anthropic API requires system message as separate parameter.
        """
        system_content = None
        conversation = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
            else:
                conversation.append({"role": msg.role.value, "content": msg.content})

        return system_content, conversation

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion using Claude."""
        system_content, conversation = self._prepare_messages(messages)

        try:
            response = await self._client.messages.create(
                model=self._model,
                messages=conversation,
                system=system_content or "",
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop or [],
                **kwargs,
            )

            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                finish_reason=response.stop_reason,
                raw_response=response,
            )

        except APIConnectionError as e:
            logger.error(f"Anthropic connection error: {e}")
            raise LLMConnectionError("Failed to connect to Anthropic API") from e
        except APITimeoutError as e:
            logger.error(f"Anthropic timeout: {e}")
            raise LLMTimeoutError(f"Anthropic request timed out after {self._timeout}s") from e
        except RateLimitError as e:
            logger.warning(f"Anthropic rate limit: {e}")
            raise LLMOverloadError("Anthropic API rate limited") from e

    async def complete_structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> T:
        """Generate structured output from Claude."""
        # Add JSON schema instruction
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        structured_messages = list(messages)

        # Modify the last message to include schema
        last_msg = structured_messages[-1]
        schema_instruction = Message(
            role=last_msg.role,
            content=f"{last_msg.content}\n\nRespond with valid JSON matching this schema:\n```json\n{schema_json}\n```\n\nRespond ONLY with the JSON object, no other text.",
        )
        structured_messages[-1] = schema_instruction

        response = await self.complete(
            structured_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Parse JSON from response
        try:
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                # Remove first line (```json) and last line (```)
                if lines[-1].strip() == "```":
                    content = "\n".join(lines[1:-1])
                else:
                    content = "\n".join(lines[1:])

            data = json.loads(content)
            return schema.model_validate(data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Claude: {e}\nContent: {response.content}")
            raise LLMValidationError(f"Invalid JSON in response: {e}") from e
        except ValidationError as e:
            logger.error(f"Response doesn't match schema: {e}")
            raise LLMValidationError(f"Response doesn't match schema: {e}") from e

    async def health_check(self) -> bool:
        """Check if Anthropic API is available."""
        try:
            # Make a minimal request
            response = await self._client.messages.create(
                model=self._model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10,
            )
            return len(response.content) > 0
        except Exception as e:
            logger.debug(f"Anthropic health check failed: {e}")
            return False
