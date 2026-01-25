"""Abstract LLM interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypeVar

from pydantic import BaseModel


class MessageRole(str, Enum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Chat message."""

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to API-compatible dict."""
        return {"role": self.role.value, "content": self.content}


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get("input_tokens", 0) or self.usage.get("prompt_tokens", 0)

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get("output_tokens", 0) or self.usage.get("completion_tokens", 0)


T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM errors."""

    pass


class LLMConnectionError(LLMError):
    """Connection to LLM failed."""

    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out."""

    pass


class LLMOverloadError(LLMError):
    """LLM is overloaded."""

    pass


class LLMValidationError(LLMError):
    """Response failed validation."""

    pass


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion for messages.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResponse with generated content

        Raises:
            LLMConnectionError: If connection fails
            LLMTimeoutError: If request times out
            LLMOverloadError: If service is overloaded
        """
        pass

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> T:
        """Generate structured output matching a Pydantic schema.

        Args:
            messages: List of conversation messages
            schema: Pydantic model class for output
            temperature: Sampling temperature (lower for structured)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments

        Returns:
            Instance of schema class with generated data

        Raises:
            LLMValidationError: If response doesn't match schema
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM service is available.

        Returns:
            True if service is healthy, False otherwise
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name/identifier."""
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        """Get the provider name (e.g., 'local', 'anthropic')."""
        pass
