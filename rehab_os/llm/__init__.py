"""LLM abstraction layer with local-first routing and cloud fallback."""

from rehab_os.llm.base import BaseLLM, LLMResponse, Message, MessageRole, LLMError
from rehab_os.llm.local import LocalLLM
from rehab_os.llm.anthropic_llm import AnthropicLLM
from rehab_os.llm.router import LLMRouter, create_router_from_settings

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "Message",
    "MessageRole",
    "LLMError",
    "LocalLLM",
    "AnthropicLLM",
    "LLMRouter",
    "create_router_from_settings",
]
