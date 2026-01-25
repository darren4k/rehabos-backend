"""Tests for LLM router with fallback logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from rehab_os.llm.base import (
    BaseLLM,
    LLMResponse,
    LLMConnectionError,
    LLMTimeoutError,
    LLMOverloadError,
    LLMValidationError,
    Message,
    MessageRole,
)
from rehab_os.llm.router import LLMRouter


class MockSchema(BaseModel):
    """Simple schema for testing structured output."""
    result: str
    confidence: float


@pytest.fixture
def mock_primary_llm():
    """Create mock primary (local) LLM."""
    llm = MagicMock(spec=BaseLLM)
    llm.complete = AsyncMock()
    llm.complete_structured = AsyncMock()
    llm.health_check = AsyncMock(return_value=True)
    llm.model_name = "local-model"
    llm.model = "local-model"
    llm.provider = "local"
    return llm


@pytest.fixture
def mock_fallback_llm():
    """Create mock fallback (cloud) LLM."""
    llm = MagicMock(spec=BaseLLM)
    llm.complete = AsyncMock()
    llm.complete_structured = AsyncMock()
    llm.health_check = AsyncMock(return_value=True)
    llm.model_name = "claude-model"
    llm.model = "claude-model"
    llm.provider = "anthropic"
    return llm


@pytest.fixture
def router(mock_primary_llm, mock_fallback_llm):
    """Create LLM router with both primary and fallback."""
    return LLMRouter(
        primary=mock_primary_llm,
        fallback=mock_fallback_llm,
        max_retries=3,
    )


@pytest.fixture
def router_no_fallback(mock_primary_llm):
    """Create LLM router without fallback."""
    return LLMRouter(primary=mock_primary_llm, fallback=None)


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello"),
    ]


@pytest.fixture
def success_response():
    """Successful LLM response."""
    return LLMResponse(
        content="Hello! How can I help?",
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 20},
    )


class TestLLMRouter:
    """Tests for LLMRouter."""

    @pytest.mark.asyncio
    async def test_primary_success(
        self, router, mock_primary_llm, sample_messages, success_response
    ):
        """Test that primary LLM is used when healthy."""
        mock_primary_llm.complete.return_value = success_response

        result = await router.complete(sample_messages)

        assert result.content == "Hello! How can I help?"
        mock_primary_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_connection_error(
        self, router, mock_primary_llm, mock_fallback_llm, sample_messages, success_response
    ):
        """Test fallback when primary has connection error."""
        mock_primary_llm.complete.side_effect = LLMConnectionError("Connection failed")
        mock_fallback_llm.complete.return_value = success_response

        result = await router.complete(sample_messages)

        assert result.content == "Hello! How can I help?"
        mock_primary_llm.complete.assert_called()
        mock_fallback_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_timeout(
        self, router, mock_primary_llm, mock_fallback_llm, sample_messages, success_response
    ):
        """Test fallback when primary times out."""
        mock_primary_llm.complete.side_effect = LLMTimeoutError("Request timed out")
        mock_fallback_llm.complete.return_value = success_response

        result = await router.complete(sample_messages)

        assert result.content == "Hello! How can I help?"
        mock_fallback_llm.complete.assert_called()

    @pytest.mark.asyncio
    async def test_fallback_on_overload(
        self, router, mock_primary_llm, mock_fallback_llm, sample_messages, success_response
    ):
        """Test fallback when primary is overloaded."""
        mock_primary_llm.complete.side_effect = LLMOverloadError("Service overloaded")
        mock_fallback_llm.complete.return_value = success_response

        result = await router.complete(sample_messages)

        mock_fallback_llm.complete.assert_called()

    @pytest.mark.asyncio
    async def test_no_fallback_raises_error(
        self, router_no_fallback, mock_primary_llm, sample_messages
    ):
        """Test that error is raised when no fallback available."""
        mock_primary_llm.complete.side_effect = LLMConnectionError("Connection failed")

        with pytest.raises(LLMConnectionError):
            await router_no_fallback.complete(sample_messages)

    @pytest.mark.asyncio
    async def test_prefer_fallback_flag(
        self, router, mock_primary_llm, mock_fallback_llm, sample_messages, success_response
    ):
        """Test prefer_fallback bypasses primary."""
        mock_fallback_llm.complete.return_value = success_response

        result = await router.complete(sample_messages, prefer_fallback=True)

        mock_primary_llm.complete.assert_not_called()
        mock_fallback_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_structured_output_primary(
        self, router, mock_primary_llm, sample_messages
    ):
        """Test structured output uses primary when healthy."""
        expected = MockSchema(result="test", confidence=0.9)
        mock_primary_llm.complete_structured.return_value = expected

        result = await router.complete_structured(sample_messages, MockSchema)

        assert result.result == "test"
        assert result.confidence == 0.9
        mock_primary_llm.complete_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_structured_output_fallback(
        self, router, mock_primary_llm, mock_fallback_llm, sample_messages
    ):
        """Test structured output falls back on error."""
        mock_primary_llm.complete_structured.side_effect = LLMTimeoutError("Timeout")
        expected = MockSchema(result="fallback", confidence=0.8)
        mock_fallback_llm.complete_structured.return_value = expected

        result = await router.complete_structured(sample_messages, MockSchema)

        assert result.result == "fallback"
        mock_fallback_llm.complete_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_both_healthy(self, router, mock_primary_llm, mock_fallback_llm):
        """Test health check when both LLMs are healthy."""
        mock_primary_llm.health_check.return_value = True
        mock_fallback_llm.health_check.return_value = True

        result = await router.health_check()

        assert result["primary"] is True
        assert result["fallback"] is True

    @pytest.mark.asyncio
    async def test_health_check_primary_unhealthy(
        self, router, mock_primary_llm, mock_fallback_llm
    ):
        """Test health check when primary is unhealthy."""
        mock_primary_llm.health_check.return_value = False
        mock_fallback_llm.health_check.return_value = True

        result = await router.health_check()

        assert result["primary"] is False
        assert result["fallback"] is True

    def test_active_provider_when_healthy(self, router):
        """Test active_provider returns primary when healthy."""
        assert router.active_provider == "local"

    def test_active_provider_after_failures(self, router):
        """Test active_provider changes after failures."""
        # Simulate failures
        router._primary_healthy = False

        assert router.active_provider == "anthropic"

    @pytest.mark.asyncio
    async def test_consecutive_failures_mark_unhealthy(
        self, router, mock_primary_llm, mock_fallback_llm, sample_messages, success_response
    ):
        """Test that consecutive failures mark primary as unhealthy."""
        mock_primary_llm.complete.side_effect = LLMConnectionError("Failed")
        mock_fallback_llm.complete.return_value = success_response

        # Make multiple requests to trigger failure threshold
        for _ in range(3):
            await router.complete(sample_messages)

        # Primary should be marked unhealthy
        assert router._primary_healthy is False
        assert router._consecutive_failures >= 3

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(
        self, router, mock_primary_llm, sample_messages, success_response
    ):
        """Test that successful call resets failure count."""
        router._consecutive_failures = 2
        mock_primary_llm.complete.return_value = success_response

        await router.complete(sample_messages)

        assert router._consecutive_failures == 0
        assert router._primary_healthy is True


class TestLLMRouterEdgeCases:
    """Edge case tests for LLM router."""

    @pytest.mark.asyncio
    async def test_both_llms_fail(
        self, router, mock_primary_llm, mock_fallback_llm, sample_messages
    ):
        """Test behavior when both LLMs fail."""
        mock_primary_llm.complete.side_effect = LLMConnectionError("Primary failed")
        mock_fallback_llm.complete.side_effect = LLMConnectionError("Fallback failed")

        with pytest.raises(LLMConnectionError):
            await router.complete(sample_messages)

    @pytest.mark.asyncio
    async def test_always_try_primary_disabled(
        self, mock_primary_llm, mock_fallback_llm, sample_messages, success_response
    ):
        """Test always_try_primary=False respects unhealthy state."""
        router = LLMRouter(
            primary=mock_primary_llm,
            fallback=mock_fallback_llm,
            always_try_primary=False,
        )
        router._primary_healthy = False
        mock_fallback_llm.complete.return_value = success_response

        await router.complete(sample_messages)

        # Should skip unhealthy primary
        mock_primary_llm.complete.assert_not_called()
        mock_fallback_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_always_try_primary_enabled(
        self, router, mock_primary_llm, mock_fallback_llm, sample_messages, success_response
    ):
        """Test always_try_primary=True tries primary even when unhealthy."""
        router._primary_healthy = False
        mock_primary_llm.complete.return_value = success_response

        await router.complete(sample_messages)

        # Should still try primary
        mock_primary_llm.complete.assert_called_once()
