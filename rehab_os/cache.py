"""Redis cache with in-memory fallback.

Production: Redis via redis-py async.
Development: In-memory dict with TTL tracking.
Falls back to InMemoryCache if Redis is unavailable.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract cache interface."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        ...

    @abstractmethod
    async def get_or_set(self, key: str, factory: Callable[[], Any], ttl: int = 300) -> Any:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...


class InMemoryCache(CacheBackend):
    """Simple dict-based cache with TTL. For dev/testing only."""

    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)

    async def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        self._store[key] = (value, time.time() + ttl)
        return True

    async def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def get_or_set(self, key: str, factory: Callable[[], Any], ttl: int = 300) -> Any:
        cached = await self.get(key)
        if cached is not None:
            return cached
        value = factory()
        await self.set(key, value, ttl)
        return value

    async def close(self) -> None:
        self._store.clear()


class RedisCache(CacheBackend):
    """Redis-backed cache. Falls back to InMemoryCache if Redis is unavailable."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis_url = redis_url
        self._redis = None
        self._fallback = InMemoryCache()
        self._using_fallback = False

    async def _connect(self) -> bool:
        if self._redis is not None:
            return True
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await self._redis.ping()
            logger.info("Redis cache connected: %s", self._redis_url)
            self._using_fallback = False
            return True
        except Exception as e:
            logger.warning("Redis unavailable, using in-memory fallback: %s", e)
            self._redis = None
            self._using_fallback = True
            return False

    async def _client(self):
        if self._redis is None:
            await self._connect()
        return self._redis

    def _serialize(self, value: Any) -> str:
        return json.dumps(value, default=str)

    def _deserialize(self, raw: Optional[str]) -> Optional[Any]:
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def get(self, key: str) -> Optional[Any]:
        client = await self._client()
        if client is None:
            return await self._fallback.get(key)
        try:
            raw = await client.get(key)
            return self._deserialize(raw)
        except Exception as e:
            logger.warning("Redis get failed for '%s': %s", key, e)
            return await self._fallback.get(key)

    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        client = await self._client()
        if client is None:
            return await self._fallback.set(key, value, ttl)
        try:
            await client.set(key, self._serialize(value), ex=ttl)
            return True
        except Exception as e:
            logger.warning("Redis set failed for '%s': %s", key, e)
            return await self._fallback.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        client = await self._client()
        if client is None:
            return await self._fallback.delete(key)
        try:
            return bool(await client.delete(key))
        except Exception as e:
            logger.warning("Redis delete failed for '%s': %s", key, e)
            return await self._fallback.delete(key)

    async def exists(self, key: str) -> bool:
        client = await self._client()
        if client is None:
            return await self._fallback.exists(key)
        try:
            return bool(await client.exists(key))
        except Exception as e:
            logger.warning("Redis exists failed for '%s': %s", key, e)
            return await self._fallback.exists(key)

    async def get_or_set(self, key: str, factory: Callable[[], Any], ttl: int = 300) -> Any:
        cached = await self.get(key)
        if cached is not None:
            return cached
        value = factory()
        await self.set(key, value, ttl)
        return value

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
        await self._fallback.close()


# Singleton
_cache: Optional[CacheBackend] = None


def get_cache() -> CacheBackend:
    """Get the global cache instance. Returns InMemoryCache if not initialized."""
    global _cache
    if _cache is None:
        _cache = InMemoryCache()
    return _cache


async def init_cache(redis_url: str = "") -> CacheBackend:
    """Initialize the global cache. Call during app startup."""
    global _cache
    if redis_url:
        _cache = RedisCache(redis_url)
    else:
        _cache = InMemoryCache()
    logger.info("Cache initialized: %s", type(_cache).__name__)
    return _cache


async def close_cache() -> None:
    """Close the global cache. Call during app shutdown."""
    global _cache
    if _cache is not None:
        await _cache.close()
        _cache = None
