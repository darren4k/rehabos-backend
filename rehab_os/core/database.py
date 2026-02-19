"""Database engine and async session factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rehab_os.config import get_settings
from rehab_os.core.models import Base, Provider

logger = logging.getLogger(__name__)


def get_database_url() -> str:
    return get_settings().core_database_url


@lru_cache
def _get_engine():
    return create_async_engine(
        get_database_url(),
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )


@lru_cache
def _get_session_factory():
    return async_sessionmaker(_get_engine(), class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async session."""
    async with _get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables (dev only; production uses alembic)."""
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed_admin()


async def seed_admin() -> None:
    """Create the first admin provider if configured and not yet present."""
    settings = get_settings()
    if not settings.first_admin_email or not settings.first_admin_password:
        return

    from rehab_os.core.auth import hash_password

    async with _get_session_factory()() as session:
        result = await session.execute(
            select(Provider).where(Provider.email == settings.first_admin_email)
        )
        if result.scalar_one_or_none():
            return

        admin = Provider(
            first_name="Admin",
            last_name="User",
            email=settings.first_admin_email,
            discipline="pt",
            role="owner",
            password_hash=hash_password(settings.first_admin_password),
            must_change_password=True,
        )
        session.add(admin)
        await session.commit()
        logger.info("Seeded admin user: %s", settings.first_admin_email)
