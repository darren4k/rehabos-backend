"""Auth routes — login, logout, refresh, invite, change-password."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.api.dependencies import get_current_user, require_admin
from rehab_os.core.auth import (
    clear_auth_cookies,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    set_auth_cookies,
    verify_password,
    REFRESH_COOKIE,
)
from rehab_os.core.database import get_db
from rehab_os.core.models import Provider

router = APIRouter(prefix="/auth")


# --- Request / Response schemas ---

class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    id: str
    first_name: str
    last_name: str
    email: Optional[str]
    role: str
    discipline: str
    organization_id: Optional[str]
    must_change_password: bool


class InviteRequest(BaseModel):
    email: str
    first_name: str
    last_name: str
    discipline: str
    role: str = "therapist"
    temp_password: str
    organization_id: Optional[uuid.UUID] = None
    credentials: Optional[str] = None
    npi: Optional[str] = None


class RegisterRequest(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str
    discipline: str
    credentials: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# --- Endpoints ---

@router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> LoginResponse:
    result = await db.execute(
        select(Provider).where(Provider.email == body.email, Provider.active.is_(True))
    )
    provider = result.scalar_one_or_none()

    if not provider or not provider.password_hash:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(body.password, provider.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access = create_access_token(str(provider.id), provider.role)
    refresh = create_refresh_token(str(provider.id))
    set_auth_cookies(response, access, refresh)

    return LoginResponse(
        id=str(provider.id),
        first_name=provider.first_name,
        last_name=provider.last_name,
        email=provider.email,
        role=provider.role,
        discipline=provider.discipline,
        organization_id=str(provider.organization_id) if provider.organization_id else None,
        must_change_password=provider.must_change_password,
    )


@router.post("/logout")
async def logout(response: Response) -> dict:
    clear_auth_cookies(response)
    return {"ok": True}


@router.post("/register", response_model=LoginResponse)
async def register(
    body: RegisterRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> LoginResponse:
    """Self-registration — creates a therapist account."""
    existing = await db.execute(
        select(Provider).where(Provider.email == body.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    provider = Provider(
        first_name=body.first_name,
        last_name=body.last_name,
        email=body.email,
        discipline=body.discipline,
        role="therapist",
        password_hash=hash_password(body.password),
        must_change_password=False,
        credentials=body.credentials,
    )
    db.add(provider)
    await db.flush()

    access = create_access_token(str(provider.id), provider.role)
    refresh = create_refresh_token(str(provider.id))
    set_auth_cookies(response, access, refresh)

    return LoginResponse(
        id=str(provider.id),
        first_name=provider.first_name,
        last_name=provider.last_name,
        email=provider.email,
        role=provider.role,
        discipline=provider.discipline,
        organization_id=str(provider.organization_id) if provider.organization_id else None,
        must_change_password=provider.must_change_password,
    )


@router.post("/refresh")
async def refresh_token(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> dict:
    token = request.cookies.get(REFRESH_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="No refresh token")

    claims = decode_token(token)
    if not claims or claims.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    provider_id = claims.get("sub")
    if not provider_id:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    try:
        pid = uuid.UUID(provider_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    result = await db.execute(
        select(Provider).where(Provider.id == pid, Provider.active.is_(True))
    )
    provider = result.scalar_one_or_none()
    if not provider:
        raise HTTPException(status_code=401, detail="User not found")

    access = create_access_token(str(provider.id), provider.role)
    new_refresh = create_refresh_token(str(provider.id))
    set_auth_cookies(response, access, new_refresh)

    return {"ok": True}


@router.post("/invite", response_model=LoginResponse)
async def invite_provider(
    body: InviteRequest,
    db: AsyncSession = Depends(get_db),
    admin: Provider = Depends(require_admin),
) -> LoginResponse:
    existing = await db.execute(
        select(Provider).where(Provider.email == body.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    if len(body.temp_password) < 8:
        raise HTTPException(status_code=400, detail="Temporary password must be at least 8 characters")

    provider = Provider(
        first_name=body.first_name,
        last_name=body.last_name,
        email=body.email,
        discipline=body.discipline,
        role=body.role,
        password_hash=hash_password(body.temp_password),
        must_change_password=True,
        organization_id=body.organization_id or admin.organization_id,
        credentials=body.credentials,
        npi=body.npi,
    )
    db.add(provider)
    await db.flush()

    return LoginResponse(
        id=str(provider.id),
        first_name=provider.first_name,
        last_name=provider.last_name,
        email=provider.email,
        role=provider.role,
        discipline=provider.discipline,
        organization_id=str(provider.organization_id) if provider.organization_id else None,
        must_change_password=provider.must_change_password,
    )


@router.post("/change-password")
async def change_password(
    body: ChangePasswordRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
    current_user: Provider = Depends(get_current_user),
) -> dict:
    if not current_user.password_hash:
        raise HTTPException(status_code=400, detail="No password set")

    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    if not verify_password(body.current_password, current_user.password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    current_user.password_hash = hash_password(body.new_password)
    current_user.must_change_password = False
    await db.flush()

    # Reissue tokens so claims (role, etc.) are fresh
    access = create_access_token(str(current_user.id), current_user.role)
    refresh = create_refresh_token(str(current_user.id))
    set_auth_cookies(response, access, refresh)

    return {"ok": True}
