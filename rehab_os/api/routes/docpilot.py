"""DocPilot voice command proxy.

Routes voice commands from the dashboard to DocPilot's API server
on port 3847. Also proxies TTS so DocPilot can speak back.
"""

import logging
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any

router = APIRouter(prefix="/docpilot", tags=["docpilot"])
logger = logging.getLogger(__name__)

DOCPILOT_URL = "http://localhost:3847"


class VoiceCommand(BaseModel):
    command: str
    context: dict = {}


@router.post("/command")
async def voice_command(req: VoiceCommand):
    """Send a natural language command to DocPilot."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            res = await client.post(
                f"{DOCPILOT_URL}/command",
                json={"command": req.command, "context": req.context},
            )
            return JSONResponse(content=res.json(), status_code=res.status_code)
        except httpx.ConnectError:
            raise HTTPException(503, "DocPilot is not running. Start it on port 3847.")


@router.post("/command/connect")
async def voice_connect(emr: str = "hellonote"):
    """Start a voice-commanded EMR session."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            res = await client.post(
                f"{DOCPILOT_URL}/command/connect",
                json={"emr": emr},
            )
            return JSONResponse(content=res.json(), status_code=res.status_code)
        except httpx.ConnectError:
            raise HTTPException(503, "DocPilot is not running.")


@router.get("/command/status")
async def voice_status():
    """Get DocPilot voice commander status."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            res = await client.get(f"{DOCPILOT_URL}/command/status")
            return JSONResponse(content=res.json(), status_code=res.status_code)
        except httpx.ConnectError:
            return JSONResponse(content={"activeSession": None, "status": "offline"})


@router.get("/health")
async def docpilot_health():
    """Check DocPilot availability."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            res = await client.get(f"{DOCPILOT_URL}/health")
            return JSONResponse(content=res.json())
        except httpx.ConnectError:
            return JSONResponse(
                content={"status": "offline", "message": "DocPilot not running"},
                status_code=503,
            )
