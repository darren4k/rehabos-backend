"""Voice Service Integration with Qwen3-TTS.

This module provides text-to-speech functionality using Qwen3-TTS
running on the local DGX Spark server for high-quality voice synthesis.

Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS.git
Features:
- Natural speech synthesis
- Multiple voice options
- Emotion/style control
- Real-time streaming

DGX Spark Integration:
- Connects to local Qwen3-TTS server
- Low-latency audio generation
- No external API costs
"""

import logging
import base64
from typing import Optional, Literal
from enum import Enum
import httpx
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import io

from rehab_os.config import get_settings

router = APIRouter(prefix="/voice", tags=["voice"])

logger = logging.getLogger(__name__)


# ==================
# CONFIGURATION
# ==================

class VoiceStyle(str, Enum):
    """Voice style options for TTS."""
    NEUTRAL = "neutral"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CALM = "calm"
    ENCOURAGING = "encouraging"


class VoiceGender(str, Enum):
    FEMALE = "female"
    MALE = "male"


class TTSRequest(BaseModel):
    """Request for text-to-speech synthesis."""
    text: str
    voice_style: VoiceStyle = VoiceStyle.PROFESSIONAL
    voice_gender: VoiceGender = VoiceGender.FEMALE
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=1.0, ge=0.5, le=2.0)
    language: str = "en"
    streaming: bool = False


class TTSResponse(BaseModel):
    """Response from TTS synthesis."""
    audio_base64: str
    duration_seconds: float
    sample_rate: int
    format: str


class VoiceConfig(BaseModel):
    """Voice service configuration."""
    enabled: bool = True
    server_url: str = "http://localhost:8080"  # Qwen3-TTS server
    default_voice: VoiceStyle = VoiceStyle.PROFESSIONAL
    default_gender: VoiceGender = VoiceGender.FEMALE
    max_text_length: int = 5000


# ==================
# TTS SERVER CLIENT
# ==================

class QwenTTSClient:
    """Client for Qwen3-TTS server running on DGX Spark."""

    # Map gender to Qwen3-TTS speaker names
    SPEAKERS = {
        VoiceGender.FEMALE: "ono_anna",  # Female voice
        VoiceGender.MALE: "aiden",       # Male voice
    }

    # All available speakers: aiden, dylan, eric, ono_anna, ryan

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)  # Longer timeout for TTS

    async def synthesize(
        self,
        text: str,
        voice_style: VoiceStyle = VoiceStyle.PROFESSIONAL,
        voice_gender: VoiceGender = VoiceGender.FEMALE,
        speed: float = 1.0,
        pitch: float = 1.0,
        language: str = "en"
    ) -> tuple[bytes, float, int]:
        """Synthesize text to speech.

        Returns (audio_bytes, duration_seconds, sample_rate)
        """
        # Map voice style to Qwen3-TTS parameters
        style_prompts = {
            VoiceStyle.NEUTRAL: "neutral and clear",
            VoiceStyle.PROFESSIONAL: "professional and articulate",
            VoiceStyle.FRIENDLY: "warm and friendly",
            VoiceStyle.CALM: "calm and soothing",
            VoiceStyle.ENCOURAGING: "encouraging and positive"
        }

        prompt = style_prompts.get(voice_style, "neutral")
        speaker = self.SPEAKERS.get(voice_gender, "aiden")

        try:
            # Call Qwen3-TTS API
            response = await self.client.post(
                f"{self.server_url}/v1/audio/speech",
                json={
                    "input": text,
                    "voice": speaker,
                    "style": prompt,
                    "speed": speed,
                    "response_format": "wav"
                }
            )

            if response.status_code == 200:
                audio_data = response.content
                # Qwen3-TTS outputs 24kHz audio
                sample_rate = 24000
                # Estimate duration from WAV data (16-bit mono)
                duration = len(audio_data) / (sample_rate * 2) if audio_data else 0
                return audio_data, duration, sample_rate
            else:
                logger.error(f"TTS server error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=502, detail="TTS server error")

        except httpx.ConnectError:
            logger.warning("TTS server not available, using fallback")
            raise HTTPException(
                status_code=503,
                detail="Voice service unavailable. Ensure Qwen3-TTS is running on DGX Spark."
            )

    async def synthesize_stream(
        self,
        text: str,
        voice_style: VoiceStyle = VoiceStyle.PROFESSIONAL,
        voice_gender: VoiceGender = VoiceGender.FEMALE,
        speed: float = 1.0
    ):
        """Stream synthesized audio."""
        style_prompts = {
            VoiceStyle.NEUTRAL: "neutral and clear",
            VoiceStyle.PROFESSIONAL: "professional and articulate",
            VoiceStyle.FRIENDLY: "warm and friendly",
            VoiceStyle.CALM: "calm and soothing",
            VoiceStyle.ENCOURAGING: "encouraging and positive"
        }

        prompt = style_prompts.get(voice_style, "neutral")
        speaker = self.SPEAKERS.get(voice_gender, "aiden")

        try:
            async with self.client.stream(
                "POST",
                f"{self.server_url}/v1/audio/speech",
                json={
                    "input": text,
                    "voice": speaker,
                    "style": prompt,
                    "speed": speed,
                    "response_format": "wav",
                    "stream": True
                }
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

        except httpx.ConnectError:
            logger.warning("TTS streaming not available")
            raise HTTPException(
                status_code=503,
                detail="Voice streaming unavailable"
            )

    async def check_health(self) -> bool:
        """Check if TTS server is healthy."""
        try:
            response = await self.client.get(f"{self.server_url}/health")
            return response.status_code == 200
        except:
            return False


# Global TTS client (lazy initialization)
_tts_client: Optional[QwenTTSClient] = None


def get_tts_client() -> QwenTTSClient:
    """Get or create TTS client."""
    global _tts_client
    if _tts_client is None:
        settings = get_settings()
        # Default to port 8080 on localhost (DGX Spark)
        server_url = getattr(settings, 'tts_server_url', 'http://localhost:8080')
        _tts_client = QwenTTSClient(server_url)
    return _tts_client


# ==================
# API ENDPOINTS
# ==================

@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """Synthesize text to speech using Qwen3-TTS.

    Runs on local DGX Spark server for low-latency, high-quality voice.
    """
    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 chars)")

    client = get_tts_client()

    audio_bytes, duration, sample_rate = await client.synthesize(
        text=request.text,
        voice_style=request.voice_style,
        voice_gender=request.voice_gender,
        speed=request.speed,
        pitch=request.pitch,
        language=request.language
    )

    return TTSResponse(
        audio_base64=base64.b64encode(audio_bytes).decode(),
        duration_seconds=duration,
        sample_rate=sample_rate,
        format="wav"
    )


@router.post("/synthesize/stream")
async def synthesize_speech_stream(request: TTSRequest):
    """Stream synthesized speech audio.

    Returns audio as streaming response for real-time playback.
    """
    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 chars)")

    client = get_tts_client()

    async def audio_stream():
        async for chunk in client.synthesize_stream(
            text=request.text,
            voice_style=request.voice_style,
            voice_gender=request.voice_gender,
            speed=request.speed
        ):
            yield chunk

    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "no-cache"
        }
    )


@router.get("/health")
async def check_voice_health():
    """Check voice service health."""
    client = get_tts_client()
    is_healthy = await client.check_health()

    return {
        "status": "healthy" if is_healthy else "unavailable",
        "service": "qwen3-tts",
        "server_url": client.server_url,
        "available": is_healthy,
        "message": "Qwen3-TTS running on DGX Spark" if is_healthy else "Start Qwen3-TTS server on DGX Spark"
    }


@router.get("/voices")
async def list_available_voices():
    """List available voice options."""
    return {
        "voices": [
            {
                "id": "female_professional",
                "gender": "female",
                "style": "professional",
                "description": "Clear, articulate female voice for clinical communication"
            },
            {
                "id": "female_friendly",
                "gender": "female",
                "style": "friendly",
                "description": "Warm female voice for patient interaction"
            },
            {
                "id": "male_professional",
                "gender": "male",
                "style": "professional",
                "description": "Clear, articulate male voice for clinical communication"
            },
            {
                "id": "male_calm",
                "gender": "male",
                "style": "calm",
                "description": "Calm, reassuring male voice"
            }
        ],
        "styles": [v.value for v in VoiceStyle],
        "genders": [v.value for v in VoiceGender],
        "speed_range": {"min": 0.5, "max": 2.0, "default": 1.0},
        "pitch_range": {"min": 0.5, "max": 2.0, "default": 1.0}
    }


@router.get("/setup-instructions")
async def get_setup_instructions():
    """Get instructions for setting up Qwen3-TTS on DGX Spark."""
    return {
        "title": "Qwen3-TTS Setup on DGX Spark",
        "repository": "https://github.com/QwenLM/Qwen3-TTS.git",
        "steps": [
            {
                "step": 1,
                "title": "Clone Repository",
                "command": "git clone https://github.com/QwenLM/Qwen3-TTS.git"
            },
            {
                "step": 2,
                "title": "Install Dependencies",
                "command": "cd Qwen3-TTS && pip install -r requirements.txt"
            },
            {
                "step": 3,
                "title": "Download Model",
                "description": "Download the Qwen3-TTS model weights",
                "command": "python download_model.py"
            },
            {
                "step": 4,
                "title": "Start Server",
                "description": "Start the TTS server on port 8080",
                "command": "python server.py --port 8080 --host 0.0.0.0"
            },
            {
                "step": 5,
                "title": "Configure RehabOS",
                "description": "Set TTS_SERVER_URL in your environment",
                "command": "export TTS_SERVER_URL=http://your-dgx-spark-ip:8080"
            }
        ],
        "hardware_requirements": {
            "gpu": "NVIDIA GPU with 8GB+ VRAM",
            "ram": "16GB+ recommended",
            "storage": "10GB for model"
        },
        "dgx_spark_notes": [
            "DGX Spark provides optimal performance for real-time TTS",
            "Use CUDA for hardware acceleration",
            "Consider running multiple instances for high-load scenarios"
        ]
    }
