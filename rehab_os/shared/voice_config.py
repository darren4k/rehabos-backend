"""
Qwen3-TTS Voice Configuration
Shared across all RehabOS projects on DGX Spark

Usage:
    from rehab_os.shared.voice_config import VOICES, LANGUAGES, synthesize_speech
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import httpx
import os

# TTS Server Configuration
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://192.168.68.123:8080")


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"


@dataclass
class Voice:
    id: str
    name: str
    description: str
    gender: Gender
    language: str
    accent: str
    voice_design: Optional[str] = None  # For extended voices


# Server-native voices (available directly on TTS server)
SERVER_VOICES: List[Voice] = [
    # English
    Voice("vivian", "Vivian", "Clear English female", Gender.FEMALE, "en", "American"),
    Voice("ryan", "Ryan", "Warm English male", Gender.MALE, "en", "American"),
    Voice("aiden", "Aiden", "Friendly male voice", Gender.MALE, "en", "American"),
    Voice("eric", "Eric", "Professional male", Gender.MALE, "en", "American"),
    # Multilingual
    Voice("serena", "Serena", "Calm bilingual female", Gender.FEMALE, "multi", "Neutral"),
    # Asian languages
    Voice("ono_anna", "Anna", "Gentle Japanese female", Gender.FEMALE, "ja", "Japanese"),
    Voice("sohee", "Sohee", "Bright Korean female", Gender.FEMALE, "ko", "Korean"),
    Voice("dylan", "Dylan", "Beijing dialect male", Gender.MALE, "zh", "Beijing"),
    Voice("uncle_fu", "Uncle Fu", "Warm Chinese elder", Gender.MALE, "zh", "Mandarin"),
]

# Extended voices (use voice design with multilingual base)
EXTENDED_VOICES: List[Voice] = [
    # Spanish
    Voice("sofia_es", "Sofia", "Warm Spanish female", Gender.FEMALE, "es", "Spanish",
          "A warm, friendly female voice speaking Spanish with a clear Castilian accent"),
    Voice("carlos_es", "Carlos", "Professional Spanish male", Gender.MALE, "es", "Spanish",
          "A professional, calm male voice speaking Spanish with a neutral Latin American accent"),
    # French
    Voice("marie_fr", "Marie", "Elegant French female", Gender.FEMALE, "fr", "French",
          "An elegant, sophisticated female voice speaking French with a Parisian accent"),
    Voice("pierre_fr", "Pierre", "Calm French male", Gender.MALE, "fr", "French",
          "A calm, reassuring male voice speaking French with a standard French accent"),
    # Italian
    Voice("giulia_it", "Giulia", "Warm Italian female", Gender.FEMALE, "it", "Italian",
          "A warm, expressive female voice speaking Italian with a standard Italian accent"),
    Voice("marco_it", "Marco", "Friendly Italian male", Gender.MALE, "it", "Italian",
          "A friendly, energetic male voice speaking Italian with a Roman accent"),
    # Hindi
    Voice("priya_hi", "Priya", "Gentle Hindi female", Gender.FEMALE, "hi", "Indian",
          "A gentle, caring female voice speaking Hindi with a clear pronunciation"),
    Voice("raj_hi", "Raj", "Professional Hindi male", Gender.MALE, "hi", "Indian",
          "A professional, warm male voice speaking Hindi with standard pronunciation"),
    # Filipino/Tagalog
    Voice("maria_tl", "Maria", "Friendly Filipino female", Gender.FEMALE, "tl", "Filipino",
          "A friendly, warm female voice speaking Tagalog with a Manila accent"),
    Voice("jose_tl", "Jose", "Caring Filipino male", Gender.MALE, "tl", "Filipino",
          "A caring, gentle male voice speaking Tagalog with clear pronunciation"),
    # Portuguese
    Voice("ana_pt", "Ana", "Warm Portuguese female", Gender.FEMALE, "pt", "Brazilian",
          "A warm, friendly female voice speaking Portuguese with a Brazilian accent"),
    Voice("lucas_pt", "Lucas", "Professional Portuguese male", Gender.MALE, "pt", "Brazilian",
          "A professional male voice speaking Portuguese with a Brazilian accent"),
    # German
    Voice("anna_de", "Anna", "Clear German female", Gender.FEMALE, "de", "German",
          "A clear, professional female voice speaking German with standard pronunciation"),
    Voice("hans_de", "Hans", "Calm German male", Gender.MALE, "de", "German",
          "A calm, professional male voice speaking German with standard pronunciation"),
    # Russian
    Voice("natasha_ru", "Natasha", "Warm Russian female", Gender.FEMALE, "ru", "Russian",
          "A warm, gentle female voice speaking Russian with standard pronunciation"),
    Voice("ivan_ru", "Ivan", "Professional Russian male", Gender.MALE, "ru", "Russian",
          "A professional male voice speaking Russian with clear Moscow accent"),
    # Arabic
    Voice("fatima_ar", "Fatima", "Gentle Arabic female", Gender.FEMALE, "ar", "Arabic",
          "A gentle, caring female voice speaking Arabic with Modern Standard Arabic pronunciation"),
    Voice("ahmed_ar", "Ahmed", "Calm Arabic male", Gender.MALE, "ar", "Arabic",
          "A calm, professional male voice speaking Arabic with MSA pronunciation"),
    # Vietnamese
    Voice("linh_vi", "Linh", "Clear Vietnamese female", Gender.FEMALE, "vi", "Vietnamese",
          "A clear, friendly female voice speaking Vietnamese with Northern accent"),
    Voice("minh_vi", "Minh", "Professional Vietnamese male", Gender.MALE, "vi", "Vietnamese",
          "A professional male voice speaking Vietnamese with Northern accent"),
    # Thai
    Voice("somchai_th", "Somchai", "Friendly Thai male", Gender.MALE, "th", "Thai",
          "A friendly male voice speaking Thai with clear Central Thai pronunciation"),
    Voice("nari_th", "Nari", "Gentle Thai female", Gender.FEMALE, "th", "Thai",
          "A gentle, warm female voice speaking Thai with standard pronunciation"),
    # Indonesian
    Voice("dewi_id", "Dewi", "Warm Indonesian female", Gender.FEMALE, "id", "Indonesian",
          "A warm, friendly female voice speaking Indonesian with standard Jakarta accent"),
    Voice("budi_id", "Budi", "Professional Indonesian male", Gender.MALE, "id", "Indonesian",
          "A professional male voice speaking Indonesian with clear pronunciation"),
]

# All voices combined
ALL_VOICES: List[Voice] = SERVER_VOICES + EXTENDED_VOICES

# Voice styles/personalities
VOICE_STYLES = [
    {"id": "professional", "name": "Professional", "description": "Clear and clinical"},
    {"id": "friendly", "name": "Friendly", "description": "Warm and approachable"},
    {"id": "calm", "name": "Calm", "description": "Soothing and relaxed"},
    {"id": "energetic", "name": "Energetic", "description": "Upbeat and motivating"},
    {"id": "empathetic", "name": "Empathetic", "description": "Caring and understanding"},
    {"id": "instructional", "name": "Instructional", "description": "Clear teaching style"},
]

# Supported languages
LANGUAGES = [
    {"id": "all", "name": "All Languages"},
    {"id": "en", "name": "English"},
    {"id": "es", "name": "Spanish"},
    {"id": "fr", "name": "French"},
    {"id": "it", "name": "Italian"},
    {"id": "pt", "name": "Portuguese"},
    {"id": "de", "name": "German"},
    {"id": "hi", "name": "Hindi"},
    {"id": "tl", "name": "Filipino"},
    {"id": "zh", "name": "Chinese"},
    {"id": "ja", "name": "Japanese"},
    {"id": "ko", "name": "Korean"},
    {"id": "vi", "name": "Vietnamese"},
    {"id": "th", "name": "Thai"},
    {"id": "id", "name": "Indonesian"},
    {"id": "ru", "name": "Russian"},
    {"id": "ar", "name": "Arabic"},
    {"id": "multi", "name": "Multilingual"},
]

DEFAULT_VOICE = "vivian"
DEFAULT_STYLE = "friendly"


def get_voice(voice_id: str) -> Optional[Voice]:
    """Get voice by ID."""
    for voice in ALL_VOICES:
        if voice.id == voice_id:
            return voice
    return None


def get_voices_by_language(language: str) -> List[Voice]:
    """Filter voices by language code."""
    if language == "all":
        return ALL_VOICES
    return [v for v in ALL_VOICES if v.language == language]


def is_extended_voice(voice_id: str) -> bool:
    """Check if voice requires voice design (extended voice)."""
    return any(v.id == voice_id for v in EXTENDED_VOICES)


def get_base_voice(voice_id: str) -> str:
    """Get the base server voice for extended voices."""
    if is_extended_voice(voice_id):
        return "serena"  # Use multilingual base
    return voice_id


async def synthesize_speech(
    text: str,
    voice: str = DEFAULT_VOICE,
    style: str = DEFAULT_STYLE,
    speed: float = 1.0,
) -> Optional[bytes]:
    """
    Synthesize speech using Qwen3-TTS.

    Args:
        text: Text to synthesize
        voice: Voice ID
        style: Speaking style
        speed: Speech speed (0.5 to 2.0)

    Returns:
        Audio bytes (WAV format) or None on error
    """
    voice_obj = get_voice(voice)
    base_voice = get_base_voice(voice)

    # Build request
    request_body: Dict[str, Any] = {
        "input": text,
        "voice": base_voice,
        "style": style,
        "speed": speed,
        "response_format": "wav",
    }

    # Add voice design for extended voices
    if voice_obj and voice_obj.voice_design:
        request_body["input"] = f"[Voice: {voice_obj.voice_design}] {text}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{TTS_BASE_URL}/v1/audio/speech",
                json=request_body,
            )

            if response.status_code == 200:
                return response.content
            else:
                print(f"TTS API error: {response.status_code}")
                return None

    except Exception as e:
        print(f"TTS synthesis error: {e}")
        return None


def synthesize_speech_sync(
    text: str,
    voice: str = DEFAULT_VOICE,
    style: str = DEFAULT_STYLE,
    speed: float = 1.0,
) -> Optional[bytes]:
    """
    Synchronous version of synthesize_speech.
    """
    import asyncio
    return asyncio.run(synthesize_speech(text, voice, style, speed))


# Export for convenience
VOICES = ALL_VOICES
