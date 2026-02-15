"""
Shared utilities and configurations for RehabOS projects.
"""

from .voice_config import (
    Voice,
    Gender,
    VOICES,
    ALL_VOICES,
    SERVER_VOICES,
    EXTENDED_VOICES,
    VOICE_STYLES,
    LANGUAGES,
    DEFAULT_VOICE,
    DEFAULT_STYLE,
    get_voice,
    get_voices_by_language,
    synthesize_speech,
    synthesize_speech_sync,
)

__all__ = [
    "Voice",
    "Gender",
    "VOICES",
    "ALL_VOICES",
    "SERVER_VOICES",
    "EXTENDED_VOICES",
    "VOICE_STYLES",
    "LANGUAGES",
    "DEFAULT_VOICE",
    "DEFAULT_STYLE",
    "get_voice",
    "get_voices_by_language",
    "synthesize_speech",
    "synthesize_speech_sync",
]
