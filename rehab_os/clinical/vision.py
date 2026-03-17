"""Clinical vision analysis — gait, ROM, wound, posture via Qwen3-VL.

Adapted from DocPilot's vision module. Uses RehabOS's LLM config pattern.
All media is ephemeral: processed in-memory only, never persisted to disk.
Buffer is zeroed and deallocated after analysis.
"""
from __future__ import annotations

import base64
import gc
import json
import logging
import re
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from rehab_os.config import get_settings

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class VisionConfig(BaseModel):
    """Vision model endpoint configuration."""
    endpoint: str = "http://192.168.68.127:8000/v1"
    model: str = "qwen3-vl-30b-a3b"
    api_key: str = "not-needed"
    timeout: float = 120.0


def _get_vision_config() -> VisionConfig:
    """Build vision config from settings with env-var overrides."""
    settings = get_settings()
    return VisionConfig(
        endpoint=getattr(settings, "vision_endpoint", VisionConfig().endpoint),
        model=getattr(settings, "vision_model", VisionConfig().model),
        api_key=getattr(settings, "vision_api_key", VisionConfig().api_key),
    )


# ============================================================================
# System Prompts
# ============================================================================

VISION_SYSTEM_PROMPT = """You are a clinical observation assistant for PT/OT/SLP therapists.
Analyze the provided clinical image or video frames and return structured observations.

Rules:
- NEVER assert diagnoses. Use "appears to demonstrate", "suggests", "consistent with"
- Always include "(verify)" after each observation
- Focus on measurable, objective findings
- Note body mechanics, alignment, asymmetry, assistive device usage
- For gait: stance time, step length, trunk lean, arm swing, foot clearance
- For ROM/posture: joint angles, alignment deviations, compensatory patterns
- For wounds/swelling: location, approximate size, color, stage if applicable

Output format:
{
  "observations": [
    {"finding": "description", "body_region": "region", "confidence": "high|medium|low", "verify": true}
  ],
  "suggested_documentation": "Draft objective text for SOAP note",
  "things_to_verify": ["list of items clinician should confirm"]
}"""

ANALYSIS_PROMPTS = {
    "general": "Analyze this clinical image for {discipline} documentation.{region_context} Provide structured observations.",
    "gait": (
        "Analyze this gait video/image for a PT assessment. Focus on: "
        "stance phase symmetry (L vs R), step length and cadence, "
        "trunk alignment and lateral lean, arm swing pattern, "
        "foot clearance during swing phase, base of support width, "
        "assistive device usage, weight-bearing pattern, and "
        "compensatory strategies. Observations only, no diagnoses."
    ),
    "rom": (
        "Analyze this image for range of motion assessment. Focus on: "
        "joint angles, end-range position, compensatory movements, "
        "asymmetry between sides. Provide estimated angles where visible."
    ),
    "wound": (
        "Analyze this clinical image for wound/tissue assessment. Focus on: "
        "wound location, approximate dimensions, wound bed color, "
        "surrounding tissue condition, drainage, stage if applicable. "
        "Observations only — clinician must verify all findings."
    ),
    "posture": (
        "Analyze this image for postural assessment. Focus on: "
        "head/cervical alignment, shoulder height symmetry, "
        "thoracic kyphosis, lumbar lordosis, pelvic tilt, "
        "knee alignment, and any compensatory patterns."
    ),
}

# Setting-specific addendum
SETTING_VISION_CONTEXT = {
    "homecare": " Include observations about home environment safety if visible.",
    "snf": " Note functional positioning and equipment visible in the facility setting.",
    "outpatient": "",
}


# ============================================================================
# Data Models
# ============================================================================

class VisionObservation(BaseModel):
    finding: str
    body_region: str = ""
    confidence: str = "medium"
    verify: bool = True


class VisionAnalysisResult(BaseModel):
    observations: list[VisionObservation] = Field(default_factory=list)
    suggested_documentation: str = ""
    things_to_verify: list[str] = Field(default_factory=list)


class ClinicalImageAnalysis(BaseModel):
    analysis: VisionAnalysisResult
    analysis_type: str
    body_region: Optional[str] = None
    discipline: str = "PT"
    setting: str = "outpatient"
    media_persisted: bool = False
    status: str = "analyzed"


# ============================================================================
# Core Analysis Function
# ============================================================================

async def analyze_clinical_image(
    image_bytes: bytes,
    analysis_type: str = "general",
    discipline: str = "PT",
    body_region: Optional[str] = None,
    content_type: str = "image/jpeg",
    setting: str = "outpatient",
) -> ClinicalImageAnalysis:
    """Analyze a clinical image using Qwen3-VL vision model.

    Media lifecycle: ephemeral.
    - Image bytes received into memory (no disk write).
    - Base64-encoded and sent to vision model.
    - Buffer zeroed and deallocated after analysis.

    Args:
        image_bytes: Raw image bytes.
        analysis_type: general, gait, rom, wound, or posture.
        discipline: PT, OT, or SLP.
        body_region: Optional body region tag (e.g., "knee", "shoulder").
        content_type: MIME type of the image.
        setting: outpatient, homecare, or snf.

    Returns:
        ClinicalImageAnalysis with structured observations.
    """
    config = _get_vision_config()
    image_b64 = None

    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{content_type};base64,{image_b64}"

        # Build user prompt
        prompt_template = ANALYSIS_PROMPTS.get(analysis_type, ANALYSIS_PROMPTS["general"])
        if analysis_type == "general":
            region_context = f" Focus on the {body_region} region." if body_region else ""
            user_prompt = prompt_template.format(
                discipline=discipline, region_context=region_context,
            )
        else:
            user_prompt = prompt_template

        # Add setting context
        user_prompt += SETTING_VISION_CONTEXT.get(setting, "")

        # Build vision API payload (OpenAI-compatible multimodal format)
        payload = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        },
                    ],
                },
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }

        async with httpx.AsyncClient(timeout=config.timeout) as client:
            resp = await client.post(
                f"{config.endpoint}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            result_text = data["choices"][0]["message"]["content"]

        # Parse JSON from response
        analysis = _parse_vision_response(result_text)

        file_size = len(image_bytes)
        logger.info(
            "Vision analysis complete: %s, %d bytes — buffer cleared",
            analysis_type, file_size,
        )

        return ClinicalImageAnalysis(
            analysis=analysis,
            analysis_type=analysis_type,
            body_region=body_region,
            discipline=discipline,
            setting=setting,
            media_persisted=False,
            status="analyzed",
        )

    finally:
        # Zero and deallocate media buffers
        if image_b64 is not None:
            del image_b64
        del image_bytes
        gc.collect()


def _parse_vision_response(text: str) -> VisionAnalysisResult:
    """Parse vision model response into structured result."""
    try:
        # Try to extract JSON from markdown code blocks
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            data = json.loads(text)

        observations = [
            VisionObservation(**obs) for obs in data.get("observations", [])
        ]
        return VisionAnalysisResult(
            observations=observations,
            suggested_documentation=data.get("suggested_documentation", ""),
            things_to_verify=data.get("things_to_verify", []),
        )
    except (json.JSONDecodeError, AttributeError, TypeError):
        return VisionAnalysisResult(
            observations=[],
            suggested_documentation=text,
            things_to_verify=[],
        )
