"""Vision API — clinical image analysis via Qwen3-VL on DGX1.

Provides gait, ROM, wound, and posture analysis endpoints.
All media is ephemeral: processed in-memory, never persisted to disk.
"""
from __future__ import annotations

import gc
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from rehab_os.api.dependencies import get_current_user
from rehab_os.clinical.vision import ClinicalImageAnalysis, analyze_clinical_image
from rehab_os.core.models import Provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/analyze", response_model=ClinicalImageAnalysis)
async def analyze_image(
    file: UploadFile = File(...),
    analysis_type: str = Form("general"),
    discipline: str = Form("PT"),
    body_region: Optional[str] = Form(None),
    setting: str = Form("outpatient"),
    current_user: Provider = Depends(get_current_user),
):
    """Analyze a clinical image using Qwen3-VL.

    Media lifecycle: ephemeral.
    - Image received into memory buffer (no disk write).
    - Base64-encoded and sent to vision model.
    - Buffer zeroed and deallocated after analysis.

    Args:
        file: Image file (JPEG, PNG).
        analysis_type: general, gait, rom, wound, or posture.
        discipline: PT, OT, or SLP.
        body_region: Optional body region tag (e.g., "knee", "shoulder").
        setting: outpatient, homecare, or snf.
    """
    if analysis_type not in ("general", "gait", "rom", "wound", "posture"):
        raise HTTPException(400, "analysis_type must be: general, gait, rom, wound, or posture")

    image_bytes = None
    try:
        image_bytes = await file.read()
        content_type = file.content_type or "image/jpeg"

        result = await analyze_clinical_image(
            image_bytes=image_bytes,
            analysis_type=analysis_type,
            discipline=discipline,
            body_region=body_region,
            content_type=content_type,
            setting=setting,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Vision analysis failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if image_bytes is not None:
            del image_bytes
        gc.collect()


@router.post("/analyze-gait", response_model=ClinicalImageAnalysis)
async def analyze_gait(
    file: UploadFile = File(...),
    discipline: str = Form("PT"),
    setting: str = Form("outpatient"),
    current_user: Provider = Depends(get_current_user),
):
    """Analyze a gait image or video frame.

    Convenience endpoint that sets analysis_type=gait.
    """
    image_bytes = None
    try:
        image_bytes = await file.read()
        content_type = file.content_type or "image/jpeg"

        result = await analyze_clinical_image(
            image_bytes=image_bytes,
            analysis_type="gait",
            discipline=discipline,
            content_type=content_type,
            setting=setting,
        )

        return result

    except Exception as e:
        logger.exception("Gait analysis failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if image_bytes is not None:
            del image_bytes
        gc.collect()
