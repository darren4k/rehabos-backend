"""API routes for document upload and AI data extraction."""

import base64
import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from rehab_os.api.audit import log_phi_access
from rehab_os.api.dependencies import get_current_user
from rehab_os.core.models import Provider

logger = logging.getLogger(__name__)

router = APIRouter()

# ─── Response Models ──────────────────────────────────────────────────────────


class MedicationEntry(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    prescriber: Optional[str] = None
    start_date: Optional[str] = None
    indication: Optional[str] = None


class DiagnosisEntry(BaseModel):
    description: str
    icd_code: Optional[str] = None
    onset_date: Optional[str] = None
    status: str = "active"


class LabResult(BaseModel):
    test_name: str
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    flag: Optional[str] = None
    date: Optional[str] = None


class ImagingFinding(BaseModel):
    modality: str
    body_region: str
    findings: str
    impression: str
    date: Optional[str] = None


class DocumentExtraction(BaseModel):
    document_type: str
    confidence: float = 0.0
    medications: Optional[list[MedicationEntry]] = None
    diagnoses: Optional[list[DiagnosisEntry]] = None
    lab_results: Optional[list[LabResult]] = None
    imaging_findings: Optional[list[ImagingFinding]] = None
    demographics: Optional[dict] = None
    insurance: Optional[dict] = None
    past_medical_history: Optional[list[str]] = None
    surgical_history: Optional[list[str]] = None
    allergies: Optional[list[str]] = None
    vitals: Optional[dict] = None
    recommendations: Optional[list[str]] = None
    raw_text: str = ""
    note_section_mapping: dict[str, list[str]] = {}


class PopulateResult(BaseModel):
    extraction: DocumentExtraction
    sections_updated: dict[str, list[str]]


# ─── Extraction Prompt ────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a clinical data extraction specialist. Extract ALL structured medical data from this document.

For medication lists, extract: drug name, dosage, frequency, route, indication if mentioned.
For discharge summaries, extract: diagnoses with ICD-10 codes, procedures, medications, follow-up instructions, restrictions.
For lab results, extract: test name, value with units, reference range, whether it's flagged abnormal.
For imaging reports, extract: modality, body region, key findings, radiologist impression.

Return ONLY valid JSON (no markdown fences):
{
  "document_type": "medication_list|discharge_summary|lab_results|imaging|insurance|referral|other",
  "confidence": 0.95,
  "medications": [{"name": "...", "dosage": "...", "frequency": "...", "route": "...", "prescriber": "...", "start_date": "...", "indication": "..."}],
  "diagnoses": [{"description": "...", "icd_code": "...", "onset_date": "...", "status": "active|resolved|chronic"}],
  "lab_results": [{"test_name": "...", "value": "...", "unit": "...", "reference_range": "...", "flag": "normal|high|low|critical", "date": "..."}],
  "imaging_findings": [{"modality": "...", "body_region": "...", "findings": "...", "impression": "...", "date": "..."}],
  "demographics": {"name": "...", "dob": "...", "address": "...", "phone": "..."},
  "insurance": {"payer": "...", "member_id": "...", "group": "..."},
  "past_medical_history": ["..."],
  "surgical_history": ["..."],
  "allergies": ["..."],
  "vitals": {"bp": "...", "hr": "...", "temp": "...", "rr": "...", "spo2": "..."},
  "recommendations": ["..."]
}
Omit empty arrays/objects. Include only what's actually present in the document."""

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/heic",
    "application/pdf",
}

# ─── Section mapping logic ────────────────────────────────────────────────────

SECTION_MAP: dict[str, dict[str, list[str]]] = {
    "medication_list": {"subjective": ["medications", "allergies"]},
    "discharge_summary": {
        "subjective": ["medications", "allergies", "past_medical_history", "surgical_history"],
        "objective": ["vitals", "lab_results"],
        "assessment": ["diagnoses"],
        "plan": ["recommendations"],
    },
    "lab_results": {"objective": ["lab_results", "vitals"]},
    "imaging": {"objective": ["imaging_findings"]},
    "insurance": {"demographics": ["insurance"]},
    "referral": {
        "subjective": ["past_medical_history", "medications"],
        "assessment": ["diagnoses"],
        "plan": ["recommendations"],
    },
    "other": {},
}


def _build_section_mapping(extraction: dict) -> dict[str, list[str]]:
    doc_type = extraction.get("document_type", "other")
    base = SECTION_MAP.get(doc_type, SECTION_MAP["other"]).copy()
    # Filter to only sections that have data
    result: dict[str, list[str]] = {}
    for section, fields in base.items():
        present = [f for f in fields if extraction.get(f)]
        if present:
            result[section] = present
    return result


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF using pymupdf."""
    import fitz  # pymupdf

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


async def _call_llm_for_extraction(
    request: Request,
    file_bytes: bytes,
    content_type: str,
) -> DocumentExtraction:
    """Send document to LLM and parse extraction result."""
    llm_router = request.app.state.llm_router
    # Access the primary LLM's OpenAI client directly for vision support
    primary_llm = llm_router.primary
    client = primary_llm._client

    is_pdf = content_type == "application/pdf"

    if is_pdf:
        raw_text = _extract_pdf_text(file_bytes)
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": f"Extract structured data from this document:\n\n{raw_text}"},
        ]
    else:
        # Image — use vision
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        mime = content_type if content_type != "image/heic" else "image/jpeg"
        raw_text = "[image document]"
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract structured data from this document image."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ]

    try:
        response = await client.chat.completions.create(
            model=primary_llm._model,
            messages=messages,
            temperature=0.2,
            max_tokens=4096,
        )
        content = response.choices[0].message.content or "{}"
    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        raise HTTPException(status_code=502, detail="Document extraction failed")

    # Parse JSON from response (strip markdown fences if present)
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON: %s", text[:200])
        return DocumentExtraction(
            document_type="other",
            confidence=0.0,
            raw_text=raw_text if is_pdf else "[image]",
            note_section_mapping={},
        )

    data["raw_text"] = raw_text if is_pdf else "[image document]"
    data["note_section_mapping"] = _build_section_mapping(data)

    try:
        return DocumentExtraction(**data)
    except Exception as e:
        logger.warning("Failed to parse extraction into model: %s", e)
        return DocumentExtraction(
            document_type=data.get("document_type", "other"),
            confidence=data.get("confidence", 0.0),
            raw_text=data.get("raw_text", ""),
            note_section_mapping=data.get("note_section_mapping", {}),
        )


# ─── Routes ──────────────────────────────────────────────────────────────────


@router.post("/documents/extract", response_model=DocumentExtraction)
async def extract_document(
    request: Request,
    file: UploadFile = File(...),
    current_user: Provider = Depends(get_current_user),
):
    """Upload an image or PDF and extract structured clinical data."""
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    extraction = await _call_llm_for_extraction(request, file_bytes, content_type)
    log_phi_access(
        user_id=str(current_user.id),
        action="create",
        resource_type="document_extraction",
        resource_id="",
        ip_address=request.client.host if request.client else "",
    )
    return extraction


@router.post("/documents/extract-and-populate", response_model=PopulateResult)
async def extract_and_populate(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    current_user: Provider = Depends(get_current_user),
):
    """Extract document data and merge into an active note session."""
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    extraction = await _call_llm_for_extraction(request, file_bytes, content_type)

    # Build sections_updated from the mapping
    sections_updated: dict[str, list[str]] = {}
    mapping = extraction.note_section_mapping

    for section, fields in mapping.items():
        items: list[str] = []
        for field_name in fields:
            val = getattr(extraction, field_name, None)
            if val is None:
                continue
            if isinstance(val, list):
                for entry in val:
                    if isinstance(entry, BaseModel):
                        items.append(entry.model_dump_json())
                    else:
                        items.append(str(entry))
            elif isinstance(val, dict):
                items.append(json.dumps(val))
        if items:
            sections_updated[section] = items

    log_phi_access(
        user_id=str(current_user.id),
        action="create",
        resource_type="document_extraction",
        resource_id=session_id,
        ip_address=request.client.host if request.client else "",
    )
    return PopulateResult(extraction=extraction, sections_updated=sections_updated)
