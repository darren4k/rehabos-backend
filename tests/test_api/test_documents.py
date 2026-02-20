"""Tests for document extraction endpoints."""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import apply_auth_override

from rehab_os.api.routes.documents import (
    DocumentExtraction,
    MedicationEntry,
    DiagnosisEntry,
    LabResult,
    ImagingFinding,
    _build_section_mapping,
    _extract_pdf_text,
)


# ─── Unit tests for models ───────────────────────────────────────────────────


def test_medication_entry():
    m = MedicationEntry(name="Metformin", dosage="500mg", frequency="BID", route="oral")
    assert m.name == "Metformin"
    assert m.route == "oral"


def test_diagnosis_entry_defaults():
    d = DiagnosisEntry(description="Type 2 Diabetes")
    assert d.status == "active"
    assert d.icd_code is None


def test_lab_result():
    lr = LabResult(test_name="HbA1c", value="7.2", unit="%", flag="high")
    assert lr.flag == "high"


def test_imaging_finding():
    f = ImagingFinding(modality="x-ray", body_region="lumbar spine", findings="DDD L4-L5", impression="Mild degeneration")
    assert f.modality == "x-ray"


def test_document_extraction_minimal():
    e = DocumentExtraction(document_type="other", raw_text="test")
    assert e.confidence == 0.0
    assert e.medications is None


def test_document_extraction_full():
    e = DocumentExtraction(
        document_type="medication_list",
        confidence=0.95,
        medications=[MedicationEntry(name="Lisinopril", dosage="10mg")],
        raw_text="med list",
        note_section_mapping={"subjective": ["medications"]},
    )
    assert len(e.medications) == 1
    assert e.note_section_mapping["subjective"] == ["medications"]


# ─── Section mapping ─────────────────────────────────────────────────────────


def test_build_section_mapping_medication():
    data = {
        "document_type": "medication_list",
        "medications": [{"name": "Aspirin"}],
        "allergies": ["Penicillin"],
    }
    mapping = _build_section_mapping(data)
    assert "subjective" in mapping
    assert "medications" in mapping["subjective"]
    assert "allergies" in mapping["subjective"]


def test_build_section_mapping_labs():
    data = {
        "document_type": "lab_results",
        "lab_results": [{"test_name": "CBC", "value": "normal"}],
    }
    mapping = _build_section_mapping(data)
    assert "objective" in mapping
    assert "lab_results" in mapping["objective"]


def test_build_section_mapping_empty():
    data = {"document_type": "other"}
    mapping = _build_section_mapping(data)
    assert mapping == {}


def test_build_section_mapping_filters_absent():
    data = {
        "document_type": "discharge_summary",
        "diagnoses": [{"description": "CHF"}],
        # no medications, labs, etc.
    }
    mapping = _build_section_mapping(data)
    assert "assessment" in mapping
    assert "subjective" not in mapping


# ─── PDF text extraction ─────────────────────────────────────────────────────


def test_extract_pdf_text():
    """Test PDF text extraction with a minimal PDF."""
    # Create a minimal valid PDF
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test medication list")
        pdf_bytes = doc.tobytes()
        doc.close()

        text = _extract_pdf_text(pdf_bytes)
        assert "Test medication list" in text
    except ImportError:
        pytest.skip("pymupdf not installed")
