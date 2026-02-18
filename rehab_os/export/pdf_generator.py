"""fpdf2-based clinical note PDF renderer.

Generates in-memory PDF bytes for clinical SOAP notes.
No disk I/O — returns bytes directly via FPDF.output().
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fpdf import FPDF


NOTE_TYPE_LABELS = {
    "evaluation": "Evaluation",
    "daily_note": "Daily Note",
    "progress_note": "Progress Note",
    "recertification": "Recertification",
    "discharge_summary": "Discharge Summary",
}


class ClinicalNotePDF(FPDF):
    """PDF subclass with clinical document header/footer."""

    def __init__(self, clinic_name: str = "RehabOS Clinic"):
        super().__init__()
        self.clinic_name = clinic_name

    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 6, self.clinic_name, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 7)
        self.set_text_color(180, 0, 0)
        self.cell(0, 4, "CONFIDENTIAL - PROTECTED HEALTH INFORMATION", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.line(10, self.get_y() + 1, 200, self.get_y() + 1)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_note_pdf(
    *,
    note_type: str,
    note_date: str,
    discipline: str,
    therapist_name: Optional[str] = None,
    soap: dict[str, str],
    structured_data: Optional[dict] = None,
    compliance_score: Optional[float] = None,
    compliance_warnings: Optional[list[str]] = None,
    clinic_name: str = "RehabOS Clinic",
) -> bytes:
    """Generate a clinical note PDF and return raw bytes.

    Parameters
    ----------
    note_type : e.g. "daily_note", "evaluation"
    note_date : ISO date string
    discipline : e.g. "pt", "ot", "slp"
    therapist_name : Clinician name
    soap : Dict with keys subjective, objective, assessment, plan
    structured_data : Optional ROM/MMT/tests/billing dicts
    compliance_score : 0-100 float
    compliance_warnings : List of warning strings
    clinic_name : Header text

    Returns
    -------
    bytes : Raw PDF content
    """
    pdf = ClinicalNotePDF(clinic_name=clinic_name)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # --- Patient header block ---
    label = NOTE_TYPE_LABELS.get(note_type, note_type.replace("_", " ").title())
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, label, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    meta_parts = [f"Date: {note_date}", f"Discipline: {discipline.upper()}"]
    if therapist_name:
        meta_parts.append(f"Therapist: {therapist_name}")
    pdf.cell(0, 5, "  |  ".join(meta_parts), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- SOAP sections ---
    section_map = [
        ("S", "Subjective", soap.get("subjective", "")),
        ("O", "Objective", soap.get("objective", "")),
        ("A", "Assessment", soap.get("assessment", "")),
        ("P", "Plan", soap.get("plan", "")),
    ]
    for letter, title, content in section_map:
        if not content:
            continue
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(240, 240, 245)
        pdf.cell(0, 7, f"  {letter} - {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.ln(1)
        pdf.multi_cell(0, 5, _sanitize(content))
        pdf.ln(2)

    # --- Structured data tables ---
    sd = structured_data or {}

    if sd.get("rom"):
        _render_section_header(pdf, "Range of Motion")
        _render_table(pdf, ["Joint", "Motion", "Value", "Side"], [
            [r.get("joint", ""), r.get("motion", ""), _fmt_rom(r), r.get("side", "")]
            for r in sd["rom"]
        ])

    if sd.get("mmt"):
        _render_section_header(pdf, "Manual Muscle Testing")
        _render_table(pdf, ["Muscle Group", "Grade", "Side"], [
            [m.get("muscle_group", ""), m.get("grade", ""), m.get("side", "")]
            for m in sd["mmt"]
        ])

    if sd.get("standardized_tests"):
        _render_section_header(pdf, "Standardized Tests")
        _render_table(pdf, ["Test", "Score", "Interpretation"], [
            [
                t.get("name", ""),
                f"{t.get('score', '')}{('/' + str(t['max_score'])) if t.get('max_score') else ''} {t.get('unit', '')}".strip(),
                t.get("interpretation", ""),
            ]
            for t in sd["standardized_tests"]
        ])

    if sd.get("billing_codes"):
        _render_section_header(pdf, "Billing Codes")
        _render_table(pdf, ["Code", "Description", "Units"], [
            [
                b.get("code") or b.get("cpt_code", ""),
                b.get("description", ""),
                str(b.get("units", "")),
            ]
            for b in sd["billing_codes"]
        ])

    # --- Compliance section ---
    if compliance_score is not None:
        pdf.ln(2)
        _render_section_header(pdf, "Compliance")
        pdf.set_font("Helvetica", "", 9)
        score_text = f"Compliance Score: {compliance_score:.0f}%"
        pdf.cell(0, 5, score_text, new_x="LMARGIN", new_y="NEXT")
        if compliance_warnings:
            for w in compliance_warnings:
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(180, 120, 0)
                pdf.cell(0, 4, f"  - {_sanitize(w)}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    # --- Signature line ---
    pdf.ln(8)
    pdf.line(10, pdf.get_y(), 90, pdf.get_y())
    pdf.set_font("Helvetica", "", 8)
    pdf.ln(1)
    sig_parts = [therapist_name or "Therapist Signature"]
    sig_parts.append(f"Date: {note_date}")
    pdf.cell(0, 4, "  |  ".join(sig_parts), new_x="LMARGIN", new_y="NEXT")

    # --- Generated timestamp ---
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 4, f"Generated by RehabOS on {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


# --- Helpers ---

def _sanitize(text: str) -> str:
    """Replace Unicode characters that Helvetica (latin-1) can't render."""
    return (
        text
        .replace("\u2014", "-")   # em-dash
        .replace("\u2013", "-")   # en-dash
        .replace("\u00b0", " deg")  # degree sign (not in latin-1 for fpdf2 built-in)
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def _render_section_header(pdf: FPDF, title: str) -> None:
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 6, title.upper(), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def _render_table(pdf: FPDF, headers: list[str], rows: list[list[str]]) -> None:
    col_w = (pdf.w - 20) / len(headers)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(230, 230, 235)
    for h in headers:
        pdf.cell(col_w, 5, h, border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for row in rows:
        for cell in row:
            pdf.cell(col_w, 5, _sanitize(str(cell))[:40], border=1)
        pdf.ln()
    pdf.ln(2)


def _fmt_rom(r: dict) -> str:
    if r.get("value") is not None:
        return f"{r['value']} deg"
    return r.get("qualitative", "-")
