"""EDI 835 remittance advice parsing for payment reconciliation.

Parses Electronic Remittance Advice (ERA) files to extract payment details,
denials, adjustments, and patient responsibility amounts.

References:
- ASC X12N 835 (005010X221A1)
- CARC/RARC code sets (Washington Publishing Company)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Common Claim Adjustment Reason Codes (CARC)
CARC_DESCRIPTIONS = {
    "1": "Deductible amount",
    "2": "Coinsurance amount",
    "3": "Co-payment amount",
    "4": "The procedure code is inconsistent with the modifier used",
    "16": "Claim/service lacks information or has submission errors",
    "18": "Exact duplicate claim/service",
    "22": "This care may be covered by another payer",
    "23": "Payment adjusted — charges exceed fee schedule/maximum allowable",
    "27": "Expenses incurred after coverage terminated",
    "29": "The time limit for filing has expired",
    "45": "Charge exceeds fee schedule/maximum allowable or contracted amount",
    "50": "Non-covered services because not deemed a medical necessity",
    "96": "Non-covered charge(s)",
    "97": "Payment adjusted — benefit not provided in current benefit plan",
    "109": "Claim not covered by this payer — submit to correct payer",
    "119": "Benefit maximum has been reached for this type of service",
    "197": "Precertification/authorization/notification absent",
    "204": "This service/equipment/drug is not covered under the patient's current benefit plan",
    "253": "Sequestration — reduction in federal payment",
}

# Claim Adjustment Group Codes
GROUP_CODES = {
    "CO": "Contractual Obligation",
    "CR": "Correction and Reversal",
    "OA": "Other Adjustment",
    "PI": "Payer Initiated Reduction",
    "PR": "Patient Responsibility",
}


@dataclass
class PaymentLine:
    """Single payment line from an ERA."""

    claim_id: str = ""
    cpt_code: str = ""
    billed_amount: float = 0.0
    allowed_amount: float = 0.0
    paid_amount: float = 0.0
    patient_responsibility: float = 0.0
    adjustment_reason: str | None = None  # e.g., "CO-45", "PR-1"
    adjustment_amount: float = 0.0
    denial_code: str | None = None
    remark_code: str | None = None

    @property
    def is_denied(self) -> bool:
        return self.paid_amount == 0.0 and self.billed_amount > 0.0

    @property
    def adjustment_description(self) -> str:
        """Human-readable adjustment description."""
        if not self.adjustment_reason:
            return ""
        parts = self.adjustment_reason.split("-")
        if len(parts) == 2:
            group = GROUP_CODES.get(parts[0], parts[0])
            reason = CARC_DESCRIPTIONS.get(parts[1], f"Code {parts[1]}")
            return f"{group}: {reason}"
        return self.adjustment_reason


@dataclass
class RemittanceAdvice:
    """Parsed ERA (Electronic Remittance Advice)."""

    era_id: str = ""
    payer_name: str = ""
    payer_id: str = ""
    check_number: str | None = None
    payment_date: str = ""
    total_paid: float = 0.0
    total_billed: float = 0.0
    total_allowed: float = 0.0
    total_patient_responsibility: float = 0.0
    lines: list[PaymentLine] = field(default_factory=list)

    @property
    def denials(self) -> list[PaymentLine]:
        """Lines with $0 paid (denials)."""
        return [line for line in self.lines if line.is_denied]

    @property
    def denial_count(self) -> int:
        return len(self.denials)

    @property
    def payment_rate(self) -> float:
        """Percentage of billed amount that was paid."""
        if self.total_billed == 0:
            return 0.0
        return (self.total_paid / self.total_billed) * 100

    @property
    def write_off_amount(self) -> float:
        """Total contractual write-off (billed - allowed)."""
        return max(0.0, self.total_billed - self.total_allowed)


class RemittanceParser:
    """Parses EDI 835 remittance advice content."""

    def parse_835(self, edi_content: str) -> RemittanceAdvice:
        """Parse an EDI 835 document into a RemittanceAdvice.

        Handles the standard 835 segment structure including:
        - ISA/GS/ST envelope
        - BPR (payment info)
        - N1 (payer identification)
        - CLP (claim-level payment)
        - SVC (service-level payment)
        - CAS (adjustment details)

        Args:
            edi_content: Raw EDI 835 text content.

        Returns:
            Parsed RemittanceAdvice with all payment lines.
        """
        era = RemittanceAdvice()

        # Normalize segment terminators
        content = edi_content.replace("\n", "").replace("\r", "")

        # Detect element separator from ISA segment
        element_sep = "*"
        if content.startswith("ISA"):
            element_sep = content[3]

        # Split into segments
        # Detect segment terminator from ISA (position 105)
        seg_term = "~"
        if len(content) > 105:
            seg_term = content[105]

        segments = [s.strip() for s in content.split(seg_term) if s.strip()]

        current_claim_id = ""
        current_claim_billed = 0.0
        current_claim_paid = 0.0
        current_claim_patient = 0.0
        current_cpt = ""

        for segment in segments:
            elements = segment.split(element_sep)
            seg_id = elements[0] if elements else ""

            if seg_id == "BPR" and len(elements) > 2:
                # BPR: Financial Information
                try:
                    era.total_paid = float(elements[2]) if elements[2] else 0.0
                except (ValueError, IndexError):
                    pass
                # Payment date (element 16)
                if len(elements) > 16 and elements[16]:
                    era.payment_date = elements[16]

            elif seg_id == "TRN" and len(elements) > 2:
                # TRN: Reassociation Trace Number (check/EFT number)
                era.era_id = elements[2] if len(elements) > 2 else ""
                era.check_number = elements[2] if len(elements) > 2 else None

            elif seg_id == "N1" and len(elements) > 2:
                # N1: Payer identification
                if elements[1] == "PR":  # Payer
                    era.payer_name = elements[2] if len(elements) > 2 else ""
                    era.payer_id = elements[4] if len(elements) > 4 else ""

            elif seg_id == "CLP" and len(elements) > 4:
                # CLP: Claim Payment Information
                current_claim_id = elements[1] if len(elements) > 1 else ""
                try:
                    current_claim_billed = float(elements[3]) if len(elements) > 3 and elements[3] else 0.0
                    current_claim_paid = float(elements[4]) if len(elements) > 4 and elements[4] else 0.0
                    current_claim_patient = float(elements[5]) if len(elements) > 5 and elements[5] else 0.0
                except (ValueError, IndexError):
                    pass
                era.total_billed += current_claim_billed

            elif seg_id == "SVC" and len(elements) > 3:
                # SVC: Service Payment Information
                # Element 1 is composite: HC:CPT:MODIFIER
                svc_composite = elements[1] if len(elements) > 1 else ""
                svc_parts = svc_composite.split(":")
                current_cpt = svc_parts[1] if len(svc_parts) > 1 else svc_parts[0]

                try:
                    billed = float(elements[2]) if len(elements) > 2 and elements[2] else 0.0
                    paid = float(elements[3]) if len(elements) > 3 and elements[3] else 0.0
                except (ValueError, IndexError):
                    billed = 0.0
                    paid = 0.0

                line = PaymentLine(
                    claim_id=current_claim_id,
                    cpt_code=current_cpt,
                    billed_amount=billed,
                    paid_amount=paid,
                    allowed_amount=paid,  # Updated by CAS if present
                )
                era.lines.append(line)

            elif seg_id == "CAS" and len(elements) > 3 and era.lines:
                # CAS: Claim Adjustment Segment
                group_code = elements[1] if len(elements) > 1 else ""
                reason_code = elements[2] if len(elements) > 2 else ""
                try:
                    adj_amount = float(elements[3]) if len(elements) > 3 and elements[3] else 0.0
                except (ValueError, IndexError):
                    adj_amount = 0.0

                last_line = era.lines[-1]
                last_line.adjustment_reason = f"{group_code}-{reason_code}"
                last_line.adjustment_amount = adj_amount

                if group_code == "PR":
                    last_line.patient_responsibility += adj_amount
                    era.total_patient_responsibility += adj_amount

                # Update allowed amount
                last_line.allowed_amount = last_line.billed_amount - adj_amount

                # Mark as denial if CO-50, CO-96, CO-97, etc.
                denial_reasons = {"50", "96", "97", "109", "119", "197", "204"}
                if reason_code in denial_reasons:
                    last_line.denial_code = reason_code

        # Calculate total_allowed from lines
        era.total_allowed = sum(l.allowed_amount for l in era.lines)

        return era

    def get_denials(self, era: RemittanceAdvice) -> list[PaymentLine]:
        """Extract denied service lines from an ERA."""
        return era.denials

    def calculate_write_offs(self, era: RemittanceAdvice) -> float:
        """Calculate total contractual write-off amount."""
        return era.write_off_amount

    def summarize(self, era: RemittanceAdvice) -> dict:
        """Generate a summary of the ERA for dashboard display."""
        return {
            "era_id": era.era_id,
            "payer": era.payer_name,
            "check_number": era.check_number,
            "payment_date": era.payment_date,
            "total_billed": round(era.total_billed, 2),
            "total_allowed": round(era.total_allowed, 2),
            "total_paid": round(era.total_paid, 2),
            "total_patient_responsibility": round(era.total_patient_responsibility, 2),
            "write_off": round(era.write_off_amount, 2),
            "payment_rate_pct": round(era.payment_rate, 1),
            "line_count": len(era.lines),
            "denial_count": era.denial_count,
        }
