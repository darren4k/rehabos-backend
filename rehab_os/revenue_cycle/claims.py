"""EDI 837P professional claim generation for rehabilitation services.

Generates X12 837P transactions from encounter data, with full support for
PT/OT/SLP billing including modifiers, place of service codes, and
CMS-1500 data extraction.

References:
- ASC X12N 837P (005010X222A1)
- CMS-1500 Form Instructions
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from rehab_os.revenue_cycle.edi_formatter import (
    X12Segment,
    X12Transaction,
    dtp_segment,
    ge_segment,
    gs_segment,
    iea_segment,
    isa_segment,
    n3_segment,
    n4_segment,
    nm1_segment,
    ref_segment,
    se_segment,
    st_segment,
)

logger = logging.getLogger(__name__)

# Place of Service codes per CMS
POS_CODES = {
    "office": "11",
    "outpatient": "11",
    "home": "12",
    "homecare": "12",
    "snf": "31",
    "outpatient_hospital": "22",
    "telehealth": "02",
}

# Discipline modifier mapping
DISCIPLINE_MODIFIERS = {
    "PT": "GP",
    "OT": "GO",
    "SLP": "GN",
}


@dataclass
class ClaimLine:
    """A single service line on a professional claim."""

    cpt_code: str
    modifier: str  # GP, GO, GN
    units: int
    charge_amount: float  # Per unit
    diagnosis_pointers: list[int] = field(default_factory=lambda: [1])
    service_date: str = ""  # CCYYMMDD
    place_of_service: str = "11"  # Default: office


@dataclass
class Claim:
    """Complete professional claim ready for 837P generation."""

    claim_id: str = ""
    patient_id: str = ""
    patient_first_name: str = ""
    patient_last_name: str = ""
    patient_dob: str = ""  # CCYYMMDD
    patient_gender: str = ""  # M, F, U
    patient_address: str = ""
    patient_city: str = ""
    patient_state: str = ""
    patient_zip: str = ""
    member_id: str = ""
    provider_npi: str = ""
    provider_first_name: str = ""
    provider_last_name: str = ""
    provider_tax_id: str = ""
    facility_npi: str | None = None
    facility_name: str = ""
    facility_address: str = ""
    facility_city: str = ""
    facility_state: str = ""
    facility_zip: str = ""
    referring_npi: str | None = None
    referring_first_name: str = ""
    referring_last_name: str = ""
    payer_id: str = ""
    payer_name: str = ""
    diagnosis_codes: list[str] = field(default_factory=list)  # ICD-10
    lines: list[ClaimLine] = field(default_factory=list)
    total_charge: float = 0.0
    place_of_service: str = "11"
    authorization_number: str | None = None
    setting: str = "outpatient"
    discipline: str = "PT"

    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = str(uuid.uuid4())[:13].replace("-", "").upper()
        if not self.place_of_service:
            self.place_of_service = POS_CODES.get(self.setting, "11")
        if self.total_charge == 0.0 and self.lines:
            self.total_charge = sum(
                line.charge_amount * line.units for line in self.lines
            )


class ClaimGenerator:
    """Generates 837P professional claims and CMS-1500 form data."""

    def __init__(
        self,
        sender_id: str = "",
        receiver_id: str = "",
        submitter_name: str = "REHABOS",
    ):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.submitter_name = submitter_name

    def generate_837p(self, claim: Claim) -> X12Transaction:
        """Generate a complete 837P X12 transaction for a professional claim.

        Args:
            claim: Populated Claim dataclass.

        Returns:
            X12Transaction ready for submission.
        """
        txn = X12Transaction()
        control = str(uuid.uuid4().int)[:9]
        st_control = control[:4]

        # ── Interchange envelope ──
        txn.add(isa_segment(self.sender_id, self.receiver_id, interchange_control=control))
        txn.add(gs_segment("HC", self.sender_id, self.receiver_id, group_control="1"))
        txn.add(st_segment("837", st_control))

        # ── BHT: Beginning of Hierarchical Transaction ──
        txn.add(X12Segment(
            "BHT", "0019", "00", claim.claim_id,
            datetime.now(timezone.utc).strftime("%Y%m%d"),
            datetime.now(timezone.utc).strftime("%H%M"),
            "CH",  # Chargeable
        ))

        # ── 1000A: Submitter ──
        txn.add(nm1_segment("41", "2", self.submitter_name, id_qualifier="46", id_code=self.sender_id))
        txn.add(X12Segment("PER", "IC", self.submitter_name, "TE", "0000000000"))

        # ── 1000B: Receiver ──
        txn.add(nm1_segment("40", "2", "CLEARINGHOUSE", id_qualifier="46", id_code=self.receiver_id))

        # ── 2000A: Billing Provider HL ──
        txn.add(X12Segment("HL", "1", "", "20", "1"))
        txn.add(X12Segment("PRV", "BI", "PXC", "261QP2300X"))  # PT taxonomy

        # ── 2010AA: Billing Provider Name ──
        billing_name = f"{claim.provider_last_name}, {claim.provider_first_name}"
        txn.add(nm1_segment("85", "1", claim.provider_last_name, claim.provider_first_name, id_qualifier="XX", id_code=claim.provider_npi))
        if claim.facility_address:
            txn.add(n3_segment(claim.facility_address))
            txn.add(n4_segment(claim.facility_city, claim.facility_state, claim.facility_zip))
        if claim.provider_tax_id:
            txn.add(ref_segment("EI", claim.provider_tax_id))

        # ── 2000B: Subscriber HL ──
        txn.add(X12Segment("HL", "2", "1", "22", "0"))
        txn.add(X12Segment("SBR", "P", "18", "", "", "", "", "", "", claim.payer_id))

        # ── 2010BA: Subscriber Name ──
        txn.add(nm1_segment("IL", "1", claim.patient_last_name, claim.patient_first_name, id_qualifier="MI", id_code=claim.member_id))
        if claim.patient_address:
            txn.add(n3_segment(claim.patient_address))
            txn.add(n4_segment(claim.patient_city, claim.patient_state, claim.patient_zip))
        if claim.patient_dob:
            from rehab_os.revenue_cycle.edi_formatter import dmg_segment
            txn.add(dmg_segment(claim.patient_dob, claim.patient_gender))

        # ── 2010BB: Payer Name ──
        txn.add(nm1_segment("PR", "2", claim.payer_name, id_qualifier="PI", id_code=claim.payer_id))

        # ── 2300: Claim ──
        freq_code = "1"  # Original claim
        txn.add(X12Segment(
            "CLM",
            claim.claim_id,
            f"{claim.total_charge:.2f}",
            "",
            "",
            f"{claim.place_of_service}:B:{freq_code}",
            "Y",  # Provider signature on file
            "A",  # Assignment accepted
            "Y",  # Benefits assigned
            "I",  # Release of information
        ))

        # Diagnosis codes (HI segment)
        if claim.diagnosis_codes:
            hi_elements = []
            for i, dx in enumerate(claim.diagnosis_codes[:12]):
                prefix = "ABK" if i == 0 else "ABF"
                hi_elements.append(f"{prefix}:{dx}")
            txn.add(X12Segment("HI", *hi_elements))

        # Referring provider
        if claim.referring_npi:
            txn.add(nm1_segment(
                "DN", "1", claim.referring_last_name, claim.referring_first_name,
                id_qualifier="XX", id_code=claim.referring_npi,
            ))

        # Prior authorization
        if claim.authorization_number:
            txn.add(ref_segment("G1", claim.authorization_number))

        # ── 2400: Service Lines ──
        for i, line in enumerate(claim.lines, start=1):
            # SV1: Professional Service
            modifier_str = line.modifier if line.modifier else ""
            composite_med = f"HC:{line.cpt_code}"
            if modifier_str:
                composite_med += f":{modifier_str}"

            dx_ptrs = ":".join(str(p) for p in line.diagnosis_pointers)

            txn.add(X12Segment("LX", str(i)))
            txn.add(X12Segment(
                "SV1",
                composite_med,
                f"{line.charge_amount * line.units:.2f}",
                "UN",  # Unit
                str(line.units),
                line.place_of_service or claim.place_of_service,
                "",
                dx_ptrs,
            ))

            svc_date = line.service_date or datetime.now(timezone.utc).strftime("%Y%m%d")
            txn.add(dtp_segment("472", "D8", svc_date))

        # ── Trailers ──
        # Count segments between ST and SE (inclusive)
        seg_count = len(txn.segments) - 2 + 1  # -ISA,-GS, +SE itself
        # More precise: count from ST to here, plus SE
        st_idx = next(i for i, s in enumerate(txn.segments) if s.segment_id == "ST")
        seg_count = len(txn.segments) - st_idx + 1  # +1 for SE itself

        txn.add(se_segment(seg_count, st_control))
        txn.add(ge_segment(1, "1"))
        txn.add(iea_segment(1, control))

        return txn

    def generate_cms1500_data(self, claim: Claim) -> dict:
        """Extract CMS-1500 form field data from a claim.

        Returns a dict mapping CMS-1500 box numbers to values,
        suitable for form filling or display.
        """
        lines_data = []
        for i, line in enumerate(claim.lines[:6], start=1):  # Max 6 lines on CMS-1500
            dx_letters = [chr(64 + p) for p in line.diagnosis_pointers]  # 1->A, 2->B
            lines_data.append({
                "line_number": i,
                "date_from": line.service_date,
                "date_to": line.service_date,
                "place_of_service": line.place_of_service or claim.place_of_service,
                "cpt_code": line.cpt_code,
                "modifier": line.modifier,
                "diagnosis_pointer": ",".join(dx_letters),
                "charges": f"{line.charge_amount * line.units:.2f}",
                "units": line.units,
            })

        return {
            "box_1": "Medicare" if "medicare" in claim.payer_name.lower() else "Group",
            "box_1a": claim.member_id,
            "box_2": f"{claim.patient_last_name}, {claim.patient_first_name}",
            "box_3": claim.patient_dob,
            "box_3_sex": claim.patient_gender,
            "box_5": f"{claim.patient_address}, {claim.patient_city}, {claim.patient_state} {claim.patient_zip}",
            "box_9": "",  # Other insured
            "box_11": claim.member_id,
            "box_11c": claim.payer_name,
            "box_17": f"{claim.referring_last_name}, {claim.referring_first_name}" if claim.referring_npi else "",
            "box_17b": claim.referring_npi or "",
            "box_21": {chr(65 + i): dx for i, dx in enumerate(claim.diagnosis_codes[:12])},
            "box_23": claim.authorization_number or "",
            "box_24": lines_data,
            "box_25": claim.provider_tax_id,
            "box_26": claim.claim_id,
            "box_28": f"{claim.total_charge:.2f}",
            "box_31": f"{claim.provider_last_name}, {claim.provider_first_name}",
            "box_32": claim.facility_name,
            "box_32a": claim.facility_npi or "",
            "box_33": f"{claim.provider_last_name}, {claim.provider_first_name}",
            "box_33a": claim.provider_npi,
        }

    def validate_claim(self, claim: Claim) -> list[str]:
        """Pre-submission validation of a claim.

        Returns a list of error messages. Empty list means the claim is valid.
        """
        errors: list[str] = []

        # Required fields
        if not claim.patient_id:
            errors.append("Missing patient_id")
        if not claim.provider_npi:
            errors.append("Missing provider NPI")
        if not claim.payer_id:
            errors.append("Missing payer ID")
        if not claim.diagnosis_codes:
            errors.append("At least one ICD-10 diagnosis code is required")
        if not claim.lines:
            errors.append("Claim has no service lines")
        if not claim.member_id:
            errors.append("Missing insurance member ID")

        # NPI format (10 digits)
        if claim.provider_npi and (
            len(claim.provider_npi) != 10 or not claim.provider_npi.isdigit()
        ):
            errors.append(f"Invalid provider NPI format: {claim.provider_npi}")

        # Validate service lines
        for i, line in enumerate(claim.lines, start=1):
            if not line.cpt_code:
                errors.append(f"Line {i}: missing CPT code")
            if line.units < 1:
                errors.append(f"Line {i}: units must be >= 1")
            if line.charge_amount <= 0:
                errors.append(f"Line {i}: charge amount must be > 0")
            # Validate diagnosis pointers reference existing codes
            for ptr in line.diagnosis_pointers:
                if ptr < 1 or ptr > len(claim.diagnosis_codes):
                    errors.append(
                        f"Line {i}: diagnosis pointer {ptr} out of range "
                        f"(claim has {len(claim.diagnosis_codes)} diagnoses)"
                    )

        # ICD-10 format check (letter + digits, with optional dot)
        for dx in claim.diagnosis_codes:
            clean = dx.replace(".", "")
            if not (len(clean) >= 3 and clean[0].isalpha() and clean[1:].isdigit()):
                errors.append(f"Invalid ICD-10 format: {dx}")

        # Place of service
        valid_pos = {"02", "11", "12", "22", "31"}
        if claim.place_of_service not in valid_pos:
            errors.append(
                f"Unknown place of service: {claim.place_of_service}. "
                f"Valid: {', '.join(sorted(valid_pos))}"
            )

        # Charge total consistency
        calculated = sum(l.charge_amount * l.units for l in claim.lines)
        if abs(calculated - claim.total_charge) > 0.01 and claim.total_charge != 0:
            errors.append(
                f"Total charge mismatch: declared ${claim.total_charge:.2f} "
                f"vs calculated ${calculated:.2f}"
            )

        # Modifier check for rehab
        modifier = DISCIPLINE_MODIFIERS.get(claim.discipline.upper())
        if modifier:
            for i, line in enumerate(claim.lines, start=1):
                if line.modifier and line.modifier != modifier:
                    errors.append(
                        f"Line {i}: modifier {line.modifier} doesn't match "
                        f"discipline {claim.discipline} (expected {modifier})"
                    )

        return errors
