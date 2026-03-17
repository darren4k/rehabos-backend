"""X12 EDI segment builder for healthcare transactions.

Builds well-formed X12 EDI documents for 837P (claims), 270/271 (eligibility),
276/277 (claim status), 278 (prior auth), and 835 (remittance) transactions.

References:
- ASC X12N 837P Implementation Guide
- HIPAA Administrative Simplification Transaction Standards
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Standard X12 delimiters
ELEMENT_SEP = "*"
SEGMENT_TERM = "~"
SUB_ELEMENT_SEP = ":"
REPETITION_SEP = "^"


@dataclass
class X12Segment:
    """Single X12 EDI segment (e.g., ISA, GS, ST, CLM).

    Each segment consists of a segment identifier followed by data elements
    separated by the element separator, terminated by the segment terminator.
    """

    segment_id: str
    elements: list[str] = field(default_factory=list)

    def __init__(self, segment_id: str, *elements: str):
        self.segment_id = segment_id
        self.elements = list(elements)

    def render(self) -> str:
        """Render segment as X12 EDI string: 'ID*el1*el2*...*~'."""
        parts = [self.segment_id] + [str(e) if e is not None else "" for e in self.elements]
        return ELEMENT_SEP.join(parts) + SEGMENT_TERM

    def __repr__(self) -> str:
        return f"X12Segment({self.segment_id}, {len(self.elements)} elements)"


@dataclass
class X12Transaction:
    """Complete X12 transaction set.

    Contains an ordered list of segments forming a valid X12 interchange.
    """

    segments: list[X12Segment] = field(default_factory=list)

    def add(self, segment: X12Segment) -> None:
        """Append a segment to the transaction."""
        self.segments.append(segment)

    def render(self) -> str:
        """Render the full EDI document as a string."""
        return "\n".join(seg.render() for seg in self.segments)

    def render_inline(self) -> str:
        """Render without newlines (production format)."""
        return "".join(seg.render() for seg in self.segments)

    def segment_count(self) -> int:
        """Count segments excluding ISA/IEA/GS/GE envelope."""
        envelope_ids = {"ISA", "IEA", "GS", "GE"}
        return sum(1 for s in self.segments if s.segment_id not in envelope_ids)

    def validate(self) -> list[str]:
        """Validate transaction structure, returning error messages."""
        errors: list[str] = []

        if not self.segments:
            errors.append("Transaction has no segments")
            return errors

        # Check envelope structure
        seg_ids = [s.segment_id for s in self.segments]

        if seg_ids[0] != "ISA":
            errors.append("Transaction must start with ISA segment")
        if seg_ids[-1] != "IEA":
            errors.append("Transaction must end with IEA segment")

        # Check GS/GE pairing
        gs_count = seg_ids.count("GS")
        ge_count = seg_ids.count("GE")
        if gs_count != ge_count:
            errors.append(f"GS/GE mismatch: {gs_count} GS vs {ge_count} GE")

        # Check ST/SE pairing
        st_count = seg_ids.count("ST")
        se_count = seg_ids.count("SE")
        if st_count != se_count:
            errors.append(f"ST/SE mismatch: {st_count} ST vs {se_count} SE")

        # Validate ISA segment has 16 elements
        if seg_ids[0] == "ISA" and len(self.segments[0].elements) != 16:
            errors.append(
                f"ISA segment requires 16 elements, got {len(self.segments[0].elements)}"
            )

        # Validate SE segment count matches actual
        for i, seg in enumerate(self.segments):
            if seg.segment_id == "SE" and len(seg.elements) >= 1:
                try:
                    declared = int(seg.elements[0])
                    # Count segments between matching ST and this SE
                    actual = 0
                    for j in range(i - 1, -1, -1):
                        actual += 1
                        if self.segments[j].segment_id == "ST":
                            actual += 1  # include ST itself
                            break
                    if declared != actual:
                        errors.append(
                            f"SE declared {declared} segments but found {actual}"
                        )
                except ValueError:
                    errors.append("SE segment count is not a valid number")

        return errors


# ── Helper segment constructors ──────────────────────────────────────────────


def isa_segment(
    sender_id: str,
    receiver_id: str,
    *,
    interchange_control: str = "000000001",
    auth_qualifier: str = "00",
    auth_info: str = "          ",
    security_qualifier: str = "00",
    security_info: str = "          ",
    sender_qualifier: str = "ZZ",
    receiver_qualifier: str = "ZZ",
    usage_indicator: str = "T",  # T=test, P=production
    isa_version: str = "00501",
    repetition_sep: str = REPETITION_SEP,
) -> X12Segment:
    """Build ISA (Interchange Control Header) segment.

    Args:
        sender_id: Interchange sender ID (15 chars, right-padded).
        receiver_id: Interchange receiver ID (15 chars, right-padded).
        interchange_control: 9-digit control number.
        usage_indicator: 'T' for test, 'P' for production.
    """
    now = datetime.now(timezone.utc)
    return X12Segment(
        "ISA",
        auth_qualifier,
        auth_info,
        security_qualifier,
        security_info,
        sender_qualifier,
        sender_id.ljust(15),
        receiver_qualifier,
        receiver_id.ljust(15),
        now.strftime("%y%m%d"),  # Date YYMMDD
        now.strftime("%H%M"),  # Time HHMM
        repetition_sep,
        isa_version,
        interchange_control.zfill(9),
        "0",  # Acknowledgment requested
        usage_indicator,
        SUB_ELEMENT_SEP,
    )


def gs_segment(
    functional_id: str,
    sender_code: str,
    receiver_code: str,
    *,
    group_control: str = "1",
    version: str = "005010X222A1",
) -> X12Segment:
    """Build GS (Functional Group Header) segment.

    Args:
        functional_id: Functional identifier (HC=healthcare claim, HS=eligibility, HN=remittance).
        sender_code: Application sender code.
        receiver_code: Application receiver code.
        group_control: Group control number.
        version: Implementation guide version.
    """
    now = datetime.now(timezone.utc)
    return X12Segment(
        "GS",
        functional_id,
        sender_code,
        receiver_code,
        now.strftime("%Y%m%d"),  # Date CCYYMMDD
        now.strftime("%H%M"),  # Time HHMM
        group_control,
        "X",  # Responsible agency (X = X12)
        version,
    )


def st_segment(
    transaction_set_id: str,
    control_number: str,
    *,
    version: str = "005010X222A1",
) -> X12Segment:
    """Build ST (Transaction Set Header) segment.

    Args:
        transaction_set_id: '837' for claims, '270' for eligibility, etc.
        control_number: Transaction set control number (4-9 digits).
    """
    return X12Segment("ST", transaction_set_id, control_number.zfill(4), version)


def se_segment(segment_count: int, control_number: str) -> X12Segment:
    """Build SE (Transaction Set Trailer) segment."""
    return X12Segment("SE", str(segment_count), control_number.zfill(4))


def ge_segment(transaction_count: int, group_control: str) -> X12Segment:
    """Build GE (Functional Group Trailer) segment."""
    return X12Segment("GE", str(transaction_count), group_control)


def iea_segment(group_count: int, interchange_control: str) -> X12Segment:
    """Build IEA (Interchange Control Trailer) segment."""
    return X12Segment("IEA", str(group_count), interchange_control.zfill(9))


# ── Common segment builders ──────────────────────────────────────────────────


def nm1_segment(
    entity_id: str,
    entity_type: str,
    last_name: str,
    first_name: str = "",
    *,
    id_qualifier: str = "XX",
    id_code: str = "",
) -> X12Segment:
    """Build NM1 (Name) segment.

    Args:
        entity_id: Entity identifier code (41=submitter, 40=receiver, 85=billing, QC=patient, etc.).
        entity_type: '1' for person, '2' for non-person entity.
        last_name: Last name or organization name.
        first_name: First name (for persons).
        id_qualifier: Identification code qualifier (XX=NPI, MI=member ID).
        id_code: Identification code (NPI, member ID, etc.).
    """
    return X12Segment(
        "NM1",
        entity_id,
        entity_type,
        last_name,
        first_name,
        "",  # middle name
        "",  # prefix
        "",  # suffix
        id_qualifier,
        id_code,
    )


def ref_segment(qualifier: str, reference: str) -> X12Segment:
    """Build REF (Reference Identification) segment."""
    return X12Segment("REF", qualifier, reference)


def dmg_segment(dob: str, gender: str = "") -> X12Segment:
    """Build DMG (Demographic) segment.

    Args:
        dob: Date of birth in CCYYMMDD format.
        gender: M, F, or U.
    """
    return X12Segment("DMG", "D8", dob, gender)


def n3_segment(address_line1: str, address_line2: str = "") -> X12Segment:
    """Build N3 (Address) segment."""
    if address_line2:
        return X12Segment("N3", address_line1, address_line2)
    return X12Segment("N3", address_line1)


def n4_segment(city: str, state: str, zip_code: str) -> X12Segment:
    """Build N4 (City/State/ZIP) segment."""
    return X12Segment("N4", city, state, zip_code)


def dtp_segment(qualifier: str, date_format: str, date_value: str) -> X12Segment:
    """Build DTP (Date/Time) segment.

    Args:
        qualifier: Date qualifier (472=service date, 291=plan begin, etc.).
        date_format: 'D8' for CCYYMMDD, 'RD8' for date range.
        date_value: Date string.
    """
    return X12Segment("DTP", qualifier, date_format, date_value)
