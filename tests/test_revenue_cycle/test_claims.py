"""Tests for EDI 837P professional claim generation."""

import pytest

from rehab_os.revenue_cycle.claims import (
    Claim,
    ClaimGenerator,
    ClaimLine,
    DISCIPLINE_MODIFIERS,
    POS_CODES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_claim(**overrides) -> Claim:
    """Create a valid claim with sensible defaults, overridable."""
    defaults = dict(
        patient_id="PAT001",
        patient_first_name="Jane",
        patient_last_name="Doe",
        patient_dob="19580315",
        patient_gender="F",
        patient_address="123 Main St",
        patient_city="Springfield",
        patient_state="IL",
        patient_zip="62701",
        member_id="MEM123456",
        provider_npi="1234567890",
        provider_first_name="John",
        provider_last_name="Smith",
        provider_tax_id="987654321",
        facility_npi="0987654321",
        facility_name="RehabOS Clinic",
        facility_address="456 Oak Ave",
        facility_city="Springfield",
        facility_state="IL",
        facility_zip="62702",
        payer_id="BCBS01",
        payer_name="Blue Cross Blue Shield",
        diagnosis_codes=["M54.5", "M79.3"],
        discipline="PT",
        place_of_service="11",
        lines=[
            ClaimLine(
                cpt_code="97110",
                modifier="GP",
                units=2,
                charge_amount=50.00,
                diagnosis_pointers=[1],
                service_date="20260315",
                place_of_service="11",
            ),
            ClaimLine(
                cpt_code="97140",
                modifier="GP",
                units=1,
                charge_amount=45.00,
                diagnosis_pointers=[1, 2],
                service_date="20260315",
                place_of_service="11",
            ),
        ],
    )
    defaults.update(overrides)
    return Claim(**defaults)


# ---------------------------------------------------------------------------
# Claim validation
# ---------------------------------------------------------------------------

class TestClaimValidation:
    def test_valid_claim_no_errors(self):
        claim = _make_claim()
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert errors == []

    def test_invalid_npi_too_short(self):
        claim = _make_claim(provider_npi="12345")
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("Invalid provider NPI" in e for e in errors)

    def test_invalid_npi_alpha(self):
        claim = _make_claim(provider_npi="123456789A")
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("Invalid provider NPI" in e for e in errors)

    def test_invalid_icd10_bad_format(self):
        claim = _make_claim(diagnosis_codes=["INVALID"])
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("Invalid ICD-10" in e for e in errors)

    def test_valid_icd10_with_dot(self):
        claim = _make_claim(diagnosis_codes=["M54.5"])
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        icd_errors = [e for e in errors if "ICD-10" in e]
        assert icd_errors == []

    def test_valid_icd10_without_dot(self):
        claim = _make_claim(diagnosis_codes=["M545"])
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        icd_errors = [e for e in errors if "ICD-10" in e]
        assert icd_errors == []

    def test_missing_patient_id(self):
        claim = _make_claim(patient_id="")
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("Missing patient_id" in e for e in errors)

    def test_missing_payer_id(self):
        claim = _make_claim(payer_id="")
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("Missing payer ID" in e for e in errors)

    def test_missing_diagnosis_codes(self):
        claim = _make_claim(diagnosis_codes=[])
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("ICD-10 diagnosis code" in e for e in errors)

    def test_missing_service_lines(self):
        claim = _make_claim(lines=[])
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("no service lines" in e for e in errors)

    def test_missing_member_id(self):
        claim = _make_claim(member_id="")
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("Missing insurance member ID" in e for e in errors)

    def test_line_missing_cpt(self):
        line = ClaimLine(cpt_code="", modifier="GP", units=1, charge_amount=50.0)
        claim = _make_claim(lines=[line])
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("missing CPT code" in e for e in errors)

    def test_line_zero_units(self):
        line = ClaimLine(cpt_code="97110", modifier="GP", units=0, charge_amount=50.0)
        claim = _make_claim(lines=[line])
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("units must be >= 1" in e for e in errors)

    def test_line_zero_charge(self):
        line = ClaimLine(cpt_code="97110", modifier="GP", units=1, charge_amount=0.0)
        claim = _make_claim(lines=[line])
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("charge amount must be > 0" in e for e in errors)

    def test_invalid_place_of_service(self):
        claim = _make_claim(place_of_service="99")
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("Unknown place of service" in e for e in errors)

    def test_modifier_mismatch_discipline(self):
        line = ClaimLine(cpt_code="97110", modifier="GO", units=1, charge_amount=50.0)
        claim = _make_claim(discipline="PT", lines=[line])
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("modifier GO doesn't match discipline PT" in e for e in errors)

    def test_diagnosis_pointer_out_of_range(self):
        line = ClaimLine(cpt_code="97110", modifier="GP", units=1, charge_amount=50.0,
                         diagnosis_pointers=[5])
        claim = _make_claim(diagnosis_codes=["M54.5"], lines=[line])
        gen = ClaimGenerator()
        errors = gen.validate_claim(claim)
        assert any("diagnosis pointer 5 out of range" in e for e in errors)


# ---------------------------------------------------------------------------
# Claim line charge calculation
# ---------------------------------------------------------------------------

class TestClaimLineCharge:
    def test_total_charge_auto_calculated(self):
        lines = [
            ClaimLine(cpt_code="97110", modifier="GP", units=3, charge_amount=50.0),
            ClaimLine(cpt_code="97140", modifier="GP", units=2, charge_amount=45.0),
        ]
        claim = Claim(lines=lines, patient_id="P1", provider_npi="1234567890",
                      payer_id="X", member_id="M", diagnosis_codes=["M54.5"])
        # 3*50 + 2*45 = 240
        assert claim.total_charge == 240.0

    def test_total_charge_not_overridden_when_zero(self):
        lines = [ClaimLine(cpt_code="97110", modifier="GP", units=1, charge_amount=100.0)]
        claim = Claim(lines=lines, total_charge=0.0)
        assert claim.total_charge == 100.0

    def test_total_charge_kept_when_explicit(self):
        lines = [ClaimLine(cpt_code="97110", modifier="GP", units=1, charge_amount=100.0)]
        claim = Claim(lines=lines, total_charge=999.0)
        assert claim.total_charge == 999.0

    def test_claim_id_auto_generated(self):
        claim = Claim()
        assert claim.claim_id  # non-empty
        assert len(claim.claim_id) > 0


# ---------------------------------------------------------------------------
# 837P generation
# ---------------------------------------------------------------------------

class TestGenerate837P:
    def test_structure_has_envelope(self):
        claim = _make_claim()
        gen = ClaimGenerator(sender_id="SENDER1", receiver_id="RECVR1")
        txn = gen.generate_837p(claim)
        seg_ids = [s.segment_id for s in txn.segments]
        assert seg_ids[0] == "ISA"
        assert "GS" in seg_ids
        assert "ST" in seg_ids
        assert "SE" in seg_ids
        assert "GE" in seg_ids
        assert seg_ids[-1] == "IEA"

    def test_structure_has_clm_segment(self):
        claim = _make_claim()
        gen = ClaimGenerator()
        txn = gen.generate_837p(claim)
        seg_ids = [s.segment_id for s in txn.segments]
        assert "CLM" in seg_ids

    def test_structure_has_hi_segment(self):
        claim = _make_claim()
        gen = ClaimGenerator()
        txn = gen.generate_837p(claim)
        seg_ids = [s.segment_id for s in txn.segments]
        assert "HI" in seg_ids

    def test_structure_has_sv1_lines(self):
        claim = _make_claim()
        gen = ClaimGenerator()
        txn = gen.generate_837p(claim)
        sv1_segs = [s for s in txn.segments if s.segment_id == "SV1"]
        assert len(sv1_segs) == 2  # 2 service lines

    def test_bht_segment_present(self):
        claim = _make_claim()
        gen = ClaimGenerator()
        txn = gen.generate_837p(claim)
        bht = [s for s in txn.segments if s.segment_id == "BHT"]
        assert len(bht) == 1
        assert bht[0].elements[0] == "0019"

    def test_auth_number_creates_ref_g1(self):
        claim = _make_claim(authorization_number="AUTH123")
        gen = ClaimGenerator()
        txn = gen.generate_837p(claim)
        ref_segs = [s for s in txn.segments if s.segment_id == "REF" and s.elements[0] == "G1"]
        assert len(ref_segs) == 1
        assert ref_segs[0].elements[1] == "AUTH123"

    def test_referring_provider_creates_nm1_dn(self):
        claim = _make_claim(referring_npi="9876543210", referring_first_name="Dr",
                            referring_last_name="Referrer")
        gen = ClaimGenerator()
        txn = gen.generate_837p(claim)
        dn_segs = [s for s in txn.segments if s.segment_id == "NM1" and s.elements[0] == "DN"]
        assert len(dn_segs) == 1


# ---------------------------------------------------------------------------
# CMS-1500 form data
# ---------------------------------------------------------------------------

class TestCMS1500:
    def test_all_boxes_populated(self):
        claim = _make_claim()
        gen = ClaimGenerator()
        data = gen.generate_cms1500_data(claim)
        expected_keys = [
            "box_1", "box_1a", "box_2", "box_3", "box_3_sex", "box_5",
            "box_9", "box_11", "box_11c", "box_17", "box_17b", "box_21",
            "box_23", "box_24", "box_25", "box_26", "box_28",
            "box_31", "box_32", "box_32a", "box_33", "box_33a",
        ]
        for key in expected_keys:
            assert key in data, f"Missing CMS-1500 {key}"

    def test_box_2_patient_name(self):
        claim = _make_claim()
        gen = ClaimGenerator()
        data = gen.generate_cms1500_data(claim)
        assert data["box_2"] == "Doe, Jane"

    def test_box_21_diagnosis_codes(self):
        claim = _make_claim(diagnosis_codes=["M54.5", "M79.3"])
        gen = ClaimGenerator()
        data = gen.generate_cms1500_data(claim)
        assert data["box_21"]["A"] == "M54.5"
        assert data["box_21"]["B"] == "M79.3"

    def test_box_24_service_lines(self):
        claim = _make_claim()
        gen = ClaimGenerator()
        data = gen.generate_cms1500_data(claim)
        lines = data["box_24"]
        assert len(lines) == 2
        assert lines[0]["cpt_code"] == "97110"
        assert lines[0]["modifier"] == "GP"

    def test_box_28_total_charge(self):
        claim = _make_claim()
        gen = ClaimGenerator()
        data = gen.generate_cms1500_data(claim)
        expected = f"{claim.total_charge:.2f}"
        assert data["box_28"] == expected

    def test_box_33a_provider_npi(self):
        claim = _make_claim()
        gen = ClaimGenerator()
        data = gen.generate_cms1500_data(claim)
        assert data["box_33a"] == "1234567890"

    def test_box_1_medicare(self):
        claim = _make_claim(payer_name="Medicare Part B")
        gen = ClaimGenerator()
        data = gen.generate_cms1500_data(claim)
        assert data["box_1"] == "Medicare"

    def test_box_1_group(self):
        claim = _make_claim(payer_name="Blue Cross Blue Shield")
        gen = ClaimGenerator()
        data = gen.generate_cms1500_data(claim)
        assert data["box_1"] == "Group"


# ---------------------------------------------------------------------------
# Place of service mapping
# ---------------------------------------------------------------------------

class TestPlaceOfService:
    def test_office(self):
        assert POS_CODES["office"] == "11"

    def test_home(self):
        assert POS_CODES["home"] == "12"

    def test_snf(self):
        assert POS_CODES["snf"] == "31"

    def test_telehealth(self):
        assert POS_CODES["telehealth"] == "02"

    def test_outpatient_hospital(self):
        assert POS_CODES["outpatient_hospital"] == "22"


# ---------------------------------------------------------------------------
# Modifier mapping
# ---------------------------------------------------------------------------

class TestModifiers:
    def test_pt_modifier(self):
        assert DISCIPLINE_MODIFIERS["PT"] == "GP"

    def test_ot_modifier(self):
        assert DISCIPLINE_MODIFIERS["OT"] == "GO"

    def test_slp_modifier(self):
        assert DISCIPLINE_MODIFIERS["SLP"] == "GN"
