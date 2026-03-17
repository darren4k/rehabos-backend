"""Tests for EDI 270/271 eligibility verification."""

import pytest

from rehab_os.revenue_cycle.eligibility import (
    EligibilityRequest,
    EligibilityResponse,
    EligibilityService,
    SERVICE_TYPE_CODES,
)


def _make_request(**overrides) -> EligibilityRequest:
    defaults = dict(
        patient_first_name="Jane",
        patient_last_name="Doe",
        patient_dob="19580315",
        member_id="MEM123456",
        payer_id="BCBS01",
        provider_npi="1234567890",
        service_type="PT",
    )
    defaults.update(overrides)
    return EligibilityRequest(**defaults)


# ---------------------------------------------------------------------------
# Stub mode (no clearinghouse)
# ---------------------------------------------------------------------------

class TestEligibilityStubMode:
    @pytest.mark.asyncio
    async def test_stub_when_unconfigured(self):
        svc = EligibilityService()
        assert not svc.is_configured
        resp = await svc.check_eligibility(_make_request())
        assert resp.is_stub is True
        assert resp.eligible is False
        assert resp.plan_name == "UNKNOWN"
        assert len(resp.warnings) == 1
        assert "Clearinghouse not configured" in resp.warnings[0]

    @pytest.mark.asyncio
    async def test_stub_preserves_member_id(self):
        svc = EligibilityService()
        resp = await svc.check_eligibility(_make_request(member_id="XYZ999"))
        assert resp.member_id == "XYZ999"

    @pytest.mark.asyncio
    async def test_configured_flag(self):
        svc = EligibilityService(clearinghouse_url="https://example.com", api_key="key")
        assert svc.is_configured is True


# ---------------------------------------------------------------------------
# Request payload building
# ---------------------------------------------------------------------------

class TestEligibilityRequestFields:
    def test_build_payload_has_required_fields(self):
        svc = EligibilityService(clearinghouse_url="https://ch.example.com", api_key="k")
        req = _make_request()
        payload = svc._build_request_payload(req)
        assert payload["provider_npi"] == "1234567890"
        assert payload["payer_id"] == "BCBS01"
        assert payload["subscriber"]["first_name"] == "Jane"
        assert payload["subscriber"]["member_id"] == "MEM123456"
        assert "transaction_id" in payload

    def test_service_type_code_pt(self):
        svc = EligibilityService(clearinghouse_url="https://ch.example.com", api_key="k")
        req = _make_request(service_type="PT")
        payload = svc._build_request_payload(req)
        assert payload["service_type_code"] == "BK"

    def test_service_type_code_ot(self):
        svc = EligibilityService(clearinghouse_url="https://ch.example.com", api_key="k")
        req = _make_request(service_type="OT")
        payload = svc._build_request_payload(req)
        assert payload["service_type_code"] == "BJ"

    def test_service_type_code_slp(self):
        svc = EligibilityService(clearinghouse_url="https://ch.example.com", api_key="k")
        req = _make_request(service_type="SLP")
        payload = svc._build_request_payload(req)
        assert payload["service_type_code"] == "BV"

    def test_service_type_code_unknown_falls_back(self):
        svc = EligibilityService(clearinghouse_url="https://ch.example.com", api_key="k")
        req = _make_request(service_type="UNKNOWN")
        payload = svc._build_request_payload(req)
        assert payload["service_type_code"] == "30"  # generic fallback


# ---------------------------------------------------------------------------
# Batch eligibility
# ---------------------------------------------------------------------------

class TestBatchEligibility:
    @pytest.mark.asyncio
    async def test_batch_returns_one_per_request(self):
        svc = EligibilityService()  # stub mode
        requests = [_make_request(member_id=f"M{i}") for i in range(3)]
        results = await svc.batch_check(requests)
        assert len(results) == 3
        assert all(r.is_stub for r in results)

    @pytest.mark.asyncio
    async def test_batch_preserves_member_ids(self):
        svc = EligibilityService()
        requests = [_make_request(member_id="A"), _make_request(member_id="B")]
        results = await svc.batch_check(requests)
        assert results[0].member_id == "A"
        assert results[1].member_id == "B"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestEligibilityResponseStructure:
    def test_parse_response_eligible(self):
        svc = EligibilityService(clearinghouse_url="https://ch.example.com", api_key="k")
        data = {
            "eligible": True,
            "plan_name": "Gold Plan",
            "coverage_start": "20260101",
            "coverage_end": "20261231",
            "benefits": {
                "copay": 25.0,
                "coinsurance_pct": 20.0,
                "deductible": 500.0,
                "deductible_met": 300.0,
                "visits": {"authorized": 30, "used": 5},
                "prior_auth_required": False,
            },
        }
        req = _make_request()
        resp = svc._parse_response(data, req)
        assert resp.eligible is True
        assert resp.plan_name == "Gold Plan"
        assert resp.copay == 25.0
        assert resp.visits_authorized == 30
        assert resp.visits_used == 5
        assert resp.visits_remaining == 25
        assert resp.prior_auth_required is False
        assert resp.warnings == []

    def test_parse_response_prior_auth_warning(self):
        svc = EligibilityService(clearinghouse_url="https://ch.example.com", api_key="k")
        data = {
            "eligible": True,
            "plan_name": "HMO Plan",
            "benefits": {
                "prior_auth_required": True,
                "visits": {},
            },
        }
        req = _make_request(service_type="PT")
        resp = svc._parse_response(data, req)
        assert resp.prior_auth_required is True
        assert any("Prior authorization required" in w for w in resp.warnings)

    def test_parse_response_low_visits_warning(self):
        svc = EligibilityService(clearinghouse_url="https://ch.example.com", api_key="k")
        data = {
            "eligible": True,
            "plan_name": "Plan X",
            "benefits": {
                "visits": {"authorized": 20, "used": 17},
                "prior_auth_required": False,
            },
        }
        req = _make_request()
        resp = svc._parse_response(data, req)
        assert resp.visits_remaining == 3
        assert any("3 authorized visits remaining" in w for w in resp.warnings)

    def test_error_response(self):
        svc = EligibilityService()
        req = _make_request()
        resp = svc._error_response(req, "Timeout occurred")
        assert resp.eligible is False
        assert resp.plan_name == "ERROR"
        assert resp.is_stub is True
        assert "Timeout occurred" in resp.warnings[0]

    def test_response_defaults(self):
        resp = EligibilityResponse()
        assert resp.eligible is False
        assert resp.copay is None
        assert resp.visits_remaining is None
        assert resp.warnings == []
        assert resp.is_stub is False
