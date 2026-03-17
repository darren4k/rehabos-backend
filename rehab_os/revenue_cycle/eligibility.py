"""EDI 270/271 eligibility verification for rehabilitation services.

Checks patient insurance eligibility and benefits for PT/OT/SLP services.
Gracefully degrades to stub responses when clearinghouse is not configured.

References:
- ASC X12N 270/271 (005010X279A1)
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# Service type codes for rehab disciplines
SERVICE_TYPE_CODES = {
    "PT": "BK",   # Physical Therapy
    "OT": "BJ",   # Occupational Therapy
    "SLP": "BV",  # Speech Therapy
}


@dataclass
class EligibilityRequest:
    """Request payload for 270 eligibility inquiry."""

    patient_first_name: str
    patient_last_name: str
    patient_dob: str  # CCYYMMDD
    member_id: str
    payer_id: str
    provider_npi: str
    service_type: str = "PT"  # PT, OT, SLP


@dataclass
class EligibilityResponse:
    """Parsed 271 eligibility response."""

    eligible: bool = False
    plan_name: str = ""
    member_id: str = ""
    coverage_start: str | None = None
    coverage_end: str | None = None
    copay: float | None = None
    coinsurance_pct: float | None = None
    deductible: float | None = None
    deductible_met: float | None = None
    visits_authorized: int | None = None
    visits_used: int | None = None
    visits_remaining: int | None = None
    prior_auth_required: bool = False
    authorization_number: str | None = None
    raw_response: dict | None = None
    warnings: list[str] = field(default_factory=list)
    is_stub: bool = False  # True if response is from stub (clearinghouse unavailable)


class EligibilityService:
    """EDI 270/271 eligibility checker.

    Submits eligibility inquiries to a clearinghouse and parses responses.
    Falls back to stub responses with warnings if the clearinghouse is
    not configured or unavailable.
    """

    def __init__(self, clearinghouse_url: str = "", api_key: str = ""):
        """Initialize eligibility service.

        Args:
            clearinghouse_url: Clearinghouse API endpoint for eligibility checks.
            api_key: API key for clearinghouse authentication.
        """
        self.clearinghouse_url = clearinghouse_url.rstrip("/") if clearinghouse_url else ""
        self.api_key = api_key
        self._configured = bool(self.clearinghouse_url and self.api_key)

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def check_eligibility(
        self, request: EligibilityRequest
    ) -> EligibilityResponse:
        """Check eligibility for a single patient.

        If the clearinghouse is not configured, returns a stub response
        with is_stub=True and a warning.
        """
        if not self._configured:
            return self._stub_response(request)

        try:
            payload = self._build_request_payload(request)
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.clearinghouse_url}/eligibility/270",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                return self._parse_response(resp.json(), request)

        except httpx.TimeoutException:
            logger.warning("Eligibility check timed out for member %s", request.member_id)
            return self._error_response(request, "Clearinghouse request timed out")
        except httpx.HTTPStatusError as e:
            logger.error("Eligibility check HTTP error: %s", e.response.status_code)
            return self._error_response(request, f"Clearinghouse HTTP error: {e.response.status_code}")
        except Exception as e:
            logger.exception("Eligibility check failed for member %s", request.member_id)
            return self._error_response(request, f"Eligibility check failed: {str(e)}")

    async def batch_check(
        self, requests: list[EligibilityRequest]
    ) -> list[EligibilityResponse]:
        """Check eligibility for multiple patients.

        Processes sequentially to respect clearinghouse rate limits.
        """
        results = []
        for req in requests:
            result = await self.check_eligibility(req)
            results.append(result)
        return results

    def _build_request_payload(self, request: EligibilityRequest) -> dict:
        """Build clearinghouse API payload from eligibility request."""
        svc_code = SERVICE_TYPE_CODES.get(request.service_type.upper(), "30")
        return {
            "transaction_id": str(uuid.uuid4()),
            "provider_npi": request.provider_npi,
            "payer_id": request.payer_id,
            "subscriber": {
                "first_name": request.patient_first_name,
                "last_name": request.patient_last_name,
                "dob": request.patient_dob,
                "member_id": request.member_id,
            },
            "service_type_code": svc_code,
            "date_of_service": datetime.now(timezone.utc).strftime("%Y%m%d"),
        }

    def _parse_response(
        self, data: dict, request: EligibilityRequest
    ) -> EligibilityResponse:
        """Parse clearinghouse JSON response into EligibilityResponse."""
        benefits = data.get("benefits", {})
        visits = benefits.get("visits", {})

        warnings = []
        prior_auth = benefits.get("prior_auth_required", False)
        if prior_auth:
            warnings.append(
                f"Prior authorization required for {request.service_type} services"
            )

        visits_auth = visits.get("authorized")
        visits_used = visits.get("used")
        visits_remaining = None
        if visits_auth is not None and visits_used is not None:
            visits_remaining = max(0, visits_auth - visits_used)
            if visits_remaining <= 5:
                warnings.append(
                    f"Only {visits_remaining} authorized visits remaining"
                )

        return EligibilityResponse(
            eligible=data.get("eligible", False),
            plan_name=data.get("plan_name", ""),
            member_id=request.member_id,
            coverage_start=data.get("coverage_start"),
            coverage_end=data.get("coverage_end"),
            copay=benefits.get("copay"),
            coinsurance_pct=benefits.get("coinsurance_pct"),
            deductible=benefits.get("deductible"),
            deductible_met=benefits.get("deductible_met"),
            visits_authorized=visits_auth,
            visits_used=visits_used,
            visits_remaining=visits_remaining,
            prior_auth_required=prior_auth,
            authorization_number=benefits.get("authorization_number"),
            raw_response=data,
            warnings=warnings,
        )

    def _stub_response(self, request: EligibilityRequest) -> EligibilityResponse:
        """Return a stub response when clearinghouse is not configured."""
        return EligibilityResponse(
            eligible=False,
            plan_name="UNKNOWN",
            member_id=request.member_id,
            warnings=[
                "Clearinghouse not configured — eligibility could not be verified. "
                "Configure CLEARINGHOUSE_URL and CLEARINGHOUSE_API_KEY to enable "
                "real-time eligibility checks."
            ],
            is_stub=True,
        )

    def _error_response(
        self, request: EligibilityRequest, error_msg: str
    ) -> EligibilityResponse:
        """Return an error response preserving available information."""
        return EligibilityResponse(
            eligible=False,
            plan_name="ERROR",
            member_id=request.member_id,
            warnings=[error_msg],
            is_stub=True,
        )
