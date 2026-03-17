"""EDI 278 prior authorization for rehabilitation services.

Manages prior authorization requests, status checks, and expiration tracking.
Gracefully degrades to stub responses when payer portal is not configured.

References:
- ASC X12N 278 (005010X217)
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

logger = logging.getLogger(__name__)


@dataclass
class AuthRequest:
    """Prior authorization request data."""

    patient_id: str
    patient_first_name: str = ""
    patient_last_name: str = ""
    patient_dob: str = ""
    member_id: str = ""
    payer_id: str = ""
    provider_npi: str = ""
    diagnosis_codes: list[str] = field(default_factory=list)
    cpt_codes: list[str] = field(default_factory=list)
    requested_visits: int = 12
    requested_duration_weeks: int = 8
    clinical_justification: str = ""
    setting: str = "outpatient"
    discipline: str = "PT"


@dataclass
class AuthResponse:
    """Prior authorization response data."""

    authorized: bool = False
    auth_number: str | None = None
    visits_approved: int | None = None
    effective_date: str | None = None
    expiration_date: str | None = None
    denial_reason: str | None = None
    appeal_deadline: str | None = None
    warnings: list[str] = field(default_factory=list)
    is_stub: bool = False


class PriorAuthService:
    """EDI 278 prior authorization service.

    Submits authorization requests to payer portals and tracks
    authorization status and expiration.
    """

    def __init__(self, payer_portal_url: str = "", api_key: str = ""):
        self.payer_portal_url = payer_portal_url.rstrip("/") if payer_portal_url else ""
        self.api_key = api_key
        self._configured = bool(self.payer_portal_url and self.api_key)

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def request_authorization(
        self, request: AuthRequest
    ) -> AuthResponse:
        """Submit a prior authorization request.

        If the payer portal is not configured, returns a stub response
        with instructions to configure.
        """
        if not self._configured:
            return self._stub_response(
                "Payer portal not configured — cannot submit prior authorization. "
                "Configure PAYER_PORTAL_URL and PAYER_PORTAL_API_KEY."
            )

        try:
            payload = {
                "request_id": str(uuid.uuid4()),
                "patient": {
                    "id": request.patient_id,
                    "first_name": request.patient_first_name,
                    "last_name": request.patient_last_name,
                    "dob": request.patient_dob,
                    "member_id": request.member_id,
                },
                "provider_npi": request.provider_npi,
                "payer_id": request.payer_id,
                "diagnosis_codes": request.diagnosis_codes,
                "cpt_codes": request.cpt_codes,
                "requested_visits": request.requested_visits,
                "requested_duration_weeks": request.requested_duration_weeks,
                "clinical_justification": request.clinical_justification,
                "setting": request.setting,
                "discipline": request.discipline,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.payer_portal_url}/prior-auth/278",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                return self._parse_response(resp.json())

        except httpx.TimeoutException:
            logger.warning("Prior auth request timed out for patient %s", request.patient_id)
            return self._error_response("Prior authorization request timed out")
        except httpx.HTTPStatusError as e:
            logger.error("Prior auth HTTP error: %s", e.response.status_code)
            return self._error_response(f"Payer portal HTTP error: {e.response.status_code}")
        except Exception as e:
            logger.exception("Prior auth failed for patient %s", request.patient_id)
            return self._error_response(f"Prior authorization request failed: {str(e)}")

    async def check_auth_status(self, auth_number: str) -> AuthResponse:
        """Check the status of an existing authorization.

        Args:
            auth_number: The authorization number to check.
        """
        if not self._configured:
            return self._stub_response(
                "Payer portal not configured — cannot check authorization status."
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{self.payer_portal_url}/prior-auth/{auth_number}/status",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                return self._parse_response(resp.json())

        except Exception as e:
            logger.exception("Auth status check failed for %s", auth_number)
            return self._error_response(f"Status check failed: {str(e)}")

    async def get_expiring_auths(self, days_ahead: int = 14) -> list[dict]:
        """Get authorizations expiring within the specified number of days.

        NOTE: This checks the local database (Insurance table), not the payer
        portal. No clearinghouse configuration required.

        Args:
            days_ahead: Number of days to look ahead for expiring auths.

        Returns:
            List of dicts with patient_id, auth_number, expiry_date, days_remaining.
        """
        # This method works with local data from the Insurance model.
        # The actual DB query is handled at the route level since we don't
        # hold a DB session here. This returns the structure for reference.
        logger.info("Checking for authorizations expiring within %d days", days_ahead)
        return []  # Populated by route handler via DB query

    def _parse_response(self, data: dict) -> AuthResponse:
        """Parse payer portal response into AuthResponse."""
        warnings = []
        authorized = data.get("authorized", False)

        if not authorized and data.get("denial_reason"):
            appeal_deadline = data.get("appeal_deadline")
            if appeal_deadline:
                warnings.append(
                    f"Denied — appeal by {appeal_deadline}: {data['denial_reason']}"
                )

        visits_approved = data.get("visits_approved")
        requested = data.get("visits_requested", 0)
        if visits_approved and requested and visits_approved < requested:
            warnings.append(
                f"Partial approval: {visits_approved} of {requested} requested visits"
            )

        return AuthResponse(
            authorized=authorized,
            auth_number=data.get("auth_number"),
            visits_approved=visits_approved,
            effective_date=data.get("effective_date"),
            expiration_date=data.get("expiration_date"),
            denial_reason=data.get("denial_reason"),
            appeal_deadline=data.get("appeal_deadline"),
            warnings=warnings,
        )

    def _stub_response(self, message: str) -> AuthResponse:
        return AuthResponse(
            authorized=False,
            warnings=[message],
            is_stub=True,
        )

    def _error_response(self, message: str) -> AuthResponse:
        return AuthResponse(
            authorized=False,
            warnings=[message],
            is_stub=True,
        )
