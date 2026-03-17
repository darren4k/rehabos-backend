"""EDI 276/277 claim status inquiry service.

Checks claim adjudication status with payers via clearinghouse.
Gracefully degrades to stub responses when not configured.

References:
- ASC X12N 276/277 (005010X212)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# Claim status category codes (277)
STATUS_CATEGORIES = {
    "A0": "Forwarded — The claim has been forwarded to another entity",
    "A1": "Pending — The claim is pending payer review",
    "A2": "Pending — The claim is pending additional information",
    "A3": "Pending — In adjudication",
    "A4": "Pending — Returned for correction",
    "F0": "Finalized — Payment is pending",
    "F1": "Finalized — Payment has been made",
    "F2": "Finalized — Denied",
    "F3": "Finalized — Revised",
    "F4": "Finalized — Reversed",
    "R0": "Rejected — General",
    "R1": "Rejected — Missing or invalid information",
    "R3": "Rejected — Not covered",
}


@dataclass
class ClaimStatusResult:
    """Result of a claim status inquiry."""

    claim_id: str = ""
    status_code: str = ""
    status_description: str = ""
    status_date: str = ""
    payer_claim_id: str = ""
    total_charge: float = 0.0
    paid_amount: float = 0.0
    check_number: str | None = None
    payment_date: str | None = None
    additional_info: str = ""
    warnings: list[str] = field(default_factory=list)
    is_stub: bool = False

    @property
    def is_finalized(self) -> bool:
        return self.status_code.startswith("F")

    @property
    def is_denied(self) -> bool:
        return self.status_code == "F2"

    @property
    def is_pending(self) -> bool:
        return self.status_code.startswith("A")


class ClaimStatusService:
    """EDI 276/277 claim status inquiry service."""

    def __init__(self, clearinghouse_url: str = "", api_key: str = ""):
        self.clearinghouse_url = clearinghouse_url.rstrip("/") if clearinghouse_url else ""
        self.api_key = api_key
        self._configured = bool(self.clearinghouse_url and self.api_key)

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def check_status(
        self, claim_id: str, payer_id: str
    ) -> ClaimStatusResult:
        """Check the adjudication status of a single claim.

        Args:
            claim_id: The claim identifier.
            payer_id: The payer identifier.

        Returns:
            ClaimStatusResult with current status.
        """
        if not self._configured:
            return ClaimStatusResult(
                claim_id=claim_id,
                status_code="",
                status_description="Clearinghouse not configured",
                warnings=[
                    "Clearinghouse not configured — cannot check claim status. "
                    "Configure CLEARINGHOUSE_URL and CLEARINGHOUSE_API_KEY."
                ],
                is_stub=True,
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.clearinghouse_url}/claim-status/276",
                    json={"claim_id": claim_id, "payer_id": payer_id},
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                return self._parse_response(claim_id, resp.json())

        except httpx.TimeoutException:
            logger.warning("Claim status check timed out for %s", claim_id)
            return self._error_result(claim_id, "Claim status check timed out")
        except httpx.HTTPStatusError as e:
            logger.error("Claim status HTTP error: %s", e.response.status_code)
            return self._error_result(claim_id, f"HTTP error: {e.response.status_code}")
        except Exception as e:
            logger.exception("Claim status check failed for %s", claim_id)
            return self._error_result(claim_id, f"Status check failed: {str(e)}")

    async def batch_check(
        self, claim_ids: list[tuple[str, str]]
    ) -> list[ClaimStatusResult]:
        """Check status of multiple claims.

        Args:
            claim_ids: List of (claim_id, payer_id) tuples.

        Returns:
            List of ClaimStatusResult in same order as input.
        """
        results = []
        for claim_id, payer_id in claim_ids:
            result = await self.check_status(claim_id, payer_id)
            results.append(result)
        return results

    def _parse_response(self, claim_id: str, data: dict) -> ClaimStatusResult:
        """Parse clearinghouse response into ClaimStatusResult."""
        status_code = data.get("status_code", "")
        status_desc = STATUS_CATEGORIES.get(
            status_code, data.get("status_description", f"Status: {status_code}")
        )

        warnings = []
        if status_code == "F2":
            warnings.append("Claim has been denied — review remittance for denial reason")
        elif status_code == "A2":
            warnings.append("Payer is requesting additional information")
        elif status_code.startswith("R"):
            warnings.append("Claim was rejected — correct and resubmit")

        return ClaimStatusResult(
            claim_id=claim_id,
            status_code=status_code,
            status_description=status_desc,
            status_date=data.get("status_date", ""),
            payer_claim_id=data.get("payer_claim_id", ""),
            total_charge=data.get("total_charge", 0.0),
            paid_amount=data.get("paid_amount", 0.0),
            check_number=data.get("check_number"),
            payment_date=data.get("payment_date"),
            additional_info=data.get("additional_info", ""),
            warnings=warnings,
        )

    def _error_result(self, claim_id: str, message: str) -> ClaimStatusResult:
        return ClaimStatusResult(
            claim_id=claim_id,
            status_code="",
            status_description="Error",
            warnings=[message],
            is_stub=True,
        )
