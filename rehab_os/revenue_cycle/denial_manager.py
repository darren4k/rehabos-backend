"""Denial tracking, analytics, and appeal generation.

Manages the lifecycle of claim denials from initial recording through
appeal resolution, with LLM-powered appeal letter generation.

References:
- CARC/RARC code sets
- CMS Medicare Appeals Process (5 levels)
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from rehab_os.revenue_cycle.remittance import CARC_DESCRIPTIONS, PaymentLine

logger = logging.getLogger(__name__)

# Standard appeal deadlines by payer type (days from denial)
APPEAL_DEADLINES = {
    "Medicare": 120,
    "Medicaid": 60,
    "Commercial": 180,
    "HMO": 60,
    "default": 90,
}

DENIAL_STATUSES = ("new", "reviewing", "appealing", "won", "lost", "written_off")


@dataclass
class Denial:
    """A tracked claim denial."""

    denial_id: str = ""
    claim_id: str = ""
    patient_id: str = ""
    patient_name: str = ""
    payer_name: str = ""
    denial_date: str = ""  # CCYYMMDD
    denial_code: str = ""  # CARC code
    denial_reason: str = ""
    billed_amount: float = 0.0
    appeal_deadline: str | None = None
    status: str = "new"
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.denial_id:
            self.denial_id = f"DEN-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(timezone.utc).strftime("%Y%m%d")
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        if not self.denial_reason and self.denial_code:
            self.denial_reason = CARC_DESCRIPTIONS.get(
                self.denial_code, f"Denial code {self.denial_code}"
            )

    @property
    def is_appealable(self) -> bool:
        """Check if the denial can still be appealed."""
        if self.status in ("won", "lost", "written_off"):
            return False
        if not self.appeal_deadline:
            return True
        try:
            deadline = datetime.strptime(self.appeal_deadline, "%Y%m%d")
            return datetime.now(timezone.utc).replace(tzinfo=None) < deadline
        except ValueError:
            return True

    @property
    def days_until_appeal_deadline(self) -> int | None:
        """Days remaining until appeal deadline."""
        if not self.appeal_deadline:
            return None
        try:
            deadline = datetime.strptime(self.appeal_deadline, "%Y%m%d")
            delta = deadline - datetime.now(timezone.utc).replace(tzinfo=None)
            return max(0, delta.days)
        except ValueError:
            return None


class DenialManager:
    """Manages denial tracking, analytics, and appeal workflows.

    Maintains an in-memory denial store (production would use DB).
    The appeal letter generation uses the LLM router from app state.
    """

    def __init__(self):
        self._denials: dict[str, Denial] = {}

    def record_denial(
        self,
        payment_line: PaymentLine,
        claim_id: str = "",
        patient_id: str = "",
        patient_name: str = "",
        payer_name: str = "",
        payer_type: str = "default",
    ) -> Denial:
        """Record a new denial from a remittance payment line.

        Args:
            payment_line: The denied PaymentLine from ERA parsing.
            claim_id: Associated claim ID.
            patient_id: Patient identifier.
            patient_name: Patient display name.
            payer_name: Payer display name.
            payer_type: Payer category for appeal deadline calculation.

        Returns:
            The created Denial record.
        """
        # Calculate appeal deadline
        deadline_days = APPEAL_DEADLINES.get(payer_type, APPEAL_DEADLINES["default"])
        appeal_deadline = (
            datetime.now(timezone.utc) + timedelta(days=deadline_days)
        ).strftime("%Y%m%d")

        denial = Denial(
            claim_id=claim_id or payment_line.claim_id,
            patient_id=patient_id,
            patient_name=patient_name,
            payer_name=payer_name,
            denial_date=datetime.now(timezone.utc).strftime("%Y%m%d"),
            denial_code=payment_line.denial_code or "",
            denial_reason=payment_line.adjustment_description,
            billed_amount=payment_line.billed_amount,
            appeal_deadline=appeal_deadline,
        )

        self._denials[denial.denial_id] = denial
        logger.info(
            "Recorded denial %s for claim %s: %s ($%.2f)",
            denial.denial_id, denial.claim_id,
            denial.denial_reason, denial.billed_amount,
        )
        return denial

    def get_denial(self, denial_id: str) -> Denial | None:
        """Get a single denial by ID."""
        return self._denials.get(denial_id)

    def update_status(self, denial_id: str, status: str, notes: str = "") -> Denial | None:
        """Update denial status."""
        denial = self._denials.get(denial_id)
        if not denial:
            return None
        if status not in DENIAL_STATUSES:
            raise ValueError(f"Invalid status: {status}. Must be one of {DENIAL_STATUSES}")
        denial.status = status
        denial.updated_at = datetime.now(timezone.utc).strftime("%Y%m%d")
        if notes:
            denial.notes = notes
        return denial

    def get_open_denials(self) -> list[Denial]:
        """Get all denials that are not resolved (new, reviewing, appealing)."""
        return [
            d for d in self._denials.values()
            if d.status in ("new", "reviewing", "appealing")
        ]

    def get_expiring_appeals(self, days: int = 30) -> list[Denial]:
        """Get denials with appeal deadlines expiring within N days."""
        results = []
        for denial in self._denials.values():
            remaining = denial.days_until_appeal_deadline
            if remaining is not None and remaining <= days and denial.is_appealable:
                results.append(denial)
        return sorted(results, key=lambda d: d.appeal_deadline or "")

    def get_denial_stats(self) -> dict[str, Any]:
        """Compute denial analytics for dashboard display."""
        all_denials = list(self._denials.values())
        if not all_denials:
            return {
                "total_denials": 0,
                "total_amount": 0.0,
                "open_count": 0,
                "won_count": 0,
                "lost_count": 0,
                "written_off_count": 0,
                "by_code": {},
                "by_payer": {},
                "appeal_success_rate": 0.0,
            }

        by_code: dict[str, int] = {}
        by_payer: dict[str, int] = {}
        status_counts = {s: 0 for s in DENIAL_STATUSES}

        for d in all_denials:
            code_key = d.denial_code or "unknown"
            by_code[code_key] = by_code.get(code_key, 0) + 1
            payer_key = d.payer_name or "unknown"
            by_payer[payer_key] = by_payer.get(payer_key, 0) + 1
            status_counts[d.status] = status_counts.get(d.status, 0) + 1

        appealed = status_counts.get("won", 0) + status_counts.get("lost", 0)
        success_rate = (
            (status_counts["won"] / appealed * 100) if appealed > 0 else 0.0
        )

        return {
            "total_denials": len(all_denials),
            "total_amount": round(sum(d.billed_amount for d in all_denials), 2),
            "open_count": status_counts.get("new", 0) + status_counts.get("reviewing", 0) + status_counts.get("appealing", 0),
            "won_count": status_counts.get("won", 0),
            "lost_count": status_counts.get("lost", 0),
            "written_off_count": status_counts.get("written_off", 0),
            "by_code": dict(sorted(by_code.items(), key=lambda x: x[1], reverse=True)),
            "by_payer": dict(sorted(by_payer.items(), key=lambda x: x[1], reverse=True)),
            "appeal_success_rate": round(success_rate, 1),
        }

    async def generate_appeal_letter(
        self, denial_id: str, llm_router: Any
    ) -> str:
        """Generate an LLM-powered appeal letter for a denial.

        Args:
            denial_id: The denial to appeal.
            llm_router: LLMRouter instance from app.state.llm_router.

        Returns:
            Generated appeal letter text.

        Raises:
            ValueError: If denial not found or not appealable.
        """
        denial = self._denials.get(denial_id)
        if not denial:
            raise ValueError(f"Denial {denial_id} not found")
        if not denial.is_appealable:
            raise ValueError(
                f"Denial {denial_id} is not appealable (status: {denial.status})"
            )

        prompt = f"""Generate a professional medical appeal letter for a physical therapy claim denial.

Denial Details:
- Claim ID: {denial.claim_id}
- Patient: {denial.patient_name}
- Payer: {denial.payer_name}
- Denial Code: {denial.denial_code}
- Denial Reason: {denial.denial_reason}
- Billed Amount: ${denial.billed_amount:.2f}
- Appeal Deadline: {denial.appeal_deadline}

Write a formal appeal letter that:
1. References the specific claim and denial
2. Addresses the denial reason with clinical justification
3. Cites relevant medical necessity criteria
4. Requests reconsideration of the denial
5. Is professional, concise, and follows standard appeal letter format

Do NOT include placeholder brackets — use the actual values provided."""

        from rehab_os.llm.base import Message

        messages = [
            Message(role="system", content="You are a healthcare billing specialist experienced in writing successful appeal letters for rehabilitation therapy claims."),
            Message(role="user", content=prompt),
        ]

        try:
            response = await llm_router.complete(
                messages, temperature=0.3, max_tokens=2000
            )
            letter = response.content

            # Update denial status
            denial.status = "appealing"
            denial.updated_at = datetime.now(timezone.utc).strftime("%Y%m%d")

            return letter
        except Exception as e:
            logger.exception("Failed to generate appeal letter for %s", denial_id)
            raise ValueError(f"Appeal letter generation failed: {str(e)}") from e
