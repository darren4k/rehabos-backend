"""Revenue Cycle Management module for RehabOS.

Provides end-to-end revenue cycle functionality:
- X12 EDI formatting (837P, 270/271, 276/277, 278, 835)
- Professional claim generation (CMS-1500 / 837P)
- Eligibility verification (270/271)
- Prior authorization management (278)
- Remittance advice parsing (835)
- Denial tracking and LLM-powered appeal generation
- Claim status inquiry (276/277)
"""

from rehab_os.revenue_cycle.claim_status import ClaimStatusResult, ClaimStatusService
from rehab_os.revenue_cycle.claims import Claim, ClaimGenerator, ClaimLine
from rehab_os.revenue_cycle.denial_manager import Denial, DenialManager
from rehab_os.revenue_cycle.edi_formatter import X12Segment, X12Transaction
from rehab_os.revenue_cycle.eligibility import (
    EligibilityRequest,
    EligibilityResponse,
    EligibilityService,
)
from rehab_os.revenue_cycle.prior_auth import AuthRequest, AuthResponse, PriorAuthService
from rehab_os.revenue_cycle.remittance import PaymentLine, RemittanceAdvice, RemittanceParser

__all__ = [
    # EDI
    "X12Segment",
    "X12Transaction",
    # Claims
    "Claim",
    "ClaimLine",
    "ClaimGenerator",
    # Eligibility
    "EligibilityRequest",
    "EligibilityResponse",
    "EligibilityService",
    # Prior Auth
    "AuthRequest",
    "AuthResponse",
    "PriorAuthService",
    # Remittance
    "PaymentLine",
    "RemittanceAdvice",
    "RemittanceParser",
    # Denials
    "Denial",
    "DenialManager",
    # Claim Status
    "ClaimStatusResult",
    "ClaimStatusService",
]
