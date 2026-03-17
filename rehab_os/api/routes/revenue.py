"""Revenue cycle management API routes.

Provides endpoints for claim generation, eligibility verification,
prior authorization, denial management, and revenue cycle KPIs.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rehab_os.api.dependencies import get_current_user
from rehab_os.core.database import get_db
from rehab_os.core.models import Insurance, Provider

router = APIRouter(
    prefix="/revenue",
    tags=["revenue-cycle"],
    dependencies=[Depends(get_current_user)],
)

logger = logging.getLogger(__name__)


# ── Request / Response schemas ───────────────────────────────────────────────


class ClaimRequest(BaseModel):
    patient_id: str
    encounter_id: str | None = None
    provider_npi: str
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
    payer_id: str
    payer_name: str = ""
    member_id: str
    patient_first_name: str
    patient_last_name: str
    patient_dob: str = ""
    patient_gender: str = ""
    patient_address: str = ""
    patient_city: str = ""
    patient_state: str = ""
    patient_zip: str = ""
    diagnosis_codes: list[str]
    lines: list[dict]  # Each: {cpt_code, modifier, units, charge_amount, diagnosis_pointers, service_date, place_of_service}
    authorization_number: str | None = None
    setting: str = "outpatient"
    discipline: str = "PT"


class EligibilityCheckRequest(BaseModel):
    patient_first_name: str
    patient_last_name: str
    patient_dob: str
    member_id: str
    payer_id: str
    provider_npi: str
    service_type: str = "PT"


class PriorAuthRequest(BaseModel):
    patient_id: str
    patient_first_name: str = ""
    patient_last_name: str = ""
    patient_dob: str = ""
    member_id: str = ""
    payer_id: str
    provider_npi: str
    diagnosis_codes: list[str]
    cpt_codes: list[str]
    requested_visits: int = 12
    requested_duration_weeks: int = 8
    clinical_justification: str = ""
    setting: str = "outpatient"
    discipline: str = "PT"


# ── Service singletons (lazy init) ──────────────────────────────────────────


def _get_services(request: Request) -> dict[str, Any]:
    """Lazy-initialize revenue cycle services from app state."""
    if not hasattr(request.app.state, "_revenue_services"):
        from rehab_os.config import get_settings
        from rehab_os.revenue_cycle.claims import ClaimGenerator
        from rehab_os.revenue_cycle.eligibility import EligibilityService
        from rehab_os.revenue_cycle.prior_auth import PriorAuthService
        from rehab_os.revenue_cycle.claim_status import ClaimStatusService
        from rehab_os.revenue_cycle.denial_manager import DenialManager

        settings = get_settings()
        request.app.state._revenue_services = {
            "claim_generator": ClaimGenerator(
                sender_id=settings.clearinghouse_sender_id,
                receiver_id=settings.clearinghouse_receiver_id,
            ),
            "eligibility": EligibilityService(
                clearinghouse_url=settings.clearinghouse_url,
                api_key=settings.clearinghouse_api_key,
            ),
            "prior_auth": PriorAuthService(
                payer_portal_url=settings.payer_portal_url,
                api_key=settings.payer_portal_api_key,
            ),
            "claim_status": ClaimStatusService(
                clearinghouse_url=settings.clearinghouse_url,
                api_key=settings.clearinghouse_api_key,
            ),
            "denial_manager": DenialManager(),
        }
    return request.app.state._revenue_services


# ── Claims ───────────────────────────────────────────────────────────────────


@router.post("/claims")
async def generate_claim(body: ClaimRequest, request: Request):
    """Generate and validate an 837P professional claim."""
    from rehab_os.revenue_cycle.claims import Claim, ClaimLine

    services = _get_services(request)
    generator = services["claim_generator"]

    lines = []
    for ln in body.lines:
        lines.append(ClaimLine(
            cpt_code=ln["cpt_code"],
            modifier=ln.get("modifier", ""),
            units=ln.get("units", 1),
            charge_amount=ln.get("charge_amount", 0.0),
            diagnosis_pointers=ln.get("diagnosis_pointers", [1]),
            service_date=ln.get("service_date", ""),
            place_of_service=ln.get("place_of_service", ""),
        ))

    claim = Claim(
        patient_id=body.patient_id,
        patient_first_name=body.patient_first_name,
        patient_last_name=body.patient_last_name,
        patient_dob=body.patient_dob,
        patient_gender=body.patient_gender,
        patient_address=body.patient_address,
        patient_city=body.patient_city,
        patient_state=body.patient_state,
        patient_zip=body.patient_zip,
        member_id=body.member_id,
        provider_npi=body.provider_npi,
        provider_first_name=body.provider_first_name,
        provider_last_name=body.provider_last_name,
        provider_tax_id=body.provider_tax_id,
        facility_npi=body.facility_npi,
        facility_name=body.facility_name,
        facility_address=body.facility_address,
        facility_city=body.facility_city,
        facility_state=body.facility_state,
        facility_zip=body.facility_zip,
        referring_npi=body.referring_npi,
        referring_first_name=body.referring_first_name,
        referring_last_name=body.referring_last_name,
        payer_id=body.payer_id,
        payer_name=body.payer_name,
        diagnosis_codes=body.diagnosis_codes,
        lines=lines,
        authorization_number=body.authorization_number,
        setting=body.setting,
        discipline=body.discipline,
    )

    # Validate first
    errors = generator.validate_claim(claim)
    if errors:
        return {
            "status": "validation_failed",
            "errors": errors,
            "claim_id": claim.claim_id,
        }

    # Generate 837P
    txn = generator.generate_837p(claim)
    validation_errors = txn.validate()

    # Also generate CMS-1500 data
    cms1500 = generator.generate_cms1500_data(claim)

    return {
        "status": "generated",
        "claim_id": claim.claim_id,
        "total_charge": claim.total_charge,
        "line_count": len(claim.lines),
        "edi_837p": txn.render(),
        "edi_validation_errors": validation_errors,
        "cms1500_data": cms1500,
    }


# ── Eligibility ──────────────────────────────────────────────────────────────


@router.post("/eligibility")
async def check_eligibility(body: EligibilityCheckRequest, request: Request):
    """Check patient insurance eligibility via EDI 270/271."""
    from rehab_os.revenue_cycle.eligibility import EligibilityRequest

    services = _get_services(request)
    svc = services["eligibility"]

    req = EligibilityRequest(
        patient_first_name=body.patient_first_name,
        patient_last_name=body.patient_last_name,
        patient_dob=body.patient_dob,
        member_id=body.member_id,
        payer_id=body.payer_id,
        provider_npi=body.provider_npi,
        service_type=body.service_type,
    )

    result = await svc.check_eligibility(req)
    return {
        "eligible": result.eligible,
        "plan_name": result.plan_name,
        "member_id": result.member_id,
        "coverage_start": result.coverage_start,
        "coverage_end": result.coverage_end,
        "copay": result.copay,
        "coinsurance_pct": result.coinsurance_pct,
        "deductible": result.deductible,
        "deductible_met": result.deductible_met,
        "visits_authorized": result.visits_authorized,
        "visits_used": result.visits_used,
        "visits_remaining": result.visits_remaining,
        "prior_auth_required": result.prior_auth_required,
        "authorization_number": result.authorization_number,
        "warnings": result.warnings,
        "is_stub": result.is_stub,
    }


# ── Prior Authorization ──────────────────────────────────────────────────────


@router.post("/prior-auth")
async def request_prior_auth(body: PriorAuthRequest, request: Request):
    """Request prior authorization via EDI 278."""
    from rehab_os.revenue_cycle.prior_auth import AuthRequest

    services = _get_services(request)
    svc = services["prior_auth"]

    req = AuthRequest(
        patient_id=body.patient_id,
        patient_first_name=body.patient_first_name,
        patient_last_name=body.patient_last_name,
        patient_dob=body.patient_dob,
        member_id=body.member_id,
        payer_id=body.payer_id,
        provider_npi=body.provider_npi,
        diagnosis_codes=body.diagnosis_codes,
        cpt_codes=body.cpt_codes,
        requested_visits=body.requested_visits,
        requested_duration_weeks=body.requested_duration_weeks,
        clinical_justification=body.clinical_justification,
        setting=body.setting,
        discipline=body.discipline,
    )

    result = await svc.request_authorization(req)
    return {
        "authorized": result.authorized,
        "auth_number": result.auth_number,
        "visits_approved": result.visits_approved,
        "effective_date": result.effective_date,
        "expiration_date": result.expiration_date,
        "denial_reason": result.denial_reason,
        "appeal_deadline": result.appeal_deadline,
        "warnings": result.warnings,
        "is_stub": result.is_stub,
    }


@router.get("/auth-expiring")
async def get_expiring_auths(
    days: int = Query(14, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """Get authorizations expiring within N days from local database."""
    cutoff = date.today() + timedelta(days=days)
    stmt = (
        select(Insurance)
        .where(Insurance.expiry_date is not None)
        .where(Insurance.expiry_date <= cutoff)
        .where(Insurance.expiry_date >= date.today())
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    return [
        {
            "patient_id": str(r.patient_id),
            "payer_name": r.payer_name,
            "auth_number": r.auth_number,
            "expiry_date": r.expiry_date.isoformat() if r.expiry_date else None,
            "authorized_visits": r.authorized_visits,
            "visits_used": r.visits_used,
            "visits_remaining": (r.authorized_visits or 0) - (r.visits_used or 0),
            "days_remaining": (r.expiry_date - date.today()).days if r.expiry_date else None,
        }
        for r in records
    ]


# ── Denials ──────────────────────────────────────────────────────────────────


@router.get("/denials")
async def get_open_denials(request: Request):
    """Get all open (unresolved) denials."""
    services = _get_services(request)
    mgr = services["denial_manager"]
    denials = mgr.get_open_denials()
    return [
        {
            "denial_id": d.denial_id,
            "claim_id": d.claim_id,
            "patient_id": d.patient_id,
            "patient_name": d.patient_name,
            "payer_name": d.payer_name,
            "denial_date": d.denial_date,
            "denial_code": d.denial_code,
            "denial_reason": d.denial_reason,
            "billed_amount": d.billed_amount,
            "appeal_deadline": d.appeal_deadline,
            "days_until_deadline": d.days_until_appeal_deadline,
            "status": d.status,
            "is_appealable": d.is_appealable,
        }
        for d in denials
    ]


@router.post("/denials/{denial_id}/appeal")
async def generate_appeal(denial_id: str, request: Request):
    """Generate an LLM-powered appeal letter for a denial."""
    services = _get_services(request)
    mgr = services["denial_manager"]

    llm_router = getattr(request.app.state, "llm_router", None)
    if not llm_router:
        raise HTTPException(status_code=503, detail="LLM router not available")

    try:
        letter = await mgr.generate_appeal_letter(denial_id, llm_router)
        return {"denial_id": denial_id, "appeal_letter": letter, "status": "generated"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Claim Status ─────────────────────────────────────────────────────────────


@router.get("/claim-status/{claim_id}")
async def check_claim_status(
    claim_id: str,
    payer_id: str = Query(...),
    request: Request = None,
):
    """Check claim adjudication status via EDI 276/277."""
    services = _get_services(request)
    svc = services["claim_status"]

    result = await svc.check_status(claim_id, payer_id)
    return {
        "claim_id": result.claim_id,
        "status_code": result.status_code,
        "status_description": result.status_description,
        "status_date": result.status_date,
        "payer_claim_id": result.payer_claim_id,
        "total_charge": result.total_charge,
        "paid_amount": result.paid_amount,
        "check_number": result.check_number,
        "payment_date": result.payment_date,
        "is_finalized": result.is_finalized,
        "is_denied": result.is_denied,
        "is_pending": result.is_pending,
        "warnings": result.warnings,
        "is_stub": result.is_stub,
    }


# ── Dashboard KPIs ───────────────────────────────────────────────────────────


@router.get("/dashboard")
async def revenue_dashboard(request: Request):
    """Revenue cycle KPIs and summary metrics."""
    services = _get_services(request)
    mgr = services["denial_manager"]
    eligibility_svc = services["eligibility"]
    prior_auth_svc = services["prior_auth"]

    denial_stats = mgr.get_denial_stats()
    expiring = mgr.get_expiring_appeals(days=30)

    return {
        "denial_stats": denial_stats,
        "expiring_appeals": [
            {
                "denial_id": d.denial_id,
                "claim_id": d.claim_id,
                "patient_name": d.patient_name,
                "billed_amount": d.billed_amount,
                "appeal_deadline": d.appeal_deadline,
                "days_remaining": d.days_until_appeal_deadline,
            }
            for d in expiring
        ],
        "services_configured": {
            "eligibility": eligibility_svc.is_configured,
            "prior_auth": prior_auth_svc.is_configured,
            "claim_status": services["claim_status"].is_configured,
        },
    }
