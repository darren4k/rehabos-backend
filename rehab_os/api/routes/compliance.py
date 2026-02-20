"""HIPAA Compliance and Data Handling API.

RehabOS is designed for HIPAA compliance:
- NO persistent storage of PHI on server
- All patient data is processed in-memory only
- Audit logging for all PHI access
- Client-side data storage responsibility
"""

from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, Request, Header
from pydantic import BaseModel, Field

from rehab_os.api.dependencies import get_current_user
from rehab_os.core.models import Provider
import hashlib
import json
import logging

router = APIRouter(prefix="/compliance", tags=["compliance"])

# Audit logger for HIPAA compliance
audit_logger = logging.getLogger("rehab_os.audit")


class AuditEntry(BaseModel):
    """HIPAA audit log entry."""
    timestamp: str
    event_type: str
    user_id: Optional[str]
    client_ip: str
    action: str
    resource: str
    data_accessed: list[str]  # Types of data accessed, not the data itself
    success: bool
    session_hash: Optional[str] = None  # Hashed session identifier


class DataHandlingPolicy(BaseModel):
    """RehabOS data handling policy."""
    phi_storage: str = "none"
    phi_retention: str = "session_only"
    encryption_in_transit: str = "TLS 1.2+"
    encryption_at_rest: str = "not_applicable"
    audit_logging: str = "enabled"
    data_location: str = "client_side_only"


class ComplianceInfo(BaseModel):
    """HIPAA compliance information."""
    hipaa_compliant: bool = True
    data_handling: DataHandlingPolicy
    recommendations: list[str]
    client_responsibilities: list[str]
    server_guarantees: list[str]


def log_phi_access(
    request: Request,
    action: str,
    resource: str,
    data_types: list[str],
    user_id: Optional[str] = None,
    success: bool = True,
):
    """Log PHI access for HIPAA audit trail."""
    client_ip = request.client.host if request.client else "unknown"

    entry = AuditEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        event_type="PHI_ACCESS",
        user_id=user_id,
        client_ip=client_ip,
        action=action,
        resource=resource,
        data_accessed=data_types,
        success=success,
    )

    # Log to audit trail (in production, this would go to secure audit storage)
    audit_logger.info(json.dumps(entry.model_dump()))

    return entry


@router.get("/info", response_model=ComplianceInfo)
async def get_compliance_info(current_user: Provider = Depends(get_current_user)):
    """Get HIPAA compliance information for RehabOS.

    RehabOS is designed for HIPAA compliance:
    - NO patient data is stored on the server
    - All processing is done in-memory only
    - Data is never written to disk
    - All communication must be over HTTPS

    Organizations using RehabOS are responsible for:
    - Securing their own client-side data storage
    - Implementing appropriate access controls
    - Signing a BAA if required
    """
    return ComplianceInfo(
        hipaa_compliant=True,
        data_handling=DataHandlingPolicy(),
        recommendations=[
            "Always use HTTPS for API communication",
            "Implement client-side encryption for stored patient data",
            "Use your EMR's existing HIPAA-compliant storage",
            "Implement role-based access control on client side",
            "Maintain your own audit logs of data access",
            "Train users on PHI handling procedures",
        ],
        client_responsibilities=[
            "Secure storage of patient data (RehabOS does not store PHI)",
            "Access control and user authentication",
            "Encryption of data at rest on client systems",
            "Audit logging of user access to patient data",
            "Business Associate Agreement if required",
            "Workforce training on HIPAA compliance",
        ],
        server_guarantees=[
            "No persistent storage of PHI",
            "In-memory processing only",
            "No logging of PHI content",
            "Encrypted communication (HTTPS required)",
            "Audit logging of API access (without PHI content)",
            "Stateless processing - no session data retained",
        ],
    )


@router.get("/audit-policy")
async def get_audit_policy(current_user: Provider = Depends(get_current_user)):
    """Get the audit logging policy."""
    return {
        "audit_logging_enabled": True,
        "logged_events": [
            "API requests (endpoint, method, timestamp)",
            "Authentication attempts",
            "Error events",
            "Data type accessed (not content)",
        ],
        "not_logged": [
            "Patient names or identifiers",
            "Medical record content",
            "Diagnosis details",
            "Any PHI content",
        ],
        "retention_policy": "Audit logs retained per organization policy",
        "access": "Audit logs available to system administrators only",
    }


@router.get("/data-flow")
async def get_data_flow(current_user: Provider = Depends(get_current_user)):
    """Explains how patient data flows through RehabOS."""
    return {
        "data_flow": {
            "step_1": {
                "name": "Client Request",
                "description": "Client sends patient data in API request",
                "data_location": "In transit (encrypted via HTTPS)",
            },
            "step_2": {
                "name": "Server Processing",
                "description": "Server processes request in memory",
                "data_location": "Server RAM only (never written to disk)",
            },
            "step_3": {
                "name": "Knowledge Retrieval",
                "description": "System retrieves evidence-based recommendations",
                "data_location": "Only clinical knowledge accessed, not PHI stored",
            },
            "step_4": {
                "name": "Response",
                "description": "Recommendations sent back to client",
                "data_location": "In transit (encrypted via HTTPS)",
            },
            "step_5": {
                "name": "Memory Cleared",
                "description": "Request data cleared from server memory",
                "data_location": "No data retained",
            },
        },
        "key_points": [
            "PHI never written to server disk",
            "PHI never stored in database",
            "PHI never included in logs",
            "Each request is stateless and independent",
            "Client is responsible for data storage",
        ],
    }


@router.post("/log-access")
async def log_data_access(
    request: Request,
    action: str,
    resource: str,
    data_types: list[str],
    x_user_id: Optional[str] = Header(None),
    current_user: Provider = Depends(get_current_user),
):
    """Manually log a PHI access event.

    Use this to create audit entries for compliance reporting.
    """
    entry = log_phi_access(
        request=request,
        action=action,
        resource=resource,
        data_types=data_types,
        user_id=x_user_id,
    )

    return {"status": "logged", "entry_id": hashlib.sha256(
        entry.timestamp.encode()
    ).hexdigest()[:12]}


@router.get("/encryption-requirements")
async def get_encryption_requirements(current_user: Provider = Depends(get_current_user)):
    """Get encryption requirements for HIPAA compliance."""
    return {
        "in_transit": {
            "required": True,
            "protocol": "TLS 1.2 or higher",
            "implementation": "HTTPS for all API calls",
            "certificate": "Valid SSL certificate required",
        },
        "at_rest_server": {
            "required": False,
            "reason": "No PHI stored on server",
        },
        "at_rest_client": {
            "required": True,
            "responsibility": "Client organization",
            "recommendations": [
                "Use EMR's built-in encryption",
                "AES-256 for local storage",
                "Encrypted database for patient data",
                "Full-disk encryption on workstations",
            ],
        },
    }


@router.get("/integration-guide")
async def get_integration_guide(current_user: Provider = Depends(get_current_user)):
    """Guide for HIPAA-compliant integration with EMR systems."""
    return {
        "overview": "RehabOS integrates with EMRs as a clinical decision support tool",
        "integration_patterns": {
            "pattern_1": {
                "name": "Direct API Integration",
                "description": "EMR calls RehabOS API directly",
                "data_flow": "EMR → RehabOS API → EMR",
                "phi_storage": "EMR only (RehabOS processes, does not store)",
            },
            "pattern_2": {
                "name": "Middleware Integration",
                "description": "Integration engine mediates between EMR and RehabOS",
                "data_flow": "EMR → Integration Engine → RehabOS → Integration Engine → EMR",
                "phi_storage": "EMR and Integration Engine (if required)",
            },
            "pattern_3": {
                "name": "Standalone with Manual Entry",
                "description": "Clinician manually enters data, results copied back",
                "data_flow": "Clinician → RehabOS → Clinician → EMR",
                "phi_storage": "EMR only",
            },
        },
        "baa_requirements": {
            "when_required": "When RehabOS processes PHI on behalf of covered entity",
            "covers": [
                "Data processing agreement",
                "Security requirements",
                "Breach notification procedures",
                "Permitted uses and disclosures",
            ],
        },
        "security_checklist": [
            "HTTPS enabled for all API calls",
            "API authentication implemented",
            "Audit logging configured",
            "Access controls in place",
            "Workforce training completed",
            "BAA signed if required",
            "Incident response plan in place",
        ],
    }
