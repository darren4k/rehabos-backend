"""
NPPES (National Plan and Provider Enumeration System) Integration

Provides NPI lookup and provider verification against the CMS NPPES registry.
https://npiregistry.cms.hhs.gov/api/

Ported from Magda Health's NPPES integration with PT/OT/SLP-specific
taxonomy filtering for rehabilitation provider workflows.

Supports:
- Individual NPI lookups with validation
- Provider search by name, specialty, state
- Referring provider verification
- Rehabilitation taxonomy filtering (PT, OT, SLP)
- 24-hour in-memory caching
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# NPPES API Configuration
NPPES_API_URL = "https://npiregistry.cms.hhs.gov/api/"
NPPES_API_VERSION = "2.1"
DEFAULT_TIMEOUT = 30.0
CACHE_TTL_HOURS = 24

# Rehabilitation-specific taxonomy codes
REHAB_TAXONOMY_CODES = {
    # Physical Therapy
    "225100000X": "Physical Therapist",
    "2251C2600X": "PT - Cardiopulmonary",
    "2251E1200X": "PT - Electrophysiology",
    "2251E1300X": "PT - Ergonomics",
    "2251G0304X": "PT - Geriatrics",
    "2251H1200X": "PT - Hand",
    "2251H1300X": "PT - Human Factors",
    "2251N0400X": "PT - Neurology",
    "2251P0200X": "PT - Pediatrics",
    "2251S0007X": "PT - Sports",
    "2251X0800X": "PT - Orthopedic",
    # Occupational Therapy
    "225X00000X": "Occupational Therapist",
    "225XE0001X": "OT - Enviro Modification",
    "225XE1200X": "OT - Ergonomics",
    "225XF0002X": "OT - Feeding/Swallowing",
    "225XG0600X": "OT - Gerontology",
    "225XH1200X": "OT - Hand",
    "225XH1300X": "OT - Human Factors",
    "225XL0004X": "OT - Low Vision",
    "225XM0800X": "OT - Mental Health",
    "225XN1300X": "OT - Neurorehabilitation",
    "225XP0019X": "OT - Physical Rehabilitation",
    "225XP0200X": "OT - Pediatrics",
    # Speech-Language Pathology
    "235Z00000X": "Speech-Language Pathologist",
    "2355A2700X": "SLP - Audiology",
    "2355S0801X": "SLP - Speech",
    # Physical Therapist Assistant
    "225200000X": "Physical Therapist Assistant",
    # Occupational Therapy Assistant
    "224Z00000X": "Occupational Therapy Assistant",
}


class EntityType(str, Enum):
    INDIVIDUAL = "1"
    ORGANIZATION = "2"


class ProviderStatus(str, Enum):
    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    DEACTIVATED = "deactivated"
    ERROR = "error"


@dataclass
class ProviderAddress:
    address_type: str  # location, mailing
    address_line1: str
    address_line2: Optional[str] = None
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = "US"
    phone: Optional[str] = None
    fax: Optional[str] = None


@dataclass
class ProviderTaxonomy:
    code: str
    description: str
    license: Optional[str] = None
    state: Optional[str] = None
    is_primary: bool = False


@dataclass
class ProviderInfo:
    npi: str
    entity_type: str  # individual, organization
    status: ProviderStatus

    # Name fields (individual)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    credential: Optional[str] = None

    # Name fields (organization)
    organization_name: Optional[str] = None

    display_name: str = ""
    taxonomies: List[ProviderTaxonomy] = field(default_factory=list)
    primary_specialty: Optional[str] = None
    addresses: List[ProviderAddress] = field(default_factory=list)
    practice_address: Optional[ProviderAddress] = None
    other_identifiers: List[Dict[str, str]] = field(default_factory=list)

    enumeration_date: Optional[str] = None
    last_update: Optional[str] = None
    deactivation_date: Optional[str] = None
    reactivation_date: Optional[str] = None

    raw_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.display_name:
            if self.entity_type == "individual":
                parts = [self.prefix, self.first_name, self.middle_name, self.last_name, self.suffix]
                self.display_name = " ".join(p for p in parts if p)
                if self.credential:
                    self.display_name += f", {self.credential}"
            else:
                self.display_name = self.organization_name or ""

        if not self.primary_specialty and self.taxonomies:
            primary = next((t for t in self.taxonomies if t.is_primary), None)
            if primary:
                self.primary_specialty = primary.description
            elif self.taxonomies:
                self.primary_specialty = self.taxonomies[0].description

    @property
    def is_rehab_provider(self) -> bool:
        """Check if any taxonomy matches rehabilitation disciplines."""
        return any(t.code in REHAB_TAXONOMY_CODES for t in self.taxonomies)

    @property
    def discipline(self) -> Optional[str]:
        """Return PT/OT/SLP based on taxonomy, or None."""
        for t in self.taxonomies:
            code = t.code
            if code.startswith("2251") or code == "225200000X":
                return "PT"
            elif code.startswith("225X") or code == "224Z00000X":
                return "OT"
            elif code.startswith("235Z") or code.startswith("2355"):
                return "SLP"
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        addr = self.practice_address
        return {
            "npi": self.npi,
            "name": self.display_name,
            "first_name": self.first_name or "",
            "last_name": self.last_name or "",
            "credential": self.credential or "",
            "specialty": self.primary_specialty or "",
            "discipline": self.discipline,
            "is_rehab_provider": self.is_rehab_provider,
            "phone": addr.phone if addr else "",
            "fax": addr.fax if addr else "",
            "city": addr.city if addr else "",
            "state": addr.state if addr else "",
            "address": (
                f"{addr.address_line1}, {addr.city}, {addr.state} {addr.postal_code}"
                if addr else ""
            ),
            "status": self.status.value,
            "enumeration_date": self.enumeration_date,
        }


class NPPESClient:
    """
    Client for NPPES NPI Registry API.

    Provides NPI lookup, provider search, verification, and caching.
    No API key required — this is a free public CMS API.
    """

    def __init__(
        self,
        cache_ttl_hours: int = CACHE_TTL_HOURS,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.timeout = timeout
        self._cache: Dict[str, Tuple[datetime, Any]] = {}

    def _cache_key(self, op: str, **params) -> str:
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(f"{op}:{param_str}".encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Any]:
        if key in self._cache:
            cached_at, data = self._cache[key]
            if datetime.now(timezone.utc) - cached_at < self.cache_ttl:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, key: str, data: Any) -> None:
        self._cache[key] = (datetime.now(timezone.utc), data)

    def _request_sync(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["version"] = NPPES_API_VERSION
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(NPPES_API_URL, params=params)
            response.raise_for_status()
            return response.json()

    async def _request_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["version"] = NPPES_API_VERSION
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(NPPES_API_URL, params=params)
            response.raise_for_status()
            return response.json()

    def _parse_provider(self, data: Dict[str, Any]) -> ProviderInfo:
        basic = data.get("basic", {})
        entity_type = "individual" if data.get("enumeration_type") == "NPI-1" else "organization"

        taxonomies = []
        for tax in data.get("taxonomies", []):
            taxonomies.append(ProviderTaxonomy(
                code=tax.get("code", ""),
                description=tax.get("desc", ""),
                license=tax.get("license"),
                state=tax.get("state"),
                is_primary=tax.get("primary", False),
            ))

        addresses = []
        practice_addr = None
        for addr in data.get("addresses", []):
            pa = ProviderAddress(
                address_type=addr.get("address_purpose", "").lower(),
                address_line1=addr.get("address_1", ""),
                address_line2=addr.get("address_2"),
                city=addr.get("city", ""),
                state=addr.get("state", ""),
                postal_code=addr.get("postal_code", ""),
                country=addr.get("country_name", "US"),
                phone=addr.get("telephone_number"),
                fax=addr.get("fax_number"),
            )
            addresses.append(pa)
            if pa.address_type == "location":
                practice_addr = pa

        status = ProviderStatus.VERIFIED
        if basic.get("deactivation_date"):
            status = ProviderStatus.DEACTIVATED

        return ProviderInfo(
            npi=data.get("number", ""),
            entity_type=entity_type,
            status=status,
            first_name=basic.get("first_name"),
            last_name=basic.get("last_name"),
            middle_name=basic.get("middle_name"),
            prefix=basic.get("name_prefix"),
            suffix=basic.get("name_suffix"),
            credential=basic.get("credential"),
            organization_name=basic.get("organization_name"),
            taxonomies=taxonomies,
            addresses=addresses,
            practice_address=practice_addr,
            enumeration_date=basic.get("enumeration_date"),
            last_update=basic.get("last_updated"),
            deactivation_date=basic.get("deactivation_date"),
            reactivation_date=basic.get("reactivation_date"),
            raw_data=data,
        )

    # ── Sync API ────────────────────────────────────────────────────────

    def lookup_npi(self, npi: str, use_cache: bool = True) -> Optional[ProviderInfo]:
        """Look up a provider by NPI (10-digit)."""
        if not npi or len(npi) != 10 or not npi.isdigit():
            logger.warning(f"Invalid NPI format: {npi}")
            return None

        key = self._cache_key("lookup", number=npi)
        if use_cache:
            cached = self._get_cached(key)
            if cached is not None:
                return cached

        try:
            result = self._request_sync({"number": npi})
            results = result.get("results", [])
            logger.info(f"NPI lookup {npi[:4]}****: {len(results)} results")

            if results:
                provider = self._parse_provider(results[0])
                self._set_cache(key, provider)
                return provider
            self._set_cache(key, None)
            return None
        except Exception as e:
            logger.error(f"NPI lookup failed: {e}")
            return None

    def search_providers(
        self,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        organization_name: Optional[str] = None,
        specialty: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        postal_code: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        taxonomy_code: Optional[str] = None,
        limit: int = 20,
        skip: int = 0,
        use_cache: bool = True,
    ) -> List[ProviderInfo]:
        """Search NPPES for providers by various criteria."""
        params: Dict[str, Any] = {"limit": min(limit, 200), "skip": skip}

        if first_name:
            params["first_name"] = first_name
        if last_name:
            params["last_name"] = last_name
        if organization_name:
            params["organization_name"] = organization_name
        if specialty:
            params["taxonomy_description"] = specialty
        if city:
            params["city"] = city
        if state:
            params["state"] = state
        if postal_code:
            params["postal_code"] = postal_code
        if entity_type:
            params["enumeration_type"] = f"NPI-{entity_type.value}"
        if taxonomy_code:
            params["taxonomy_code"] = taxonomy_code

        key = self._cache_key("search", **params)
        if use_cache:
            cached = self._get_cached(key)
            if cached is not None:
                return cached

        try:
            result = self._request_sync(params)
            results = result.get("results", [])
            logger.info(f"NPPES search: {len(results)} results")
            providers = [self._parse_provider(r) for r in results]
            self._set_cache(key, providers)
            return providers
        except Exception as e:
            logger.error(f"Provider search failed: {e}")
            return []

    def search_rehab_providers(
        self,
        name: Optional[str] = None,
        state: Optional[str] = None,
        discipline: Optional[str] = None,
        limit: int = 20,
    ) -> List[ProviderInfo]:
        """Search specifically for PT/OT/SLP providers.

        Args:
            name: Provider name (splits into first/last)
            state: 2-letter state code
            discipline: PT, OT, or SLP
            limit: Max results
        """
        first_name = None
        last_name = None

        if name:
            parts = name.strip().split()
            prefixes = {"dr", "dr.", "doctor", "pt", "ot", "slp"}
            parts = [p for p in parts if p.lower() not in prefixes]
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = parts[-1]
            elif len(parts) == 1:
                last_name = parts[0] + "*"

        # Map discipline to taxonomy description
        specialty_map = {
            "PT": "Physical Therapist",
            "OT": "Occupational Therapist",
            "SLP": "Speech-Language Pathologist",
        }
        specialty = specialty_map.get(discipline) if discipline else None

        providers = self.search_providers(
            first_name=first_name,
            last_name=last_name,
            state=state,
            specialty=specialty,
            entity_type=EntityType.INDIVIDUAL,
            limit=limit,
        )

        # Filter out deactivated
        return [p for p in providers if p.status != ProviderStatus.DEACTIVATED]

    def verify_provider(self, npi: str) -> Tuple[bool, Optional[ProviderInfo]]:
        """Verify an NPI is valid and active."""
        provider = self.lookup_npi(npi)
        if provider is None:
            return False, None
        if provider.status == ProviderStatus.DEACTIVATED:
            return False, provider
        return True, provider

    # ── Async API ───────────────────────────────────────────────────────

    async def lookup_npi_async(self, npi: str, use_cache: bool = True) -> Optional[ProviderInfo]:
        """Async NPI lookup."""
        if not npi or len(npi) != 10 or not npi.isdigit():
            return None

        key = self._cache_key("lookup", number=npi)
        if use_cache:
            cached = self._get_cached(key)
            if cached is not None:
                return cached

        try:
            result = await self._request_async({"number": npi})
            results = result.get("results", [])
            if results:
                provider = self._parse_provider(results[0])
                self._set_cache(key, provider)
                return provider
            self._set_cache(key, None)
            return None
        except Exception as e:
            logger.error(f"Async NPI lookup failed: {e}")
            return None

    async def search_providers_async(
        self,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        state: Optional[str] = None,
        specialty: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        limit: int = 20,
    ) -> List[ProviderInfo]:
        """Async provider search."""
        params: Dict[str, Any] = {"limit": min(limit, 200)}
        if first_name:
            params["first_name"] = first_name
        if last_name:
            params["last_name"] = last_name
        if state:
            params["state"] = state
        if specialty:
            params["taxonomy_description"] = specialty
        if entity_type:
            params["enumeration_type"] = f"NPI-{entity_type.value}"

        key = self._cache_key("search_async", **params)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        try:
            result = await self._request_async(params)
            results = result.get("results", [])
            providers = [self._parse_provider(r) for r in results]
            self._set_cache(key, providers)
            return providers
        except Exception as e:
            logger.error(f"Async provider search failed: {e}")
            return []

    def clear_cache(self) -> None:
        self._cache.clear()

    def cache_stats(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        valid = sum(1 for t, _ in self._cache.values() if now - t < self.cache_ttl)
        return {"total": len(self._cache), "valid": valid, "ttl_hours": CACHE_TTL_HOURS}


# Module-level singleton
_client: Optional[NPPESClient] = None


def get_nppes_client() -> NPPESClient:
    """Get the singleton NPPES client."""
    global _client
    if _client is None:
        _client = NPPESClient()
    return _client
