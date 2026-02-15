"""Route optimization for home-health scheduling."""

import re
from typing import Optional

from rehab_os.scheduling.models import Appointment


def _extract_zip(location: str | None) -> str | None:
    """Extract a 5-digit US zip code from a location string."""
    if not location:
        return None
    m = re.search(r"\b(\d{5})\b", location)
    return m.group(1) if m else None


class RouteOptimizer:
    """Simple nearest-neighbour route optimizer using zip-code proximity.

    This is a placeholder heuristic. In production, swap
    `estimate_travel_time` for a real Google Maps / OSRM call.
    """

    def estimate_travel_time(self, location_a: str, location_b: str) -> int:
        """Estimate travel minutes between two locations.

        Heuristic:
        - Same zip → 10 min
        - Adjacent zips (differ by ≤2) → 20 min
        - Otherwise → 30 min
        """
        zip_a = _extract_zip(location_a)
        zip_b = _extract_zip(location_b)

        if zip_a and zip_b:
            if zip_a == zip_b:
                return 10
            try:
                if abs(int(zip_a) - int(zip_b)) <= 2:
                    return 20
            except ValueError:
                pass
        return 30

    def optimize_daily_route(
        self,
        appointments: list[Appointment],
        provider_home_address: Optional[str] = None,
    ) -> list[Appointment]:
        """Reorder appointments to minimise travel (nearest-neighbour).

        Starts from *provider_home_address* if given, otherwise from the
        first appointment's location.
        """
        if len(appointments) <= 1:
            return list(appointments)

        remaining = list(appointments)
        ordered: list[Appointment] = []
        current_location = provider_home_address or (
            remaining[0].time_slot.location or ""
        )

        while remaining:
            best_idx = 0
            best_time = self.estimate_travel_time(
                current_location, remaining[0].time_slot.location or ""
            )
            for i, appt in enumerate(remaining[1:], 1):
                t = self.estimate_travel_time(
                    current_location, appt.time_slot.location or ""
                )
                if t < best_time:
                    best_time = t
                    best_idx = i

            chosen = remaining.pop(best_idx)
            ordered.append(chosen)
            current_location = chosen.time_slot.location or current_location

        return ordered

    def flag_infeasible_schedule(
        self,
        appointments: list[Appointment],
        max_travel_minutes: int = 45,
    ) -> list[str]:
        """Flag consecutive appointments whose estimated travel exceeds threshold."""
        warnings: list[str] = []
        for i in range(len(appointments) - 1):
            loc_a = appointments[i].time_slot.location or ""
            loc_b = appointments[i + 1].time_slot.location or ""
            travel = self.estimate_travel_time(loc_a, loc_b)
            if travel > max_travel_minutes:
                warnings.append(
                    f"Travel from '{loc_a}' to '{loc_b}' estimated at {travel} min "
                    f"(exceeds {max_travel_minutes} min threshold)"
                )
        return warnings
