"""Google Maps Directions API client for drive time calculations.

Provides sync and async methods for getting driving time and distance
between two locations, with optional real-time traffic estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import httpx

try:
    from jarvis_log_client import JarvisLogger
except ImportError:
    import logging

    class JarvisLogger:  # type: ignore[no-redef]
        def __init__(self, **kw: str) -> None:
            self._log = logging.getLogger(kw.get("service", __name__))

        def info(self, msg: str, **kw: object) -> None:
            self._log.info(msg)

        def warning(self, msg: str, **kw: object) -> None:
            self._log.warning(msg)

        def error(self, msg: str, **kw: object) -> None:
            self._log.error(msg)

        def debug(self, msg: str, **kw: object) -> None:
            self._log.debug(msg)


logger = JarvisLogger(service="jarvis-node")

DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
REQUEST_TIMEOUT = 15


@dataclass
class RouteResult:
    """Result from a driving directions query."""

    duration_seconds: int
    duration_text: str
    duration_in_traffic_seconds: int | None
    duration_in_traffic_text: str | None
    distance_meters: int
    distance_text: str
    origin_address: str
    destination_address: str
    summary: str

    @property
    def best_duration_seconds(self) -> int:
        """Return traffic-aware duration if available, otherwise standard."""
        return self.duration_in_traffic_seconds or self.duration_seconds

    @property
    def best_duration_text(self) -> str:
        """Return traffic-aware duration text if available, otherwise standard."""
        return self.duration_in_traffic_text or self.duration_text

    @property
    def best_duration_minutes(self) -> int:
        """Return best duration estimate in whole minutes (rounded up)."""
        return -(-self.best_duration_seconds // 60)  # ceiling division


class RoutingService:
    """Google Maps Directions API client."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def _build_params(
        self,
        origin: str,
        destination: str,
        departure_time: datetime | None = None,
    ) -> dict[str, str]:
        params: dict[str, str] = {
            "origin": origin,
            "destination": destination,
            "key": self._api_key,
            "mode": "driving",
        }
        if departure_time:
            params["departure_time"] = str(int(departure_time.timestamp()))
            params["traffic_model"] = "best_guess"
        return params

    def _parse_response(self, data: dict) -> RouteResult | None:
        """Parse the Directions API response into a RouteResult."""
        if data.get("status") != "OK" or not data.get("routes"):
            status = data.get("status", "UNKNOWN")
            error_msg = data.get("error_message", "")
            logger.warning(
                "Directions API returned non-OK status",
                status=status,
                error=error_msg,
            )
            return None

        route = data["routes"][0]
        leg = route["legs"][0]
        duration_in_traffic = leg.get("duration_in_traffic")

        return RouteResult(
            duration_seconds=leg["duration"]["value"],
            duration_text=leg["duration"]["text"],
            duration_in_traffic_seconds=(
                duration_in_traffic["value"] if duration_in_traffic else None
            ),
            duration_in_traffic_text=(
                duration_in_traffic["text"] if duration_in_traffic else None
            ),
            distance_meters=leg["distance"]["value"],
            distance_text=leg["distance"]["text"],
            origin_address=leg["start_address"],
            destination_address=leg["end_address"],
            summary=route.get("summary", ""),
        )

    def get_drive_time(
        self,
        origin: str,
        destination: str,
        departure_time: datetime | None = None,
    ) -> RouteResult | None:
        """Get driving time between two locations (synchronous).

        Args:
            origin: Starting address or place name.
            destination: Ending address or place name.
            departure_time: When to depart (enables traffic estimates).

        Returns:
            RouteResult with duration/distance/route info, or None on failure.
        """
        params = self._build_params(origin, destination, departure_time)
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                response = client.get(DIRECTIONS_URL, params=params)
                response.raise_for_status()
                return self._parse_response(response.json())
        except httpx.HTTPError as e:
            logger.error("Directions API request failed", error=str(e))
            return None

    async def get_drive_time_async(
        self,
        origin: str,
        destination: str,
        departure_time: datetime | None = None,
    ) -> RouteResult | None:
        """Get driving time between two locations (async).

        Same as get_drive_time but for use in async contexts (agents).
        """
        params = self._build_params(origin, destination, departure_time)
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(DIRECTIONS_URL, params=params)
                response.raise_for_status()
                return self._parse_response(response.json())
        except httpx.HTTPError as e:
            logger.error("Directions API request failed", error=str(e))
            return None
