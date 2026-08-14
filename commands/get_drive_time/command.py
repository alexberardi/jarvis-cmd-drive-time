"""get_drive_time — voice command for driving time and distance queries.

Handles direct asks like "How long to drive to the airport?" or
"What's my commute look like?" Uses Google Maps Directions API with
real-time traffic when available.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    from jarvis_log_client import JarvisLogger
except ImportError:
    import logging

    class JarvisLogger:  # type: ignore[no-redef]
        def __init__(self, **kw: str) -> None:
            self._log = logging.getLogger(kw.get("service", __name__))

        def info(self, msg: str, **kw: object) -> None:
            self._log.info(msg)

        def error(self, msg: str, **kw: object) -> None:
            self._log.error(msg)

from jarvis_command_sdk import (
    CommandExample,
    CommandResponse,
    IJarvisCommand,
    IJarvisParameter,
    IJarvisSecret,
    JarvisParameter,
    JarvisSecret,
    JarvisStorage,
    RequestInformation,
)

from drive_time_shared.location_resolver import resolve_destination
from drive_time_shared.routing_service import RoutingService

logger = JarvisLogger(service="jarvis-node")

_storage = JarvisStorage("get_drive_time")


class GetDriveTimeCommand(IJarvisCommand):
    """Get driving time and distance to a destination."""

    @property
    def command_name(self) -> str:
        return "get_drive_time"

    @property
    def description(self) -> str:
        return (
            "Get driving time and distance to a destination with real-time "
            "traffic estimates. Defaults to driving from the user's home address."
        )

    @property
    def parameters(self) -> List[IJarvisParameter]:
        return [
            JarvisParameter(
                name="destination",
                param_type="string",
                required=True,
                description="The destination address, place name, or landmark",
            ),
            JarvisParameter(
                name="origin",
                param_type="string",
                required=False,
                description=(
                    "Starting location (defaults to home address). "
                    "Only set if the user explicitly specifies a different origin."
                ),
            ),
        ]

    @property
    def required_secrets(self) -> List[IJarvisSecret]:
        # Origin comes from the household's `household.location` setting (injected
        # as home_context) — no per-command home-address secret. The GOOGLE_MAPS
        # key is the only thing to configure. (A legacy HOME_ADDRESS secret, if one
        # was set previously, is still honored at runtime as a fallback.)
        return [
            JarvisSecret(
                "GOOGLE_MAPS_API_KEY",
                "Google Maps API key with Directions API enabled",
                "integration",
                "string",
                required=True,
            ),
        ]

    @property
    def keywords(self) -> List[str]:
        return [
            "drive", "driving", "commute", "directions", "distance",
            "travel", "traffic", "route", "how long", "how far",
            "get there", "drive time", "leave", "departure",
        ]

    @property
    def associated_service(self) -> str | None:
        return "Google Maps"

    @property
    def setup_guide(self) -> str | None:
        return (
            "## Setup\n\n"
            "1. Go to [Google Cloud Console](https://console.cloud.google.com/)\n"
            "2. Enable the **Directions API**\n"
            "3. Create an API key and paste it here\n\n"
            "Your home address comes from your household location "
            "(Settings → Household), so there's nothing else to set here."
        )

    def generate_prompt_examples(self) -> List[CommandExample]:
        return [
            CommandExample(
                voice_command="How long to drive to the airport?",
                expected_parameters={"destination": "the airport"},
                is_primary=True,
            ),
            CommandExample(
                voice_command="What's my commute look like?",
                expected_parameters={"destination": "work"},
            ),
            CommandExample(
                voice_command="How far is it to 350 Fifth Avenue, New York?",
                expected_parameters={"destination": "350 Fifth Avenue, New York"},
            ),
            CommandExample(
                voice_command="Drive time from the office to the restaurant on Main Street",
                expected_parameters={
                    "destination": "the restaurant on Main Street",
                    "origin": "the office",
                },
            ),
        ]

    def generate_adapter_examples(self) -> List[CommandExample]:
        return [
            CommandExample(
                voice_command="How long does it take to drive to the grocery store?",
                expected_parameters={"destination": "the grocery store"},
            ),
            CommandExample(
                voice_command="What's the drive time to downtown?",
                expected_parameters={"destination": "downtown"},
            ),
            CommandExample(
                voice_command="How far is the dentist office?",
                expected_parameters={"destination": "the dentist office"},
            ),
            CommandExample(
                voice_command="How long to get to mom's house?",
                expected_parameters={"destination": "mom's house"},
            ),
            CommandExample(
                voice_command="Can you check the traffic to the mall?",
                expected_parameters={"destination": "the mall"},
            ),
            CommandExample(
                voice_command="Drive time to 123 Oak Street, Springfield",
                expected_parameters={"destination": "123 Oak Street, Springfield"},
            ),
            CommandExample(
                voice_command="How long would it take to drive from the hotel to the convention center?",
                expected_parameters={
                    "destination": "the convention center",
                    "origin": "the hotel",
                },
            ),
            CommandExample(
                voice_command="What's the drive from here to the beach?",
                expected_parameters={"destination": "the beach"},
            ),
            CommandExample(
                voice_command="How many minutes to drive to Target?",
                expected_parameters={"destination": "Target"},
            ),
            CommandExample(
                voice_command="Check how long it takes to get to the school",
                expected_parameters={"destination": "the school"},
            ),
            CommandExample(
                voice_command="Is it a long drive to the hospital?",
                expected_parameters={"destination": "the hospital"},
            ),
            CommandExample(
                voice_command="How far away is the nearest Costco?",
                expected_parameters={"destination": "the nearest Costco"},
            ),
        ]

    def run(
        self, request_info: RequestInformation, **kwargs: Any
    ) -> CommandResponse:
        destination: str = kwargs.get("destination", "")
        origin: str | None = kwargs.get("origin")

        if not destination:
            return CommandResponse.error_response(
                "I need a destination. Where would you like to drive to?",
                context_data={"message": "Where would you like to drive to?"},
            )

        # Resolve origin — explicit origin > household home_context location
        # (injected by command-center, set once at the household level) > the
        # per-command HOME_ADDRESS secret (kept as a fallback; deprecate, don't delete).
        if not origin:
            hc = getattr(request_info, "home_context", None) or {}
            origin = hc.get("location") or _storage.get_secret("HOME_ADDRESS")

        if not origin:
            return CommandResponse.error_response(
                "Home address not configured.",
                context_data={
                    "message": (
                        "I need your home address to calculate drive time. "
                        "You can set it in the app under Drive Time settings."
                    )
                },
            )

        # STRICT resolution: the CC leave-by bridge passes resolution="strict" (an
        # UNDECLARED kwarg the voice path never sets) so the Directions-raw guard is
        # applied to a calendar location — a generic type ("Dentist") is skipped
        # rather than routed to an arbitrary nearby one. Runs before the API-key
        # check because a generic location needs no key to be declined. Voice queries
        # keep their lenient passthrough (e.g. "the mall" still area-biased-geocodes).
        if kwargs.get("resolution") == "strict":
            resolved = resolve_destination(destination, origin)
            if resolved is None:
                return CommandResponse.error_response(
                    f"'{destination}' is too generic to route to a specific place.",
                    context_data={
                        "skipped": True,
                        "reason": "generic_location",
                        "message": f"I couldn't tell which {destination} you mean.",
                    },
                )
            destination = resolved

        api_key = _storage.get_secret("GOOGLE_MAPS_API_KEY")
        if not api_key:
            return CommandResponse.error_response(
                "Google Maps API key not configured.",
                context_data={
                    "message": (
                        "I need a Google Maps API key to get driving directions. "
                        "You can add one in the app under Drive Time settings."
                    )
                },
            )

        # Get drive time with current traffic
        routing = RoutingService(api_key)
        now = datetime.now(timezone.utc)
        result = routing.get_drive_time(origin, destination, departure_time=now)

        if not result:
            return CommandResponse.error_response(
                f"Could not find a driving route to {destination}.",
                context_data={
                    "message": (
                        f"I couldn't find a driving route to {destination}. "
                        "Make sure the address or place name is correct."
                    )
                },
            )

        # Build the spoken response
        duration = result.best_duration_text
        parts = [f"It's about a {duration} drive to {result.destination_address}"]
        parts.append(f"({result.distance_text})")
        if result.summary:
            parts.append(f"via {result.summary}")

        # If traffic data is available and differs meaningfully from base
        if (
            result.duration_in_traffic_seconds
            and result.duration_seconds > 0
            and abs(result.duration_in_traffic_seconds - result.duration_seconds)
            > 120  # >2 min difference
        ):
            if result.duration_in_traffic_seconds > result.duration_seconds:
                parts.append("— traffic is a bit heavy right now")
            else:
                parts.append("— traffic is lighter than usual")

        message = " ".join(parts) + "."

        return CommandResponse.success_response(
            context_data={
                "message": message,
                "duration_seconds": result.best_duration_seconds,
                "duration_text": result.best_duration_text,
                "duration_minutes": result.best_duration_minutes,
                "distance_text": result.distance_text,
                "distance_meters": result.distance_meters,
                "origin": result.origin_address,
                "destination": result.destination_address,
                "route_summary": result.summary,
                "has_traffic_data": result.duration_in_traffic_seconds is not None,
            },
        )
