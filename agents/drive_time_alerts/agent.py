"""DriveTimeAlertAgent — enriches calendar events with drive times and departure alerts.

Runs every 15 minutes. For each upcoming calendar event that has a location:
1. Computes driving time from home via Google Maps Directions API
2. Injects enriched context into the command center's memory system
   (e.g., "25 min drive to Dentist at 3:00 PM — leave by 2:25 PM")
3. Generates departure alerts when it's time to leave

The LLM uses injected memories to proactively reason about the user's
schedule without being explicitly asked — e.g., mentioning drive time when
the user asks "what does my afternoon look like?"

Requires:
- Google Maps API key (GOOGLE_MAPS_API_KEY)
- Home address (HOME_ADDRESS)
- Calendar package installed (optional — agent is a no-op without it)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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

        def warning(self, msg: str, **kw: object) -> None:
            self._log.warning(msg)

        def error(self, msg: str, **kw: object) -> None:
            self._log.error(msg)

        def debug(self, msg: str, **kw: object) -> None:
            self._log.debug(msg)

from jarvis_command_sdk import (
    AgentSchedule,
    Alert,
    IJarvisAgent,
    IJarvisSecret,
    JarvisInbox,
    JarvisSecret,
    JarvisStorage,
    RequestInformation,
)

# Only PROPOSE a leave-by reminder card once the departure is within this lead
# window — enough time to act on it, but not a flood of cards for every located
# event on the whole day's calendar at once.
MAX_LEAVE_BY_LEAD_MINUTES = 120

from drive_time_shared.routing_service import RouteResult, RoutingService

logger = JarvisLogger(service="jarvis-node")

REFRESH_INTERVAL_SECONDS = 900  # 15 minutes
CACHE_TTL_MINUTES = 30  # re-fetch drive time after this
DEFAULT_BUFFER_MINUTES = 10  # extra minutes before departure

_storage = JarvisStorage("drive_time_alerts")


class DriveTimeAlertAgent(IJarvisAgent):
    """Background agent that monitors calendar events and generates departure alerts."""

    def __init__(self) -> None:
        self._alerts: List[Alert] = []
        self._alerted_event_keys: set[str] = set()
        self._proposed_event_keys: set[str] = set()
        self._context_data: Dict[str, Any] = {}

    # ── Agent metadata ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "drive_time_alerts"

    @property
    def description(self) -> str:
        return (
            "Monitors calendar events with locations and alerts when "
            "it's time to leave based on real-time driving conditions"
        )

    @property
    def schedule(self) -> AgentSchedule:
        return AgentSchedule(
            interval_seconds=REFRESH_INTERVAL_SECONDS,
            run_on_startup=True,
        )

    @property
    def required_secrets(self) -> List[IJarvisSecret]:
        return [
            JarvisSecret(
                "GOOGLE_MAPS_API_KEY",
                "Google Maps API key with Directions API enabled",
                "integration",
                "string",
                required=True,
            ),
            JarvisSecret(
                "HOME_ADDRESS",
                "Your home address (origin for drive time calculations)",
                "integration",
                "string",
                required=True,
            ),
        ]

    @property
    def include_in_context(self) -> bool:
        return False  # we inject memories directly

    # ── Main loop ───────────────────────────────────────────────────

    async def run(self) -> None:
        """Fetch calendar events, compute drive times, inject context, generate alerts."""
        try:
            # Load secrets
            api_key = _storage.get_secret("GOOGLE_MAPS_API_KEY")
            home_address = _storage.get_secret("HOME_ADDRESS")
            if not api_key or not home_address:
                logger.debug("Drive time agent skipping — missing API key or home address")
                return

            buffer_str = _storage.get_secret("DRIVE_TIME_BUFFER_MINUTES")
            buffer_minutes = int(buffer_str) if buffer_str else DEFAULT_BUFFER_MINUTES

            # Fetch today's calendar events
            events = self._fetch_calendar_events()
            if not events:
                self._alerts = []
                self._context_data = {}
                return

            now = datetime.now(timezone.utc)
            routing = RoutingService(api_key)
            self._alerts = []
            memories: List[Dict[str, Any]] = []
            today = now.strftime("%Y-%m-%d")

            for event in events:
                result = self._process_event(
                    event, now, routing, home_address, buffer_minutes, today,
                )
                if result:
                    memories.append(result)

            # Inject drive time context into command center
            if memories:
                self._inject_memories(memories)

            self._context_data = {
                "events_with_drive_times": len(memories),
                "last_updated": now.isoformat(),
            }

        except Exception as e:
            logger.error("Drive time agent run failed", error=str(e))
            self._alerts = []

    # ── Calendar integration ────────────────────────────────────────

    def _fetch_calendar_events(self) -> List[Dict[str, Any]]:
        """Fetch today's calendar events via the calendar command.

        Imports ReadCalendarCommand at runtime — gracefully returns empty
        list if the calendar package is not installed.
        """
        try:
            try:
                from commands.get_calendar_events.command import ReadCalendarCommand
            except ImportError:
                from commands.custom_commands.get_calendar_events.command import (
                    ReadCalendarCommand,
                )
        except ImportError:
            logger.info(
                "Calendar package not installed — drive time agent has no events to process"
            )
            return []

        cmd = ReadCalendarCommand()
        today = datetime.now().strftime("%Y-%m-%d")
        request_info = RequestInformation(
            voice_command="calendar check",
            conversation_id="drive-time-agent",
        )

        response = cmd.run(request_info, resolved_datetimes=[today])
        if not response.success or not response.context_data:
            return []

        return response.context_data.get("events", [])

    # ── Per-event processing ────────────────────────────────────────

    def _process_event(
        self,
        event: Dict[str, Any],
        now: datetime,
        routing: RoutingService,
        home_address: str,
        buffer_minutes: int,
        today: str,
    ) -> Dict[str, Any] | None:
        """Process a single calendar event — compute drive time, maybe alert.

        Returns a memory dict for injection, or None if the event should be skipped.
        """
        location = (event.get("location") or "").strip()
        if not location:
            return None

        title = event.get("title") or event.get("summary", "Untitled event")
        start_str = event.get("start_time") or event.get("start")
        is_all_day = event.get("is_all_day", False)

        if not start_str or is_all_day:
            return None

        # Parse event start time
        try:
            if isinstance(start_str, str):
                start_str = start_str.replace("Z", "+00:00")
                event_start = datetime.fromisoformat(start_str)
                if event_start.tzinfo is None:
                    event_start = event_start.replace(tzinfo=timezone.utc)
            else:
                return None
        except (ValueError, TypeError):
            return None

        # Skip events that have already started
        minutes_until_start = (event_start - now).total_seconds() / 60
        if minutes_until_start < 0:
            return None

        # Get drive time (cached or fresh)
        event_id = event.get("id", title)
        route = self._get_cached_or_fresh_route(
            routing, home_address, location, event_id, today, now,
        )
        if not route:
            return None

        # Calculate departure time
        drive_minutes = route.best_duration_minutes
        departure_time = event_start - timedelta(minutes=drive_minutes + buffer_minutes)
        minutes_until_departure = (departure_time - now).total_seconds() / 60
        leave_by_display = departure_time.strftime("%I:%M %p").lstrip("0")
        start_display = event_start.strftime("%I:%M %p").lstrip("0")

        # Generate departure alert if approaching leave time
        self._maybe_alert(
            title, event_id, drive_minutes, minutes_until_departure,
            leave_by_display, start_display, now,
        )

        # Propose a tap-to-confirm card to set a persistent "leave by" reminder —
        # the user opts in (and can "Never suggest this") rather than us auto-creating
        # a reminder every time an appointment approaches.
        self._maybe_propose_leave_by(
            title, event_id, departure_time, minutes_until_departure,
            drive_minutes, location, start_display, leave_by_display, today,
        )

        # Build memory for injection
        content = (
            f"Drive to {title} ({start_display}): "
            f"~{drive_minutes} min from home"
        )
        if route.summary:
            content += f" via {route.summary}"
        content += f". Leave by {leave_by_display}."

        # TTL: expire when the event starts
        ttl_hours = max(0.5, minutes_until_start / 60)

        return {
            "content": content,
            "category": "travel",
            "key": f"drive_time:{today}:{event_id}",
            "ttl_hours": ttl_hours,
            "source": "drive-time-agent",
        }

    # ── Route caching ───────────────────────────────────────────────

    def _get_cached_or_fresh_route(
        self,
        routing: RoutingService,
        origin: str,
        destination: str,
        event_id: str,
        today: str,
        now: datetime,
    ) -> RouteResult | None:
        """Return a cached drive time if fresh, otherwise fetch a new one.

        Caches results in JarvisStorage to avoid excessive API calls.
        Cache TTL is CACHE_TTL_MINUTES (default 30 min) since traffic changes.
        """
        cache_key = f"route_cache:{today}:{event_id}"

        # Check cache
        cached = _storage.get(cache_key)
        if cached:
            cached_at_str = cached.get("cached_at", "")
            try:
                cached_at = datetime.fromisoformat(cached_at_str)
                age_minutes = (now - cached_at).total_seconds() / 60
                if age_minutes < CACHE_TTL_MINUTES:
                    return RouteResult(
                        duration_seconds=cached["duration_seconds"],
                        duration_text=cached["duration_text"],
                        duration_in_traffic_seconds=cached.get("duration_in_traffic_seconds"),
                        duration_in_traffic_text=cached.get("duration_in_traffic_text"),
                        distance_meters=cached["distance_meters"],
                        distance_text=cached["distance_text"],
                        origin_address=cached["origin_address"],
                        destination_address=cached["destination_address"],
                        summary=cached.get("summary", ""),
                    )
            except (ValueError, TypeError):
                pass  # stale or corrupt cache — re-fetch

        # Fetch fresh route
        route = routing.get_drive_time(origin, destination, departure_time=now)
        if not route:
            return None

        # Cache it
        _storage.save(
            cache_key,
            {
                "duration_seconds": route.duration_seconds,
                "duration_text": route.duration_text,
                "duration_in_traffic_seconds": route.duration_in_traffic_seconds,
                "duration_in_traffic_text": route.duration_in_traffic_text,
                "distance_meters": route.distance_meters,
                "distance_text": route.distance_text,
                "origin_address": route.origin_address,
                "destination_address": route.destination_address,
                "summary": route.summary,
                "cached_at": now.isoformat(),
            },
            expires_at=now + timedelta(hours=12),
        )

        return route

    # ── Alert generation ────────────────────────────────────────────

    def _maybe_alert(
        self,
        title: str,
        event_id: str,
        drive_minutes: int,
        minutes_until_departure: float,
        leave_by_display: str,
        start_display: str,
        now: datetime,
    ) -> None:
        """Generate a departure alert if the user should leave soon."""
        # Only alert within 30 minutes of departure time
        if minutes_until_departure < -5 or minutes_until_departure > 30:
            return

        if minutes_until_departure <= 5:
            event_key = f"{event_id}:leave_now"
            priority = 3
            ttl = timedelta(minutes=15)
            summary = (
                f"Leave now for {title} at {start_display} "
                f"({drive_minutes} min drive)"
            )
        elif minutes_until_departure <= 15:
            event_key = f"{event_id}:leave_soon"
            priority = 3
            ttl = timedelta(minutes=15)
            summary = (
                f"Leave in about {int(minutes_until_departure)} minutes "
                f"for {title} ({drive_minutes} min drive)"
            )
        else:
            event_key = f"{event_id}:heads_up"
            priority = 2
            ttl = timedelta(minutes=30)
            summary = (
                f"Heads up: {title} at {start_display} — "
                f"{drive_minutes} min drive, leave by {leave_by_display}"
            )

        if event_key in self._alerted_event_keys:
            return

        self._alerted_event_keys.add(event_key)

        self._alerts.append(Alert(
            source_agent=self.name,
            title=f"Time to leave: {title}",
            summary=summary,
            created_at=now,
            expires_at=now + ttl,
            priority=priority,
        ))

    # ── Leave-by proposal ───────────────────────────────────────────

    def _resolve_calendar_user_id(self) -> int | None:
        """Best-effort: which household member's calendar this is, to target the
        card. The reminder is ultimately owned by whoever CONFIRMS the card (their
        callback user_id), so None (a household card) is a safe fallback."""
        try:
            from get_calendar_events_shared.user_resolution import find_configured_user_ids
            uids = find_configured_user_ids()
            return uids[0] if uids else None
        except Exception:  # noqa: BLE001 — resolution optional; fall back to household
            return None

    def _maybe_propose_leave_by(
        self, title: str, event_id: str, departure_time: datetime,
        minutes_until_departure: float, drive_minutes: int, location: str,
        start_display: str, leave_by_display: str, today: str,
    ) -> None:
        """Propose a 'set a reminder to leave on time?' card — once per event, and
        only within the lead window (so we don't card every located event on the day
        at once, and the departure is still in the future when proposed)."""
        if not (0 < minutes_until_departure <= MAX_LEAVE_BY_LEAD_MINUTES):
            return
        key = f"{today}:{event_id}"
        if key in self._proposed_event_keys:
            return

        idempotency_key = f"leaveby:{key}"
        try:
            JarvisInbox("drive_time_alerts").propose_action(
                target_command="reminder",
                action="set_at",
                params={
                    "text": f"Leave for {title}",
                    "due_at_iso": departure_time.isoformat(),
                    "idempotency_key": idempotency_key,
                },
                idempotency_key=idempotency_key,
                editable=["text"],
                field_types={"due_at_iso": "datetime"},
                title="Set a reminder to leave on time?",
                summary=f"{title} at {start_display} — leave by {leave_by_display} ({drive_minutes} min drive)",
                body=(f"{title} starts at {start_display} at {location}. It's about "
                      f"{drive_minutes} min from home, so leave by {leave_by_display}."),
                # Deterministic key + semantic descriptor power "Never suggest this".
                source=f"leaveby:{event_id}",
                descriptor=f'leave-by reminders for calendar events like "{title}"',
                user_id=self._resolve_calendar_user_id(),
            )
            self._proposed_event_keys.add(key)
            logger.info("Proposed leave-by reminder", title=title, leave_by=leave_by_display)
        except Exception as e:  # noqa: BLE001 — a failed proposal must not break the run
            logger.warning("Leave-by proposal failed", error=str(e))

    # ── Memory injection ────────────────────────────────────────────

    def _inject_memories(self, memories: List[Dict[str, Any]]) -> None:
        """Push drive time context into the command center's memory system."""
        try:
            from clients.rest_client import RestClient
        except ImportError:
            logger.debug("RestClient not available — skipping memory injection")
            return

        result = RestClient.inject_memories(memories)
        if result:
            count = result.get("injected", 0) + result.get("updated", 0)
            logger.info("Drive time agent injected memories", count=count)

    # ── Interface methods ───────────────────────────────────────────

    def get_context_data(self) -> Dict[str, Any]:
        return self._context_data

    def get_alerts(self) -> List[Alert]:
        return list(self._alerts)
