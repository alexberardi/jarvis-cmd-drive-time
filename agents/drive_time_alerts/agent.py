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
- Household location set (Settings → Household) — the drive-time origin, injected
  as home_context (a legacy HOME_ADDRESS secret is still honored as a fallback)
- Calendar package installed (optional — agent is a no-op without it)
"""

from __future__ import annotations

import re
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
    set_current_user_id,
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

# A COMPLETE street address (street number + street type, then within a short
# city/suite gap a locality anchor — an UPPERCASE state abbreviation or a ZIP) is
# handed to Google Directions verbatim. Requiring the anchor keeps vague strings
# that merely contain a number ("Gate 12345", "Room 5 Dr Patel") off the direct
# path. The state check is case-SENSITIVE so a stray 2-letter word ("Dr Ed")
# isn't read as a state; the street-type list is case-insensitive.
_STREET_TYPES = (
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|"
    r"Hwy|Highway|Ct|Court|Pl|Place|Pkwy|Parkway|Cir|Circle|Ter|Terrace|Sq|Square)"
)
_FULL_ADDR_RE = re.compile(
    r"\b\d{1,6}\s+"
    r"(?i:(?:[A-Za-z0-9.'\-]+\s+){0,4}" + _STREET_TYPES + r")\b"
    r".{0,40}?"
    r"(?:\b[A-Z]{2}\b(?:\s+\d{5}(?:-\d{4})?)?|\b\d{5}(?:-\d{4})?\b)"
)


def _looks_like_full_address(location: str) -> bool:
    return bool(_FULL_ADDR_RE.search(location or ""))


# Generic place-TYPES (common nouns) mean "MY dentist / MY gym", which no
# geocoder can pin without knowing which one — the "there could be fifteen
# Dentists, don't route any" rule. Skipped outright rather than geocoded to
# whichever Google returns first. Brand/proper names ("Starbucks", "ShopRite of
# Brick") are NOT here — the area-biased geocode of a named place is a fine answer.
_GENERIC_TERMS = {
    "dentist", "dentists", "doctor", "doctors", "dr", "physician", "pediatrician",
    "orthodontist", "dermatologist", "clinic", "hospital", "pharmacy", "drugstore",
    "gym", "office", "work", "the office", "school", "college", "university",
    "daycare", "restaurant", "cafe", "coffee", "coffee shop", "bar", "pub", "diner",
    "store", "shop", "mall", "market", "supermarket", "grocery", "grocery store",
    "bank", "atm", "gas", "gas station", "post office", "library", "park", "church",
    "temple", "mosque", "synagogue", "salon", "barber", "barbershop", "spa", "vet",
    "veterinarian", "hairdresser", "airport", "station", "hotel", "motel", "home",
    "house", "appointment", "meeting", "lunch", "dinner", "breakfast", "the doctor",
    "the dentist", "the gym", "the bank", "the store", "the pharmacy", "the vet",
}


def _is_generic_location(location: str) -> bool:
    """True when the location is a bare common-noun place type (optionally with a
    leading article / trailing possessive) — no geocoder can pin it to the user's
    specific place, so it's skipped. Proper/brand names fall through and geocode."""
    q = re.sub(r"[^\w\s]", "", (location or "").strip().lower())
    q = re.sub(r"^(?:the|a|an)\s+", "", q).strip()
    return q in _GENERIC_TERMS or ("the " + q) in _GENERIC_TERMS


def _locality_of(location: str) -> str:
    """City/state/ZIP portion of the household location, for geocode bias. The
    household location may be a full STREET address (it doubles as the drive-time
    origin), so strip a leading "<number> <street> <type>," segment; a plain
    locality passes through unchanged."""
    if not location:
        return ""
    parts = [p.strip() for p in location.split(",") if p.strip()]
    if len(parts) >= 2 and re.match(r"\d{1,6}\s+\S+", parts[0]):
        return ", ".join(parts[1:])
    return location


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
        # Origin is the household's `household.location` (injected as home_context),
        # not a per-agent home-address secret. Only the Maps key is configured here.
        # A legacy HOME_ADDRESS secret, if set, is still honored at runtime.
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
    def include_in_context(self) -> bool:
        return False  # we inject memories directly

    # ── Main loop ───────────────────────────────────────────────────

    async def run(self) -> None:
        """Fetch calendar events, compute drive times, inject context, generate alerts."""
        try:
            # Load secrets + resolve the origin (household home_context location,
            # injected by command-center, with the HOME_ADDRESS secret as fallback).
            api_key = _storage.get_secret("GOOGLE_MAPS_API_KEY")
            home_address = self._resolve_home_address()
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

        # The calendar command needs to know WHOSE calendar to read. Pass the
        # configured calendar user both on the request and via the ContextVar
        # (the command uses get_current_user_id() for its OAuth token lookup) —
        # without it, it refuses with "no speaker_user_id".
        user_id = self._resolve_calendar_user_id()
        if user_id is None:
            logger.info("Drive time agent: no calendar user configured — no events")
            return []
        set_current_user_id(user_id)

        cmd = ReadCalendarCommand()
        today = datetime.now().strftime("%Y-%m-%d")  # command wants an ISO date, not "today"
        request_info = RequestInformation(
            voice_command="calendar check",
            conversation_id="drive-time-agent",
            user_id=user_id,
        )

        response = cmd.run(request_info, resolved_datetimes=[today])
        if not response.success or not response.context_data:
            logger.info(
                "Drive time agent: calendar read failed",
                error=getattr(response, "error_details", None),
            )
            return []

        return response.context_data.get("events", [])

    def _local_tz(self):
        """The node's local timezone — calendar display times are local."""
        try:
            from zoneinfo import ZoneInfo

            from utils.timezone_util import get_user_timezone
            return ZoneInfo(get_user_timezone())
        except Exception:  # noqa: BLE001 — fall back to UTC if unavailable
            return timezone.utc

    def _parse_event_start(self, start_str) -> "datetime | None":
        """Parse a calendar event start into a tz-aware UTC datetime. Accepts an
        ISO datetime, OR a bare display time ("6:45 PM") which the calendar
        command emits with the date implied by today, interpreted in the node's
        local timezone."""
        if not isinstance(start_str, str):
            return None
        s = start_str.strip()
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
        tz = self._local_tz()
        for fmt in ("%I:%M %p", "%I:%M%p", "%I %p", "%H:%M"):
            try:
                t = datetime.strptime(s, fmt).time()
            except ValueError:
                continue
            d = datetime.now(tz).date()
            return datetime(d.year, d.month, d.day, t.hour, t.minute,
                            tzinfo=tz).astimezone(timezone.utc)
        return None

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

        # Parse the event start. The calendar command returns either an ISO
        # datetime OR a bare DISPLAY time ("6:45 PM") for today, so parse both.
        event_start = self._parse_event_start(start_str)
        if event_start is None:
            return None

        # Skip events that have already started
        minutes_until_start = (event_start - now).total_seconds() / 60
        if minutes_until_start < 0:
            return None

        # Resolve the destination. A full address routes directly; a generic type
        # ("Dentist", "the office") is skipped (we don't route one of fifteen); any
        # other named place is biased with the household locality and geocoded by
        # Directions. A place Directions can't route (no route) is skipped below.
        event_id = event.get("id", title)
        destination = self._resolve_destination(location, home_address)
        if not destination:
            return None

        # Get drive time (cached or fresh)
        route = self._get_cached_or_fresh_route(
            routing, home_address, destination, event_id, today, now,
        )
        if not route:
            return None

        # Prefer Directions' own geocoded address for display — cleaner than the
        # biased query string, and shows the user exactly where we'd route.
        display_location = route.destination_address or location

        # Calculate departure time
        drive_minutes = route.best_duration_minutes
        departure_time = event_start - timedelta(minutes=drive_minutes + buffer_minutes)
        minutes_until_departure = (departure_time - now).total_seconds() / 60
        # event_start/departure are tz-aware UTC — format the SPOKEN/card times in
        # the node's LOCAL timezone, else "6:31 PM" would read as "10:31 PM" (UTC).
        _tz = self._local_tz()
        leave_by_display = departure_time.astimezone(_tz).strftime("%I:%M %p").lstrip("0")
        start_display = event_start.astimezone(_tz).strftime("%I:%M %p").lstrip("0")

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
            drive_minutes, display_location, start_display, leave_by_display, today,
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

    # ── Origin resolution ───────────────────────────────────────────

    def _resolve_home_address(self) -> str | None:
        """The drive-time origin: the household's CC-injected home_context location
        (preferred — set once at the household level, no per-agent secret) with the
        HOME_ADDRESS secret as fallback. get_home_context() is seeded by the agent
        scheduler; Config.get_dict is the fallback when the ContextVar didn't
        propagate (older SDK / spawned thread)."""
        hc = None
        try:
            from jarvis_command_sdk import get_home_context
            hc = get_home_context()
        except Exception:  # noqa: BLE001 — older SDK without home_context
            hc = None
        if not hc:
            try:
                from utils.config_service import Config
                hc = Config.get_dict("home_context")
            except Exception:  # noqa: BLE001 — not in a node runtime
                hc = None
        return (hc or {}).get("location") or _storage.get_secret("HOME_ADDRESS")

    # ── Destination resolution ──────────────────────────────────────

    def _resolve_destination(self, location: str, home_address: str) -> str | None:
        """The destination string handed to Google Directions for this event.

        - A COMPLETE street address is used verbatim (Directions geocodes it).
        - A bare place TYPE ("Dentist", "the office") is SKIPPED — no geocoder can
          tell which specific one the user means (the "fifteen Dentists" rule).
        - Any other named place ("Jersey Shore Medical Center") is biased with the
          household locality ("<name>, <city ST ZIP>") so Directions geocodes the
          nearby one, not a same-named place elsewhere. Directions returns one best
          match, or no route — in which case the event is skipped upstream."""
        if _looks_like_full_address(location):
            return location
        if _is_generic_location(location):
            logger.info(
                "Generic event location — skipping (can't pin one specific place)",
                location=location,
            )
            return None
        locality = _locality_of(home_address or "")
        return f"{location}, {locality}" if locality else location

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
