"""get_drive_time STRICT resolution — the CC leave-by bridge passes
resolution="strict" so a calendar location gets the Directions-raw guard (generic
types skipped, named places locality-biased), while the VOICE path (no resolution
kwarg) keeps its lenient passthrough. Mocks RoutingService — no network.
"""
import types

import pytest

import commands.get_drive_time.command as cmd_mod
from commands.get_drive_time.command import GetDriveTimeCommand
from drive_time_shared.routing_service import RouteResult

_HOME = "100 Test St, Testville CA 90000"


def _ri(home=_HOME):
    # The command only reads request_info.home_context — a namespace suffices.
    return types.SimpleNamespace(home_context={"location": home})


def _route():
    return RouteResult(
        duration_seconds=600, duration_text="10 mins",
        duration_in_traffic_seconds=None, duration_in_traffic_text=None,
        distance_meters=8000, distance_text="5 mi",
        origin_address=_HOME, destination_address="123 Main St, Testville CA 90000",
        summary="Main St",
    )


class _FakeRouting:
    """Captures the destination handed to Directions and returns a fixed route."""
    last: dict = {}

    def __init__(self, api_key):  # noqa: D401 — mimics RoutingService(api_key)
        pass

    def get_drive_time(self, origin, destination, departure_time=None):
        _FakeRouting.last = {"origin": origin, "destination": destination}
        return _route()


@pytest.fixture
def routing(monkeypatch):
    monkeypatch.setattr(
        cmd_mod._storage, "get_secret",
        lambda k, *a, **kw: "KEY" if k == "GOOGLE_MAPS_API_KEY" else None,
    )
    monkeypatch.setattr(cmd_mod, "RoutingService", _FakeRouting)
    _FakeRouting.last = {}
    return _FakeRouting


# ── strict mode (leave-by bridge) ───────────────────────────────────────────────
def test_strict_generic_location_skipped_without_needing_a_key():
    # A generic type is declined BEFORE the api_key is even required.
    resp = GetDriveTimeCommand().run(_ri(), destination="the dentist", resolution="strict")
    assert resp.success is False
    assert resp.context_data.get("reason") == "generic_location"


def test_strict_full_address_passed_verbatim(routing):
    resp = GetDriveTimeCommand().run(
        _ri(), destination="742 Evergreen Terrace, Springfield, IL 62704",
        resolution="strict",
    )
    assert resp.success is True
    assert routing.last["destination"] == "742 Evergreen Terrace, Springfield, IL 62704"


def test_strict_named_place_biased_with_household_locality(routing):
    resp = GetDriveTimeCommand().run(
        _ri(), destination="Community Hospital", resolution="strict",
    )
    assert resp.success is True
    assert routing.last["destination"] == "Community Hospital, Testville CA 90000"


# ── voice path (no resolution) is UNCHANGED ─────────────────────────────────────
def test_voice_generic_location_is_not_skipped(routing):
    # Without resolution="strict", "the mall" area-biased-geocodes as before — the
    # guard must NOT change the voice UX.
    resp = GetDriveTimeCommand().run(_ri(), destination="the mall")
    assert resp.success is True
    assert routing.last["destination"] == "the mall"  # raw, not skipped, not biased
