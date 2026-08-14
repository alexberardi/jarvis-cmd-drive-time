"""Destination resolution (Directions-raw): route COMPLETE addresses verbatim,
SKIP generic place types ("don't route one of fifteen Dentists"), and bias any
other named place with the household locality so Google Directions geocodes the
nearby one. No web search — Directions does the geocoding.
"""
import pytest

from agents.drive_time_alerts.agent import (
    DriveTimeAlertAgent,
    _is_generic_location,
    _locality_of,
    _looks_like_full_address,
)

_HOME = "33 National Ave, Brick NJ 08724"


# ── _looks_like_full_address ─────────────────────────────────────────
@pytest.mark.parametrize("loc", [
    "742 Evergreen Terrace, Springfield, IL 62704",
    "500 Market St, Newark, NJ 07102",
    "33 National Ave, Brick NJ 08724",
    "123 Main St, Brick, NJ 08724",
])
def test_complete_address_detected(loc):
    assert _looks_like_full_address(loc)


@pytest.mark.parametrize("loc", [
    "Dentist", "the office", "Starbucks",
    "123 Main St",       # street but no locality anchor → bias + geocode, not verbatim
    "Gate 12345",        # a bare 5-digit must NOT read as an address
    "Room 5 Dr Patel",   # number + "Dr" (a person) must NOT read as an address
    "",
])
def test_incomplete_or_vague_not_full_address(loc):
    assert not _looks_like_full_address(loc)


# ── _is_generic_location ─────────────────────────────────────────────
@pytest.mark.parametrize("loc", [
    "Dentist", "dentist", "the office", "The Office", "gym", "the dentist",
    "doctor", "Doctor's", "pharmacy", "the gym", "airport", "a bar",
])
def test_generic_place_type_detected(loc):
    assert _is_generic_location(loc)


@pytest.mark.parametrize("loc", [
    "Starbucks", "ShopRite of Brick", "Jersey Shore Medical Center",
    "Dr. Patel's Dental", "Springfield Airport Diner",
])
def test_proper_or_brand_name_not_generic(loc):
    assert not _is_generic_location(loc)


# ── _locality_of ─────────────────────────────────────────────────────
def test_locality_strips_leading_street_address():
    assert _locality_of("33 National Ave, Brick NJ 08724") == "Brick NJ 08724"
    assert _locality_of("Brick, NJ") == "Brick, NJ"
    assert _locality_of("Springfield, IL 62704") == "Springfield, IL 62704"
    assert _locality_of("") == ""


# ── _resolve_destination ─────────────────────────────────────────────
def test_full_address_used_verbatim():
    agent = DriveTimeAlertAgent()
    dest = agent._resolve_destination("742 Evergreen Terrace, Springfield, IL 62704", _HOME)
    assert dest == "742 Evergreen Terrace, Springfield, IL 62704"


def test_generic_location_skipped():
    agent = DriveTimeAlertAgent()
    assert agent._resolve_destination("Dentist", _HOME) is None
    assert agent._resolve_destination("the office", _HOME) is None


def test_named_place_biased_with_household_locality():
    agent = DriveTimeAlertAgent()
    dest = agent._resolve_destination("Jersey Shore Medical Center", _HOME)
    assert dest == "Jersey Shore Medical Center, Brick NJ 08724"


def test_named_place_without_locality_passes_through():
    agent = DriveTimeAlertAgent()
    assert (agent._resolve_destination("Jersey Shore Medical Center", "")
            == "Jersey Shore Medical Center")


# ── HOME_ADDRESS is no longer a required secret (origin = household.location) ──
def _secret_keys(obj):
    return {getattr(s, "key", getattr(s, "_key", None)) for s in obj.required_secrets}


def test_home_address_no_longer_required_on_agent_or_command():
    from commands.get_drive_time.command import GetDriveTimeCommand
    for keys in (_secret_keys(DriveTimeAlertAgent()), _secret_keys(GetDriveTimeCommand())):
        assert "GOOGLE_MAPS_API_KEY" in keys
        assert "HOME_ADDRESS" not in keys


# ── event-start parsing (calendar returns a bare display time "6:45 PM") ──────
def test_parse_event_start_display_time():
    dt = DriveTimeAlertAgent()._parse_event_start("6:45 PM")
    assert dt is not None and dt.tzinfo is not None and dt.minute == 45


def test_parse_event_start_iso():
    dt = DriveTimeAlertAgent()._parse_event_start("2026-08-13T18:45:00+00:00")
    assert dt is not None and dt.hour == 18 and dt.minute == 45


def test_parse_event_start_invalid_returns_none():
    agent = DriveTimeAlertAgent()
    assert agent._parse_event_start("not a time") is None
    assert agent._parse_event_start(None) is None
