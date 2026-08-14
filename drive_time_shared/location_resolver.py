"""Directions-raw destination resolution — shared by the ``get_drive_time`` command
(strict mode, for the CC leave-by bridge) and the ``drive_time_alerts`` agent
(memories / alerts).

- A COMPLETE street address is handed to Google Directions verbatim.
- A bare generic place TYPE ("Dentist", "the office") is SKIPPED — no geocoder can
  tell which specific one the user means (the "fifteen Dentists" rule).
- Any other named place ("Jersey Shore Medical Center") is biased with the
  household locality ("<name>, <city ST ZIP>") so Directions geocodes the nearby
  one, not a same-named place elsewhere.

This logic used to live in the agent; it moved here so the command's strict path
(driven by the CC bridge) applies the SAME guard — a bridge that called
get_drive_time on a raw "Dentist" would otherwise route to an arbitrary dentist.
"""

from __future__ import annotations

import re

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


def looks_like_full_address(location: str) -> bool:
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


def is_generic_location(location: str) -> bool:
    """True when the location is a bare common-noun place type (optionally with a
    leading article / trailing possessive) — no geocoder can pin it to the user's
    specific place, so it's skipped. Proper/brand names fall through and geocode."""
    q = re.sub(r"[^\w\s]", "", (location or "").strip().lower())
    q = re.sub(r"^(?:the|a|an)\s+", "", q).strip()
    return q in _GENERIC_TERMS or ("the " + q) in _GENERIC_TERMS


def locality_of(location: str) -> str:
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


def resolve_destination(location: str, home_address: str) -> str | None:
    """The destination string handed to Google Directions for a calendar event.

    - A COMPLETE street address is used verbatim (Directions geocodes it).
    - A bare place TYPE ("Dentist", "the office") is SKIPPED (returns None) — no
      geocoder can tell which specific one the user means (the "fifteen Dentists"
      rule).
    - Any other named place ("Jersey Shore Medical Center") is biased with the
      household locality ("<name>, <city ST ZIP>") so Directions geocodes the
      nearby one, not a same-named place elsewhere.
    """
    if looks_like_full_address(location):
        return location
    if is_generic_location(location):
        return None
    locality = locality_of(home_address or "")
    return f"{location}, {locality}" if locality else location
