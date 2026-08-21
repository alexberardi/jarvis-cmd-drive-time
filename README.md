# jarvis-cmd-drive-time

Traffic-aware driving time, distance, and "leave by" reminders for [Jarvis](https://github.com/alexberardi/jarvis-node-setup). Ask how long it takes to drive anywhere, and let Jarvis watch your calendar and remind you when it's time to head out.

## Install

```bash
python scripts/command_store.py install --url https://github.com/alexberardi/jarvis-cmd-drive-time
```

Requires the Jarvis SDK ≥ 0.9.0 (bundled with jarvis-node-setup ≥ v0.3.0).

## Voice Commands

- "How long to the airport?"
- "What's my commute look like?"
- "How far is downtown from here?"
- "How long will it take to drive to 123 Main Street?"
- "What's the drive time to the office from Mom's house?"

Answers are real-time and traffic-aware — Jarvis calls out when a route is heavier or lighter than usual so departure estimates reflect what's actually on the road.

## Leave-by reminders

A background agent watches your calendar for events that have a location. As departure time approaches, it proposes a tap-to-confirm **"Set a reminder to leave on time?"** card that factors in current traffic — so you head out with enough time to make it, without ever having to ask. Cards are deduplicated to once per event and only surface within a lead window before you need to go.

## Components

| Component | Type | Description |
|-----------|------|-------------|
| get_drive_time | command | Traffic-aware driving time + distance to any destination |
| drive_time_alerts | agent | Watches calendar events with a location and proposes "leave by" reminders |

## Configuration

| Secret | Required | Description |
|--------|----------|-------------|
| GOOGLE_MAPS_API_KEY | yes | Google Maps API key with **Directions** + **Geocoding** APIs enabled |
| HOME_ADDRESS | yes | Default origin for drive-time queries |
| DRIVE_TIME_BUFFER_MINUTES | no | Minutes of buffer to add before departure (default: 10) |

**Origin resolution:** by default, queries start from your household location (`home_context`) when it's set, otherwise from `HOME_ADDRESS`. You can also name any origin in the request ("…from the office").

## Structure

```
commands/get_drive_time/command.py   # Voice command
agents/drive_time_alerts/agent.py    # Leave-by reminder agent
drive_time_shared/                   # Shared Directions/Geocoding client + location resolution
tests/                               # Command + resolution tests
```

## License

MIT
