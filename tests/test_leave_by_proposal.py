"""drive_time_alerts agent proposes a "leave by" reminder card (the pivot: propose,
not autorun). Deduped once per event, only within the lead window, and the reminder
time is the ABSOLUTE departure_time (a card is tapped later than proposed).
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from agents.drive_time_alerts.agent import DriveTimeAlertAgent

_NOW = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)


def _call(agent, departure, minutes_until, event_id="evt-1"):
    agent._maybe_propose_leave_by(
        "Dentist", event_id, departure, minutes_until, 18, "123 Main St",
        "3:00 PM", "12:40 PM", "2026-08-13")


def test_proposes_reminder_at_absolute_departure_within_lead_window():
    agent = DriveTimeAlertAgent()
    departure = _NOW + timedelta(minutes=40)   # inside 120-min lead window
    with patch("agents.drive_time_alerts.agent.JarvisInbox") as Inbox, \
         patch.object(agent, "_resolve_calendar_user_id", return_value=7):
        _call(agent, departure, 40)
    kwargs = Inbox.return_value.propose_action.call_args.kwargs
    assert kwargs["target_command"] == "reminder"
    assert kwargs["action"] == "set_at"
    assert kwargs["params"]["due_at_iso"] == departure.isoformat()   # absolute, not relative
    assert kwargs["params"]["text"] == "Leave for Dentist"
    assert kwargs["user_id"] == 7
    assert kwargs["source"] == "leaveby:evt-1"        # deterministic key for suppression


def test_no_proposal_outside_lead_window():
    agent = DriveTimeAlertAgent()
    departure = _NOW + timedelta(minutes=200)   # > 120-min lead
    with patch("agents.drive_time_alerts.agent.JarvisInbox") as Inbox:
        _call(agent, departure, 200)
    Inbox.assert_not_called()


def test_no_proposal_when_departure_already_passed():
    agent = DriveTimeAlertAgent()
    with patch("agents.drive_time_alerts.agent.JarvisInbox") as Inbox:
        _call(agent, _NOW - timedelta(minutes=5), -5)
    Inbox.assert_not_called()


def test_dedup_proposes_once_per_event():
    agent = DriveTimeAlertAgent()
    departure = _NOW + timedelta(minutes=40)
    with patch("agents.drive_time_alerts.agent.JarvisInbox") as Inbox, \
         patch.object(agent, "_resolve_calendar_user_id", return_value=None):
        _call(agent, departure, 40)
        _call(agent, departure, 40)
    assert Inbox.return_value.propose_action.call_count == 1


# ── home_context origin resolution (drops the HOME_ADDRESS secret dependency) ──
def test_agent_resolve_home_address_prefers_home_context():
    from jarvis_command_sdk import set_home_context
    agent = DriveTimeAlertAgent()
    set_home_context({"location": "Springfield, IL 62704"})
    try:
        with patch("agents.drive_time_alerts.agent._storage") as st:
            st.get_secret.return_value = "123 Secret St"   # secret present but home_context wins
            assert agent._resolve_home_address() == "Springfield, IL 62704"
    finally:
        set_home_context(None)


def test_agent_resolve_home_address_falls_back_to_secret():
    from jarvis_command_sdk import set_home_context
    set_home_context(None)
    agent = DriveTimeAlertAgent()
    with patch("agents.drive_time_alerts.agent._storage") as st:
        st.get_secret.return_value = "123 Secret St"
        assert agent._resolve_home_address() == "123 Secret St"
