"""Tools for the calendar agent (in-memory calendar)."""

from __future__ import annotations

import agentrelay as relay

PROFILE = "Name: Dana Rivera\ntimezone: America/New_York\nworking hours: 9am-5pm"
EVENTS: list[dict[str, str]] = []


@relay.tool
def get_user_profile() -> str:
    """Return the current user's profile, including their timezone."""
    return PROFILE


@relay.tool(side_effects=True)
def create_event(title: str, start: str) -> str:
    """Create a calendar event."""
    EVENTS.append({"title": title, "start": start})
    return f"Created event '{title}' starting {start}"
