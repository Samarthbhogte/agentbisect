"""FakeLLM rules that reproduce the calendar bug offline.

The buggy `create_event` description says "at the given time" without naming a timezone, and the
model then books in UTC 85% of the time. The fixed description says the start must be in the
user's local timezone, which suppresses the rule.
"""

from __future__ import annotations

from agentbisect.providers.fake import Rule, text, tool_use

BUG_TEXT = "at the given time"
FIX_TEXT = "user's local timezone"

RULES = [
    Rule(
        name="read-profile",
        when_all=["schedule"],
        unless_any=["timezone:"],
        max_turn=0,  # the first step
        respond=tool_use("get_user_profile", {}),
    ),
    Rule(
        name="book-in-utc",
        when_all=[BUG_TEXT],
        unless_any=["Created event", FIX_TEXT],
        min_turn=1,
        respond=tool_use(
            "create_event", {"title": "Design review", "start": "2026-12-01T15:00:00Z"}
        ),
        probability=0.85,
    ),
    Rule(
        name="book-in-local-time",
        when_all=["schedule"],
        unless_any=["Created event"],
        min_turn=1,
        respond=tool_use(
            "create_event", {"title": "Design review", "start": "2026-12-01T15:00:00-05:00"}
        ),
    ),
]
DEFAULT = text("Your design review is booked for Dec 1 at 3pm.")
