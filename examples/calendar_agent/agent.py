"""Calendar agent: schedules meetings.

Bug: the `create_event` tool description says "at the given time" and never mentions a
timezone, so the agent books meetings in UTC instead of the user's zone.

    python -m examples.calendar_agent.agent --provider fake
"""

from __future__ import annotations

from typing import Any

import agentbisect as relay

from .._common import DEFAULT_MODEL, cli, make_client, run_agent
from .fake_rules import DEFAULT, RULES
from .tools import create_event, get_user_profile

SYSTEM = """You are a scheduling assistant. Look up the user's profile before booking anything,
then create the event they ask for and confirm it in one sentence."""

_START = {"type": "string", "description": "ISO 8601 start time"}
CREATE_EVENT_BUGGY = "Creates a calendar event at the given time."
CREATE_EVENT_FIXED = (
    "Creates a calendar event. `start` must be an ISO 8601 time in the user's local timezone, "
    "with its UTC offset (for example 2026-12-01T15:00:00-05:00)."
)


def tools(fixed: bool) -> list[dict[str, Any]]:
    return [
        {
            "name": "get_user_profile",
            "description": "Get the user's profile: name, timezone, and working hours.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "create_event",
            "description": CREATE_EVENT_FIXED if fixed else CREATE_EVENT_BUGGY,
            "input_schema": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "start": _START},
                "required": ["title", "start"],
            },
        },
    ]


TASK = "Please schedule a design review for Dec 1 at 3pm."


@relay.entrypoint
def main(
    task: str = TASK,
    provider: str = "anthropic",
    model: str = DEFAULT_MODEL,
    seed: int = 0,
    fixed: bool = False,
) -> str:
    client = make_client(provider, RULES, DEFAULT, seed)
    run_input = {"task": task, "provider": provider, "model": model, "seed": seed, "fixed": fixed}
    with relay.run("calendar", input=run_input) as run:
        answer = run_agent(
            client,
            model=model,
            system=SYSTEM,
            tools=tools(fixed),
            tool_funcs={"get_user_profile": get_user_profile, "create_event": create_event},
            task=task,
        )
        run.set_output({"answer": answer})
    return answer


if __name__ == "__main__":
    cli(main, TASK)
