# calendar_agent

The agent schedules "a design review for Dec 1 at 3pm". It reads the user's profile (timezone
America/New_York), then calls `create_event`.

The buggy tool description is "Creates a calendar event at the given time." It never says which
timezone `start` is in.

**Bug:** the agent books `2026-12-01T15:00:00Z` (UTC) instead of `2026-12-01T15:00:00-05:00`
(about 85% of runs with FakeLLM). **Culprit:** the `create_event` tool description, which
exercises tool-spec context units. **Fix:** a description that requires an ISO 8601 time in the
user's local timezone with its UTC offset (`--fixed`).

```bash
relay record -- python -m examples.calendar_agent.agent --provider fake --seed 1
relay show last
```
