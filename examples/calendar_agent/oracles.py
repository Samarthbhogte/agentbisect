"""Oracles for the calendar agent."""

from __future__ import annotations

from agentrelay import Failure, oracle
from agentrelay.providers.base import LLMResponse
from agentrelay.schema import Run


def _is_utc(start: str) -> bool:
    return start.endswith("Z") or start.endswith("+00:00")


@oracle("books-in-utc", level="decision")
def books_in_utc(resp: LLMResponse) -> Failure | None:
    for call in resp.tool_calls():
        start = (call.tool_input or {}).get("start", "")
        if call.tool_name == "create_event" and _is_utc(start):
            return Failure(fingerprint="create_event:utc", detail=start)
    return None


@oracle("run-books-in-utc", level="run")
def run_books_in_utc(run: Run) -> Failure | None:
    for step in run.steps:
        if step.kind == "tool_call" and step.name == "create_event":
            start = step.request.get("start", "")
            if _is_utc(start):
                return Failure(fingerprint="create_event:utc", detail=f"{step.id}: {start}")
    return None
