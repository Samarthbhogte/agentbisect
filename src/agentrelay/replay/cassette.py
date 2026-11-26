"""Cassette matching for program-level replay (BLUEPRINT 6.4)."""

from __future__ import annotations

import difflib
import json
from typing import Any

from agentrelay.schema import Run, Step, canonical_json


class ReplayDivergence(Exception):
    """A replayed LLM call does not match the recording (strict mode)."""

    def __init__(self, step: Step | None, index: int, diff: str, reason: str) -> None:
        where = step.id if step is not None else f"LLM call #{index + 1}"
        super().__init__(f"replay diverged at {where}: {reason}\n{diff}".rstrip())
        self.step = step
        self.step_id = step.id if step is not None else None
        self.call_index = index
        self.diff = diff
        self.reason = reason


class ToolNotInCassette(Exception):
    """`tools="cassette"`: a tool call with these arguments was never recorded."""


class SideEffectRefused(Exception):
    """A `side_effects=True` tool would run live during replay, which is not allowed."""


def _pretty(obj: Any) -> list[str]:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False).splitlines()


def request_diff(recorded: dict[str, Any], replayed: dict[str, Any], *, context: int = 2) -> str:
    """Unified diff of two normalized requests."""
    return "\n".join(
        difflib.unified_diff(
            _pretty(recorded),
            _pretty(replayed),
            fromfile="recorded",
            tofile="replay",
            lineterm="",
            n=context,
        )
    )


def divergence_reason(recorded: dict[str, Any], replayed: dict[str, Any]) -> str:
    """A one-line explanation: which parts of the request changed."""
    parts = []
    for key, label in (
        ("system", "the system prompt changed"),
        ("model", "the model changed"),
        ("tools", "the tool specs changed"),
        ("temperature", "the temperature changed"),
        ("max_tokens", "max_tokens changed"),
        ("extra", "other parameters changed"),
    ):
        if canonical_json(recorded.get(key)) != canonical_json(replayed.get(key)):
            parts.append(label)
    rec_msgs, rep_msgs = recorded.get("messages", []), replayed.get("messages", [])
    if canonical_json(rec_msgs) != canonical_json(rep_msgs):
        first = next(
            (
                i
                for i, (a, b) in enumerate(zip(rec_msgs, rep_msgs, strict=False))
                if canonical_json(a) != canonical_json(b)
            ),
            min(len(rec_msgs), len(rep_msgs)),
        )
        parts.append(f"the conversation differs from message {first} on")
    return "; ".join(parts) or "the request changed"


class Cassette:
    """The recorded LLM and tool steps of one run, consumed in order during replay."""

    def __init__(self, run: Run) -> None:
        self.run = run
        self.llm_steps = [s for s in run.steps if s.kind == "llm_call"]
        self.tool_steps = [s for s in run.steps if s.kind == "tool_call"]
        self._used_tools: set[str] = set()

    def llm_step(self, i: int) -> Step | None:
        return self.llm_steps[i] if i < len(self.llm_steps) else None

    def find_tool(
        self, name: str, args: dict[str, Any], *, before: int | None = None
    ) -> Step | None:
        """The first unused recorded call of `name` with exactly these arguments."""
        wanted = canonical_json(args)
        for step in self.tool_steps:
            if step.id in self._used_tools or (before is not None and step.index >= before):
                continue
            if step.name == name and canonical_json(step.request) == wanted:
                self._used_tools.add(step.id)
                return step
        return None
