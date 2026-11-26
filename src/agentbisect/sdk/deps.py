"""Dependency linking between steps.

An LLM call depends on every tool step whose ``tool_use_id`` appears in a ``tool_result`` block of
its messages. Fallback: an earlier tool step whose output text appears verbatim in the request.
A tool call depends on the LLM call whose ``tool_use`` block requested it.
"""

from __future__ import annotations

from typing import Any

from agentbisect.providers.base import LLMRequest, render_context
from agentbisect.schema import Step, canonical_json

MIN_VERBATIM_CHARS = 16


def tool_output_text(step: Step) -> str:
    result = (step.response or {}).get("result")
    return result if isinstance(result, str) else canonical_json(result)


def llm_dependencies(request: LLMRequest, prior: list[Step]) -> list[str]:
    result_ids = {
        b.tool_use_id
        for m in request.messages
        for b in m.content
        if b.type == "tool_result" and b.tool_use_id
    }
    tool_steps = [s for s in prior if s.kind == "tool_call"]
    deps = {s.id for s in tool_steps if s.tags.get("tool_use_id") in result_ids}
    rendered = render_context(request)
    for s in tool_steps:
        if s.id in deps:
            continue
        out = tool_output_text(s)
        if len(out) >= MIN_VERBATIM_CHARS and out in rendered:
            deps.add(s.id)
    return sorted(deps)


def match_tool_use(
    name: str, args: dict[str, Any], prior: list[Step], claimed: set[str]
) -> tuple[str, str] | None:
    """Find the unclaimed `tool_use` block that requested this call: (tool_use_id, llm step id).

    Searches the most recent LLM calls first; exact argument match wins over name-only match.
    """
    fallback: tuple[str, str] | None = None
    for step in reversed(prior):
        if step.kind != "llm_call" or not step.response:
            continue
        for block in step.response.get("content") or []:
            if block.get("type") != "tool_use" or block.get("tool_name") != name:
                continue
            use_id = block.get("tool_use_id")
            if not use_id or use_id in claimed:
                continue
            if canonical_json(block.get("tool_input") or {}) == canonical_json(args):
                return use_id, step.id
            fallback = fallback or (use_id, step.id)
        if fallback:
            return fallback
    return fallback
