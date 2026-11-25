"""Structural rules every (re)composed request must follow (BLUEPRINT 6.7).

* The conversation starts with `user` and roles alternate.
* Every assistant `tool_use` is answered by a `tool_result` with the same id in the very next
  (user) message, and every `tool_result` answers a `tool_use` in the message just before it.
* In that user message, `tool_result` blocks come before any other blocks.
* No message is empty, and `tool_use` ids are unique.
"""

from __future__ import annotations

from agentrelay.providers.base import ContentBlock, LLMRequest, Message


class InvalidRequest(ValueError):
    """A request breaks the provider's structural rules."""


def validate(req: LLMRequest, provider: str = "anthropic") -> list[str]:
    """All structural problems found (empty list = valid). `provider` is reserved for rules that
    differ between providers; Anthropic's are the strictest and apply to all today."""
    problems: list[str] = []
    msgs = req.messages
    if not msgs:
        return ["request has no messages"]
    if msgs[0].role != "user":
        problems.append("conversation must start with a user message")
    seen_ids: set[str] = set()
    for i, msg in enumerate(msgs):
        if not msg.content:
            problems.append(f"message {i} is empty")
        if i and msgs[i - 1].role == msg.role:
            problems.append(f"messages {i - 1} and {i} are both {msg.role}: roles must alternate")
        uses = _ids(msg, "tool_use")
        for use_id in uses:
            if use_id in seen_ids:
                problems.append(f"tool_use id {use_id} appears more than once")
            seen_ids.add(use_id)
        results = _ids(msg, "tool_result")
        if results and msg.role != "user":
            problems.append(f"message {i} has tool_result blocks but is not a user message")
        if results and not _results_first(msg):
            problems.append(f"message {i}: tool_result blocks must come before other content")
        prev_uses = _ids(msgs[i - 1], "tool_use") if i else []
        for rid in results:
            if rid not in prev_uses:
                problems.append(
                    f"message {i}: tool_result {rid} has no tool_use in message {i - 1}"
                )
        if uses:
            nxt = _ids(msgs[i + 1], "tool_result") if i + 1 < len(msgs) else []
            for use_id in uses:
                if use_id not in nxt:
                    problems.append(
                        f"message {i}: tool_use {use_id} is not answered in message {i + 1}"
                    )
    return problems


def assert_valid(req: LLMRequest, provider: str = "anthropic") -> None:
    problems = validate(req, provider)
    if problems:
        raise InvalidRequest("; ".join(problems))


def _ids(msg: Message, kind: str) -> list[str]:
    return [b.tool_use_id or "" for b in msg.content if b.type == kind]


def _results_first(msg: Message) -> bool:
    seen_other = False
    for b in msg.content:
        if b.type == "tool_result" and seen_other:
            return False
        if b.type != "tool_result":
            seen_other = True
    return True


def merge_same_role(messages: list[Message]) -> list[Message]:
    """Merge consecutive messages with the same role. In merged user messages, `tool_result`
    blocks move to the front (the provider requires them first). Already-alternating input is
    returned unchanged."""
    out: list[Message] = []
    for msg in messages:
        if out and out[-1].role == msg.role:
            merged = [*out[-1].content, *msg.content]
            if msg.role == "user":
                merged = _tool_results_first(merged)
            out[-1] = Message(role=msg.role, content=merged)
        else:
            out.append(msg)
    return out


def _tool_results_first(blocks: list[ContentBlock]) -> list[ContentBlock]:
    return [b for b in blocks if b.type == "tool_result"] + [
        b for b in blocks if b.type != "tool_result"
    ]
