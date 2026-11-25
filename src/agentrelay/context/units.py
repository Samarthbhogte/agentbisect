"""Decompose an `LLMRequest` into `ContextUnit`s and recompose it from a subset (BLUEPRINT 6.7).

Unit ids:

* ``sys.<k>``                      a system-prompt section
* ``tool.<name>``                  a tool's description (removed = empty description)
* ``msg.<i>``                      message *i*: for user messages its non-tool content, for
                                   assistant messages the whole message
* ``msg.<i>.tr.<k>``               the *k*-th tool result in message *i*
* ``msg.<i>.tr.<k>.chunk.<c>``     a chunk of a long tool result (parent: the tool result)

Never removable: the first user message and the last user message with text (the task or latest
turn). In ``placeholder`` mode (default) a removed tool result keeps its block with
``[omitted by agentrelay]``, so assistant messages that call tools stay too (not removable). In
``delete`` mode an assistant message and the tool results answering it form one group and are
removed together.

`recompose(req, all_ids) == req`, and every recomposed request passes `validate()`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from agentrelay.context.constraints import merge_same_role
from agentrelay.providers.base import ContentBlock, LLMRequest, Message, ToolSpec
from agentrelay.schema import ContextUnit

Removal = Literal["placeholder", "delete"]
PLACEHOLDER = "[omitted by agentrelay]"
CHUNK_ABOVE_TOKENS = 400
CHUNK_TARGET_TOKENS = 200
PREVIEW_CHARS = 300

_HEADING = re.compile(r"^#{1,6}\s", re.M)
_XML_OPEN = re.compile(r"^<([A-Za-z_][\w\-]*)(\s[^>]*)?>", re.M)
_BLANK_RUN = re.compile(r"\n[ \t]*\n+")


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


# ---- splitting text -----------------------------------------------------------------------
def _cut(text: str, starts: Iterable[int]) -> list[str]:
    bounds = sorted({0, *(s for s in starts if 0 < s < len(text))})
    pieces = [text[a:b] for a, b in zip(bounds, [*bounds[1:], len(text)], strict=True)]
    return [p for p in pieces if p]


def split_sections(text: str) -> list[str]:
    """Markdown headings, else top-level XML-style tags, else blank-line paragraphs. The pieces
    are exact substrings: `"".join(split_sections(t)) == t`."""
    if _HEADING.search(text):
        return _cut(text, (m.start() for m in _HEADING.finditer(text)))
    if _XML_OPEN.search(text):
        return _cut(text, (m.start() for m in _XML_OPEN.finditer(text)))
    return _cut(text, (m.end() for m in _BLANK_RUN.finditer(text)))


def split_chunks(text: str, target_chars: int = CHUNK_TARGET_TOKENS * 4) -> list[str]:
    """Paragraphs, then lines for long paragraphs, packed into chunks of about `target_chars`.
    Exact substrings, so `"".join(split_chunks(t)) == t`."""
    pieces: list[str] = []
    for para in _cut(text, (m.end() for m in _BLANK_RUN.finditer(text))):
        if len(para) <= target_chars:
            pieces.append(para)
        else:
            pieces.extend(para.splitlines(keepends=True))
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + len(piece) <= target_chars:
            chunks[-1] += piece
        else:
            chunks.append(piece)
    return chunks


def needs_chunks(text: str) -> bool:
    return approx_tokens(text) > CHUNK_ABOVE_TOKENS


# ---- decompose ----------------------------------------------------------------------------
def _unit(
    uid: str,
    kind: str,
    path: str,
    text: str,
    *,
    group: str | None = None,
    parent: str | None = None,
    removable: bool = True,
) -> ContextUnit:
    return ContextUnit(
        id=uid,
        kind=kind,
        path=path,
        text=text[:PREVIEW_CHARS],
        tokens=approx_tokens(text),
        group=group,
        parent=parent,
        removable=removable,
    )


def _protected_messages(msgs: list[Message]) -> set[int]:
    """First user message and the last user message that contains text."""
    protected: set[int] = set()
    if msgs and msgs[0].role == "user":
        protected.add(0)
    texty = [
        i
        for i, m in enumerate(msgs)
        if m.role == "user" and any(b.type == "text" for b in m.content)
    ]
    if texty:
        protected.add(texty[-1])
    elif any(m.role == "user" for m in msgs):
        protected.add(max(i for i, m in enumerate(msgs) if m.role == "user"))
    return protected


def _message_text(msg: Message) -> str:
    parts = []
    for b in msg.content:
        if b.type == "text" and b.text:
            parts.append(b.text)
        elif b.type == "tool_use":
            parts.append(f"→ {b.tool_name}({b.tool_input})")
    return "\n".join(parts)


def _pair_groups(msgs: list[Message]) -> dict[str, str]:
    """tool_use_id -> group id of the assistant message that issued it."""
    return {
        b.tool_use_id or "": f"pair.msg.{i}"
        for i, m in enumerate(msgs)
        if m.role == "assistant"
        for b in m.content
        if b.type == "tool_use"
    }


def decompose(req: LLMRequest, removal: Removal = "placeholder") -> list[ContextUnit]:
    units: list[ContextUnit] = []
    if req.system:
        for k, section in enumerate(split_sections(req.system)):
            units.append(_unit(f"sys.{k}", "system_section", f"/system#{k}", section))
    for t_idx, tool in enumerate(req.tools):
        units.append(
            _unit(f"tool.{tool.name}", "tool_spec", f"/tools/{t_idx}/description", tool.description)
        )
    protected = _protected_messages(req.messages)
    pairs = _pair_groups(req.messages) if removal == "delete" else {}
    for i, msg in enumerate(req.messages):
        units.extend(_message_units(i, msg, protected, pairs, removal))
    return units


def _message_units(
    i: int, msg: Message, protected: set[int], pairs: dict[str, str], removal: Removal
) -> list[ContextUnit]:
    out: list[ContextUnit] = []
    calls_tools = any(b.type == "tool_use" for b in msg.content)
    has_own_content = any(b.type in ("text", "tool_use") for b in msg.content)
    if msg.role == "assistant":
        group = f"pair.msg.{i}" if calls_tools and removal == "delete" else None
        removable = not calls_tools or removal == "delete"
        out.append(
            _unit(
                f"msg.{i}",
                "message",
                f"/messages/{i}",
                _message_text(msg),
                group=group,
                removable=removable,
            )
        )
        return out
    if has_own_content:
        out.append(
            _unit(
                f"msg.{i}",
                "message",
                f"/messages/{i}",
                _message_text(msg),
                removable=i not in protected,
            )
        )
    k = 0
    for b_idx, block in enumerate(msg.content):
        if block.type != "tool_result":
            continue
        tr_id = f"msg.{i}.tr.{k}"
        path = f"/messages/{i}/content/{b_idx}"
        text = block.text or ""
        out.append(
            _unit(tr_id, "tool_result", path, text, group=pairs.get(block.tool_use_id or ""))
        )
        if needs_chunks(text) and block.raw is None:
            for c, chunk in enumerate(split_chunks(text)):
                out.append(
                    _unit(f"{tr_id}.chunk.{c}", "chunk", f"{path}/text#{c}", chunk, parent=tr_id)
                )
        k += 1
    return out


# ---- recompose ----------------------------------------------------------------------------
def effective_keep(units: list[ContextUnit], keep_ids: Iterable[str]) -> set[str]:
    """Which units survive: non-removable units always; a group only if every member is kept;
    a chunk only if its parent survives."""
    keep = set(keep_ids)
    members: dict[str, list[str]] = {}
    for u in units:
        if u.group:
            members.setdefault(u.group, []).append(u.id)

    def survives(u: ContextUnit) -> bool:
        if not u.removable:
            return True
        whole_group = u.group is None or all(m in keep for m in members[u.group])
        return u.id in keep and whole_group

    alive = {u.id for u in units if survives(u)}
    return {u.id for u in units if u.id in alive and (u.parent is None or u.parent in alive)}


def recompose(
    req: LLMRequest, keep_ids: Iterable[str], removal: Removal = "placeholder"
) -> LLMRequest:
    """The request with only `keep_ids` (plus non-removable units). Pure and deterministic."""
    units = decompose(req, removal)
    alive = effective_keep(units, keep_ids)
    by_id = {u.id: u for u in units}
    system = _recompose_system(req.system, alive)
    tools = [
        t
        if f"tool.{t.name}" in alive
        else ToolSpec(name=t.name, description="", input_schema=t.input_schema)
        for t in req.tools
    ]
    messages: list[Message] = []
    for i, msg in enumerate(req.messages):
        rebuilt = _recompose_message(i, msg, alive, by_id, removal)
        if rebuilt is not None:
            messages.append(rebuilt)
    if len(messages) != len(req.messages):
        messages = merge_same_role(messages)
    return req.model_copy(
        update={"system": system, "tools": tools, "messages": messages}, deep=True
    )


def _recompose_system(system: str | None, alive: set[str]) -> str | None:
    if not system:
        return system
    sections = split_sections(system)
    kept = [s for k, s in enumerate(sections) if f"sys.{k}" in alive]
    if len(kept) == len(sections):
        return system
    joined = "".join(kept).strip()
    return joined or None


def _recompose_message(
    i: int, msg: Message, alive: set[str], by_id: dict[str, ContextUnit], removal: Removal
) -> Message | None:
    msg_id = f"msg.{i}"
    if msg.role == "assistant":
        return msg if msg_id in alive else None
    own_kept = msg_id in alive or msg_id not in by_id
    blocks: list[ContentBlock] = []
    k = 0
    for block in msg.content:
        if block.type == "tool_result":
            tr = _recompose_result(block, f"msg.{i}.tr.{k}", alive, removal)
            k += 1
            if tr is not None:
                blocks.append(tr)
        elif block.type in ("text", "tool_use"):
            if own_kept:
                blocks.append(block)
        else:
            blocks.append(block)  # images and other blocks pass through untouched
    return Message(role=msg.role, content=blocks) if blocks else None


def _recompose_result(
    block: ContentBlock, tr_id: str, alive: set[str], removal: Removal
) -> ContentBlock | None:
    if tr_id not in alive:
        if removal == "delete":
            return None
        return block.model_copy(update={"text": PLACEHOLDER, "raw": None})
    text = block.text or ""
    if not (needs_chunks(text) and block.raw is None):
        return block
    chunks = split_chunks(text)
    kept = [c for n, c in enumerate(chunks) if f"{tr_id}.chunk.{n}" in alive]
    if len(kept) == len(chunks):
        return block
    return block.model_copy(update={"text": "".join(kept).strip() or PLACEHOLDER})


def all_ids(units: list[ContextUnit]) -> frozenset[str]:
    return frozenset(u.id for u in units)


def total_tokens(units: list[ContextUnit], ids: Iterable[str] | None = None) -> int:
    """Tokens of the given units (default: all), counting each chunked tool result once."""
    chosen = set(ids) if ids is not None else {u.id for u in units}
    chunked = {u.parent for u in units if u.parent}
    return sum(u.tokens for u in units if u.id in chosen and u.id not in chunked)


# ---- full text and substitution -----------------------------------------------------------
def _locate(req: LLMRequest, unit_id: str) -> tuple[int, int, int | None]:
    """(message index, block index, chunk index or None) for a tool result or chunk id."""
    parts = unit_id.split(".")
    if len(parts) < 4 or parts[0] != "msg" or parts[2] != "tr":
        raise KeyError(f"{unit_id} is not a tool result or chunk")
    i, k = int(parts[1]), int(parts[3])
    chunk = int(parts[5]) if len(parts) == 6 and parts[4] == "chunk" else None
    results = [b for b, block in enumerate(req.messages[i].content) if block.type == "tool_result"]
    return i, results[k], chunk


def unit_text(req: LLMRequest, unit: ContextUnit) -> str:
    """The unit's full text (`ContextUnit.text` is only a 300-character preview)."""
    if unit.kind == "system_section":
        return split_sections(req.system or "")[int(unit.id.split(".")[1])]
    if unit.kind == "tool_spec":
        name = unit.id.split(".", 1)[1]
        return next(t.description for t in req.tools if t.name == name)
    if unit.kind == "message":
        return _message_text(req.messages[int(unit.id.split(".")[1])])
    i, b, chunk = _locate(req, unit.id)
    text = req.messages[i].content[b].text or ""
    return text if chunk is None else split_chunks(text)[chunk]


def substitute(req: LLMRequest, unit_id: str, new_text: str) -> LLMRequest:
    """The request with one tool result or chunk replaced by `new_text` (everything else kept)."""
    i, b, chunk = _locate(req, unit_id)
    out = req.model_copy(deep=True)
    block = out.messages[i].content[b]
    if chunk is None:
        out.messages[i].content[b] = block.model_copy(update={"text": new_text, "raw": None})
    else:
        pieces = split_chunks(block.text or "")
        pieces[chunk] = new_text
        out.messages[i].content[b] = block.model_copy(update={"text": "".join(pieces)})
    return out
