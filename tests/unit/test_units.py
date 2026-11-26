"""Context decomposition: recompose(all) == req, and every recomposition is valid."""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from agentbisect.context.constraints import merge_same_role, validate
from agentbisect.context.units import (
    PLACEHOLDER,
    all_ids,
    decompose,
    effective_keep,
    recompose,
    split_chunks,
    split_sections,
)
from agentbisect.providers.base import ContentBlock, LLMRequest, Message, ToolSpec

WORDS = [
    "agent",
    "deletes",
    "the",
    "project",
    "policy",
    "refund",
    "within",
    "30",
    "days",
    "README",
    "config",
    "notes",
    "sync",
    "server",
    "timezone",
    "UTC",
    "file",
    "tmp",
]


@st.composite
def paragraph(draw: Any, max_words: int = 40) -> str:
    return " ".join(draw(st.lists(st.sampled_from(WORDS), min_size=1, max_size=max_words)))


@st.composite
def body(draw: Any, long: bool = False) -> str:
    n = draw(st.integers(8, 14)) if long else draw(st.integers(1, 3))
    sep = draw(st.sampled_from(["\n\n", "\n\n\n", "\n"]))
    return sep.join(draw(st.lists(paragraph(), min_size=n, max_size=n)))


@st.composite
def system_prompt(draw: Any) -> str | None:
    style = draw(st.sampled_from(["none", "plain", "markdown", "xml"]))
    if style == "none":
        return None
    paras = draw(st.lists(paragraph(), min_size=1, max_size=4))
    if style == "markdown":
        return "\n\n".join(f"## Part {i}\n{p}" for i, p in enumerate(paras))
    if style == "xml":
        return "\n".join(f"<rule{i}>\n{p}\n</rule{i}>" for i, p in enumerate(paras))
    return "\n\n".join(paras)


@st.composite
def requests(draw: Any) -> LLMRequest:
    counter = iter(range(10_000))
    messages = [Message(role="user", content=[ContentBlock(type="text", text=draw(body()))])]
    for _ in range(draw(st.integers(0, 3))):
        ids = [f"tu_{next(counter)}" for _ in range(draw(st.integers(1, 3)))]
        assistant: list[ContentBlock] = []
        if draw(st.booleans()):
            assistant.append(ContentBlock(type="text", text=draw(paragraph())))
        assistant += [
            ContentBlock(type="tool_use", tool_use_id=i, tool_name="read", tool_input={"n": i})
            for i in ids
        ]
        messages.append(Message(role="assistant", content=assistant))
        user = [
            ContentBlock(
                type="tool_result",
                tool_use_id=i,
                text=draw(body(long=draw(st.booleans()))),
                is_error=draw(st.sampled_from([None, False, True])),
            )
            for i in ids
        ]
        if draw(st.booleans()):
            user.append(ContentBlock(type="text", text=draw(paragraph())))
        if draw(st.booleans()):
            user.append(ContentBlock(type="image", raw={"type": "image", "source": {"data": "AA"}}))
        messages.append(Message(role="user", content=user))
    if draw(st.booleans()):
        messages.append(
            Message(role="assistant", content=[ContentBlock(type="text", text=draw(paragraph()))])
        )
        messages.append(
            Message(role="user", content=[ContentBlock(type="text", text=draw(paragraph()))])
        )
    tools = [
        ToolSpec(name=f"tool{i}", description=draw(paragraph(12)), input_schema={})
        for i in range(draw(st.integers(0, 3)))
    ]
    return LLMRequest(
        model="fake-model",
        system=draw(system_prompt()),
        messages=messages,
        tools=tools,
        max_tokens=256,
        temperature=draw(st.sampled_from([None, 0.0])),
    )


MODES = st.sampled_from(["placeholder", "delete"])


@settings(max_examples=200)
@given(requests(), MODES)
def test_recompose_with_all_ids_is_identity(req: LLMRequest, mode: str) -> None:
    assert validate(req) == []  # the generator only makes valid requests
    units = decompose(req, mode)  # type: ignore[arg-type]
    assert len({u.id for u in units}) == len(units)
    assert recompose(req, all_ids(units), mode) == req  # type: ignore[arg-type]


@settings(max_examples=300)
@given(requests(), MODES, st.data())
def test_every_recomposition_is_valid(req: LLMRequest, mode: str, data: st.DataObject) -> None:
    units = decompose(req, mode)  # type: ignore[arg-type]
    keep = data.draw(st.sets(st.sampled_from([u.id for u in units])) if units else st.just(set()))
    out = recompose(req, keep, mode)  # type: ignore[arg-type]
    assert validate(out) == [], (keep, out)
    assert recompose(req, keep, mode) == out  # deterministic  # type: ignore[arg-type]
    first_task = req.messages[0].content[0].text
    assert out.messages[0].content[0].text == first_task  # the task always survives


@given(requests())
def test_protected_messages_and_kinds(req: LLMRequest) -> None:
    units = decompose(req)
    by_id = {u.id: u for u in units}
    assert not by_id["msg.0"].removable
    last_text_user = max(
        i
        for i, m in enumerate(req.messages)
        if m.role == "user" and any(b.type == "text" for b in m.content)
    )
    assert not by_id[f"msg.{last_text_user}"].removable
    for u in units:
        if u.kind == "chunk":
            assert u.parent in by_id and by_id[u.parent].kind == "tool_result"
        assert len(u.text) <= 300


@given(st.text(alphabet="ab \n#<>/x", max_size=400))
def test_splitters_are_exact_partitions(text: str) -> None:
    assert "".join(split_sections(text)) == text
    assert "".join(split_chunks(text)) == text
    assert all(split_sections(text)) and all(split_chunks(text))


def test_section_styles() -> None:
    assert split_sections("# A\nx\n## B\ny") == ["# A\nx\n", "## B\ny"]
    assert split_sections("<a>\nx\n</a>\n<b>\ny\n</b>") == ["<a>\nx\n</a>\n", "<b>\ny\n</b>"]
    assert split_sections("one\n\ntwo\n\n\nthree") == ["one\n\n", "two\n\n\n", "three"]


def test_chunks_are_about_200_tokens() -> None:
    text = "\n\n".join(f"paragraph {i} " + "word " * 60 for i in range(12))
    chunks = split_chunks(text)
    assert len(chunks) > 1 and all(len(c) <= 800 for c in chunks)


def tool_request() -> LLMRequest:
    long_text = "\n\n".join(f"para {i} " + "data " * 80 for i in range(6))
    return LLMRequest(
        model="m",
        system="Rule one.\n\nRule two.",
        max_tokens=10,
        messages=[
            Message(role="user", content=[ContentBlock(type="text", text="task")]),
            Message(
                role="assistant",
                content=[
                    ContentBlock(type="tool_use", tool_use_id="a", tool_name="f", tool_input={}),
                    ContentBlock(type="tool_use", tool_use_id="b", tool_name="f", tool_input={}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ContentBlock(type="tool_result", tool_use_id="a", text=long_text),
                    ContentBlock(type="tool_result", tool_use_id="b", text="short"),
                ],
            ),
            Message(role="assistant", content=[ContentBlock(type="text", text="ok")]),
            Message(role="user", content=[ContentBlock(type="text", text="now delete it")]),
        ],
        tools=[ToolSpec(name="f", description="Does f.", input_schema={})],
    )


def test_placeholder_mode_semantics() -> None:
    req = tool_request()
    units = {u.id: u for u in decompose(req)}
    assert not units["msg.1"].removable  # calls tools; its results stay as placeholders
    assert units["msg.3"].removable and not units["msg.4"].removable
    assert [u for u in units if u.startswith("msg.2.tr.0.chunk.")]
    out = recompose(req, {"msg.2.tr.0", "msg.2.tr.0.chunk.1"})
    assert out.system is None and out.tools[0].description == "" and out.tools[0].name == "f"
    result_a, result_b, _task = out.messages[2].content
    assert result_a.text and "para 1" in result_a.text and "para 0" not in result_a.text
    assert result_b.text == PLACEHOLDER and result_b.tool_use_id == "b"
    # msg.3 removed: the two user messages merge, tool results first
    assert [m.role for m in out.messages] == ["user", "assistant", "user"]
    assert [b.type for b in out.messages[2].content] == ["tool_result", "tool_result", "text"]
    assert validate(out) == []


def test_delete_mode_removes_tool_call_and_result_together() -> None:
    req = tool_request()
    units = {u.id: u for u in decompose(req, "delete")}
    assert units["msg.1"].group == units["msg.2.tr.0"].group == units["msg.2.tr.1"].group
    everything = set(units)
    half_group = recompose(req, everything - {"msg.2.tr.1"}, "delete")  # a group is never split
    assert [m.role for m in half_group.messages] == ["user", "assistant", "user"]
    assert all(b.type == "text" for m in half_group.messages for b in m.content)
    assert validate(half_group) == []
    alive = effective_keep(list(units.values()), everything - {"msg.2.tr.1"})
    assert "msg.1" not in alive and "msg.2.tr.0.chunk.0" not in alive


def test_merge_same_role() -> None:
    text = ContentBlock(type="text", text="hi")
    result = ContentBlock(type="tool_result", tool_use_id="x", text="r")
    merged = merge_same_role(
        [Message(role="user", content=[text]), Message(role="user", content=[result])]
    )
    assert merged == [Message(role="user", content=[result, text])]
