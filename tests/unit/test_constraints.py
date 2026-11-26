from __future__ import annotations

import pytest

from agentbisect.context.constraints import InvalidRequest, assert_valid, validate
from agentbisect.providers.base import ContentBlock, LLMRequest, Message


def req(*messages: Message) -> LLMRequest:
    return LLMRequest(model="m", system=None, messages=list(messages), max_tokens=10)


def user(*blocks: ContentBlock) -> Message:
    return Message(role="user", content=list(blocks))


def assistant(*blocks: ContentBlock) -> Message:
    return Message(role="assistant", content=list(blocks))


TEXT = ContentBlock(type="text", text="hi")


def use(i: str) -> ContentBlock:
    return ContentBlock(type="tool_use", tool_use_id=i, tool_name="f", tool_input={})


def result(i: str) -> ContentBlock:
    return ContentBlock(type="tool_result", tool_use_id=i, text="r")


@pytest.mark.parametrize(
    ("request_", "problem"),
    [
        (req(), "no messages"),
        (req(assistant(TEXT)), "start with a user"),
        (req(user(TEXT), user(TEXT)), "roles must alternate"),
        (req(user(TEXT), assistant(use("a")), user(TEXT)), "not answered"),
        (req(user(TEXT), assistant(TEXT), user(result("a"))), "has no tool_use"),
        (req(user(TEXT), assistant(use("a")), user(TEXT, result("a"))), "must come before"),
        (req(user(TEXT), assistant(use("a"), use("a")), user(result("a"))), "more than once"),
        (req(user(TEXT), assistant()), "is empty"),
    ],
)
def test_validate_finds_each_problem(request_: LLMRequest, problem: str) -> None:
    problems = validate(request_)
    assert any(problem in p for p in problems), problems
    with pytest.raises(InvalidRequest):
        assert_valid(request_)


def test_valid_conversation() -> None:
    good = req(
        user(TEXT),
        assistant(TEXT, use("a"), use("b")),
        user(result("a"), result("b"), TEXT),
        assistant(TEXT),
    )
    assert validate(good) == []
    assert_valid(good)
