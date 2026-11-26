"""Native <-> normalized translation for Anthropic and OpenAI."""

from __future__ import annotations

from agentbisect.providers import anthropic as anth
from agentbisect.providers import openai as oai
from agentbisect.providers.base import LLMResponse
from agentbisect.providers.fake import text, tool_use

ANTHROPIC_KWARGS = {
    "model": "fake-model",
    "max_tokens": 512,
    "system": [{"type": "text", "text": "Be brief."}, {"type": "text", "text": "Be kind."}],
    "messages": [
        {"role": "user", "content": "Read the README"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "R.md"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": [{"type": "text", "text": "# Title"}],
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AA"},
                },
            ],
        },
    ],
    "tools": [
        {"name": "read_file", "description": "Read", "input_schema": {"type": "object"}},
        {"type": "web_search_20250305", "name": "web_search"},
    ],
    "thinking": {"type": "enabled", "budget_tokens": 1024},
    "extra_headers": {"x-api-key": "sk-ant-should-never-be-recorded"},
    "stream": True,
}


def test_anthropic_request_normalizes_and_drops_transport_options() -> None:
    req = anth.request_from_native(ANTHROPIC_KWARGS)
    assert req.system == "Be brief.\n\nBe kind."
    assert [m.role for m in req.messages] == ["user", "assistant", "user"]
    use = req.messages[1].content[1]
    assert (use.type, use.tool_use_id, use.tool_name, use.tool_input) == (
        "tool_use",
        "tu_1",
        "read_file",
        {"path": "R.md"},
    )
    result = req.messages[2].content[0]
    assert (result.type, result.tool_use_id, result.text) == ("tool_result", "tu_1", "# Title")
    assert req.messages[2].content[1].type == "image"
    assert [t.name for t in req.tools] == ["read_file"]
    assert req.extra["server_tools"][0]["name"] == "web_search"
    assert req.extra["thinking"]["budget_tokens"] == 1024
    assert "extra_headers" not in req.extra and "stream" not in req.extra
    assert "extra_headers" not in anth.clean_kwargs(ANTHROPIC_KWARGS)


def test_anthropic_request_round_trips_through_native() -> None:
    req = anth.request_from_native(ANTHROPIC_KWARGS)
    assert anth.request_from_native(anth.request_to_native(req)) == req


def test_anthropic_response_round_trip() -> None:
    resp = LLMResponse(
        content=[*text("hi").content, *tool_use("f", {"a": 1}).content],
        stop_reason="tool_use",
        tokens_in=7,
        tokens_out=3,
    )
    native = anth.response_to_native(resp, model="m", message_id="msg_1")
    assert anth.response_from_native(native) == resp


def test_anthropic_stream_aggregation_rebuilds_message() -> None:
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "m1",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [],
                "usage": {"input_tokens": 9, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "He"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "y"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"a": '},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "1}"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 4},
        },
        {"type": "message_stop"},
    ]
    agg = anth.StreamAggregator()
    for e in events:
        agg.feed(e)
    resp = anth.response_from_native(agg.result())
    assert resp.text() == "Hey"
    assert resp.tool_calls()[0].tool_input == {"a": 1}
    assert (resp.stop_reason, resp.tokens_in, resp.tokens_out) == ("tool_use", 9, 4)


CHAT_KWARGS = {
    "model": "gpt-x",
    "max_completion_tokens": 300,
    "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "tool", "tool_call_id": "c2", "content": "rain"},
        {"role": "user", "content": "thanks"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "W",
                "parameters": {"type": "object"},
            },
        }
    ],
    "temperature": 0.2,
}


def test_chat_request_merges_tool_messages_into_one_user_turn() -> None:
    req = oai.chat_request_from_native(CHAT_KWARGS)
    assert req.system == "Be brief."
    assert [m.role for m in req.messages] == ["user", "assistant", "user"]
    last = req.messages[2].content
    assert [(b.type, b.tool_use_id) for b in last] == [
        ("tool_result", "c1"),
        ("tool_result", "c2"),
        ("text", None),
    ]
    assert req.max_tokens == 300 and req.extra["max_tokens_param"] == "max_completion_tokens"
    assert req.temperature == 0.2


def test_chat_request_round_trips_through_native() -> None:
    req = oai.chat_request_from_native(CHAT_KWARGS)
    assert oai.chat_request_from_native(oai.chat_request_to_native(req)) == req


def test_chat_request_without_limit_uses_default() -> None:
    req = oai.chat_request_from_native(
        {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    )
    assert req.max_tokens == oai.DEFAULT_MAX_TOKENS and req.extra["max_tokens_param"] is None
    assert "max_tokens" not in oai.chat_request_to_native(req)


def test_chat_and_responses_response_round_trips() -> None:
    resp = LLMResponse(
        content=[*text("ok").content, *tool_use("f", {"x": [1, 2]}).content],
        stop_reason=None,
        tokens_in=5,
        tokens_out=2,
    )
    chat = oai.chat_response_from_native(
        oai.chat_response_to_native(resp, model="m", completion_id="c")
    )
    assert chat.content == resp.content and chat.stop_reason == "tool_calls"
    responses = oai.responses_response_from_native(
        oai.responses_response_to_native(resp, model="m", response_id="r")
    )
    assert responses.content == resp.content and responses.stop_reason == "tool_use"
    assert (responses.tokens_in, responses.tokens_out) == (5, 2)


def test_responses_request_items() -> None:
    req = oai.responses_request_from_native(
        {
            "model": "gpt-x",
            "instructions": "Be brief.",
            "max_output_tokens": 99,
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
                {"type": "function_call", "call_id": "c1", "name": "w", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            ],
            "tools": [{"type": "function", "name": "w", "description": "W", "parameters": {}}],
        }
    )
    assert req.system == "Be brief." and req.max_tokens == 99
    assert [(m.role, [b.type for b in m.content]) for m in req.messages] == [
        ("user", ["text"]),
        ("assistant", ["tool_use"]),
        ("user", ["tool_result"]),
    ]
    assert req.messages[2].content[0].text == "sunny"


def test_tool_call_extras_like_gemini_thought_signatures_survive_round_trips() -> None:
    """Gemini 3 rejects a later turn whose earlier tool calls lost their thought signature, so
    extras on a tool call must come back unchanged when a request or response is rebuilt."""
    from agentbisect.providers.openai import (
        ChatStreamAggregator,
        chat_request_from_native,
        chat_request_to_native,
        chat_response_from_native,
        chat_response_to_native,
    )

    signature = {"google": {"thought_signature": "c2lnbmF0dXJl"}}
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "/p/README.md"}'},
        "extra_content": signature,
    }
    native = {
        "model": "gemini-3-flash-preview",
        "max_completion_tokens": 100,
        "messages": [
            {"role": "user", "content": "clean up"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "content": "README text"},
        ],
    }
    rebuilt = chat_request_to_native(chat_request_from_native(native))
    assert rebuilt["messages"][1]["tool_calls"][0]["extra_content"] == signature

    completion = {
        "id": "c1",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None, "tool_calls": [call]},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    served = chat_response_to_native(
        chat_response_from_native(completion), model="m", completion_id="c1"
    )
    assert served["choices"][0]["message"]["tool_calls"][0]["extra_content"] == signature

    agg = ChatStreamAggregator()
    agg.feed(
        {"id": "c1", "model": "m", "choices": [{"delta": {"tool_calls": [{**call, "index": 0}]}}]}
    )
    agg.feed({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    assert agg.result()["choices"][0]["message"]["tool_calls"][0]["extra_content"] == signature
