"""The capture SDK against real Anthropic/OpenAI SDK clients on a mocked HTTP transport."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

import agentrelay as relay
from agentrelay.providers.fake import FakeLLM, Rule, text, tool_use
from agentrelay.providers.fake_http import fake_anthropic_client, fake_openai_client
from agentrelay.schema import Run
from agentrelay.sdk.tools import tool_info
from agentrelay.store import RunStore

RULES = [
    Rule(
        when_all=["weather"],
        unless_any=["sunny"],
        respond=tool_use("get_weather", {"city": "Paris"}),
    )
]
DEFAULT = text("It is sunny in Paris.")
MSGS = [{"role": "user", "content": "weather in Paris?"}]


def llm() -> FakeLLM:
    return FakeLLM(rules=RULES, default=DEFAULT)


def last_run() -> Run:
    return RunStore().load("last")


def llm_steps(run: Run) -> list[Any]:
    return [s for s in run.steps if s.kind == "llm_call"]


# ---- Anthropic ----------------------------------------------------------------------------
def test_anthropic_sync_records_request_and_response() -> None:
    client = relay.wrap(fake_anthropic_client(llm()))
    with relay.run("t"):
        msg = client.messages.create(
            model="fake-model",
            max_tokens=64,
            messages=MSGS,
            extra_headers={"x-trace": "abc"},
        )
    assert msg.content[0].name == "get_weather"  # caller gets the real SDK object
    (step,) = last_run().steps
    assert (step.kind, step.name, step.provider) == ("llm_call", "fake-model", "anthropic")
    assert step.request["messages"][0]["content"][0]["text"] == "weather in Paris?"
    assert step.response is not None
    assert step.response["content"][0]["tool_name"] == "get_weather"
    assert step.raw_request is not None and "extra_headers" not in step.raw_request
    assert step.tokens_in and step.tokens_out


async def test_anthropic_async_records() -> None:
    client = relay.wrap(fake_anthropic_client(llm(), async_client=True))
    async with relay.run("t"):
        msg = await client.messages.create(model="m", max_tokens=64, messages=MSGS)
    assert msg.content[0].type == "tool_use"
    assert last_run().steps[0].response["content"][0]["tool_input"] == {"city": "Paris"}  # type: ignore[index]


def test_anthropic_stream_passes_events_through_and_records_final_message() -> None:
    client = relay.wrap(fake_anthropic_client(FakeLLM(default=text("streamed hello world"))))
    stream = client.messages.create
    with relay.run("t"), stream(model="m", max_tokens=64, messages=MSGS, stream=True) as s:
        events = list(s)
    assert events[0].type == "message_start" and events[-1].type == "message_stop"
    text_deltas = "".join(e.delta.text for e in events if e.type == "content_block_delta")
    assert text_deltas == "streamed hello world"
    step = last_run().steps[0]
    assert step.response is not None
    assert step.response["content"][0]["text"] == "streamed hello world"
    assert step.tokens_out and step.tokens_out > 0
    assert step.raw_request is not None and "stream" not in step.raw_request


async def test_anthropic_async_stream_records_tool_use() -> None:
    client = relay.wrap(fake_anthropic_client(llm(), async_client=True))
    async with relay.run("t"):
        stream = await client.messages.create(model="m", max_tokens=64, messages=MSGS, stream=True)
        events = [e async for e in stream]
    assert events[-1].type == "message_stop"
    resp = last_run().steps[0].response
    assert resp is not None and resp["content"][0]["tool_input"] == {"city": "Paris"}


def test_non_intercepted_attributes_pass_through() -> None:
    raw = fake_anthropic_client(llm())
    client = relay.wrap(raw)
    assert client.api_key == raw.api_key
    assert client.messages.count_tokens == raw.messages.count_tokens
    assert "agentrelay.wrap" in repr(client)


def test_api_errors_are_recorded_and_re_raised() -> None:
    import anthropic

    client = relay.wrap(fake_anthropic_client(FakeLLM(script=[])))  # 0 responses -> 500
    with relay.run("t"), pytest.raises(anthropic.InternalServerError):
        client.messages.create(model="m", max_tokens=8, messages=MSGS)
    step = last_run().steps[0]
    assert step.error is not None and step.error.type == "InternalServerError"
    assert step.response is None


# ---- OpenAI -------------------------------------------------------------------------------
def test_openai_chat_sync_and_stream() -> None:
    client = relay.wrap(fake_openai_client(llm()))
    with relay.run("t"):
        r = client.chat.completions.create(model="gpt-x", messages=MSGS)
        chunks = list(
            client.chat.completions.create(
                model="gpt-x", stream=True, messages=[{"role": "user", "content": "hello"}]
            )
        )
    assert r.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert chunks[0].choices[0].delta.role == "assistant"
    a, b = last_run().steps
    assert a.provider == b.provider == "openai"
    assert a.response is not None and a.response["content"][0]["tool_name"] == "get_weather"
    assert b.response is not None and b.response["content"][0]["text"] == DEFAULT.text()
    assert b.tokens_in and b.tokens_out


async def test_openai_async_chat_and_responses() -> None:
    client = relay.wrap(fake_openai_client(llm(), async_client=True))
    async with relay.run("t"):
        await client.chat.completions.create(model="gpt-x", messages=MSGS)
        await client.responses.create(model="gpt-x", input="weather in Paris?")
    a, b = last_run().steps
    assert a.response["content"][0]["tool_name"] == "get_weather"  # type: ignore[index]
    assert b.response["content"][0]["tool_input"] == {"city": "Paris"}  # type: ignore[index]
    assert b.request["messages"][0]["content"][0]["text"] == "weather in Paris?"


def test_openai_responses_stream() -> None:
    client = relay.wrap(fake_openai_client(FakeLLM(default=text("from responses"))))
    with relay.run("t"):
        events = list(client.responses.create(model="gpt-x", input="hi", stream=True))
    assert [e.type for e in events] == ["response.created", "response.completed"]
    resp = last_run().steps[0].response
    assert resp is not None and resp["content"][0]["text"] == "from responses"


def test_wrap_rejects_unknown_clients() -> None:
    with pytest.raises(TypeError):
        relay.wrap(object())


# ---- tools, runs, dependencies ------------------------------------------------------------
@relay.tool
def get_weather(city: str) -> str:
    return f"{city}: sunny, 21C"


@relay.tool(name="explode", side_effects=True)
def broken(x: int) -> str:
    raise ValueError(f"bad {x}")


@relay.tool
async def aget(city: str) -> str:
    return f"{city}: cloudy"


def agent_loop(client: Any) -> str:
    msgs: list[dict[str, Any]] = list(MSGS)
    while True:
        r = client.messages.create(model="m", max_tokens=64, messages=msgs)
        msgs.append({"role": "assistant", "content": r.content})
        uses = [b for b in r.content if b.type == "tool_use"]
        if not uses:
            return str(r.content[0].text)
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": get_weather(**b.input)}
                    for b in uses
                ],
            }
        )


def test_tool_steps_link_to_llm_steps_both_ways() -> None:
    client = relay.wrap(fake_anthropic_client(llm()))
    with relay.run("t"):
        assert agent_loop(client) == DEFAULT.text()
    s0, s1, s2 = last_run().steps
    assert [s.kind for s in (s0, s1, s2)] == ["llm_call", "tool_call", "llm_call"]
    assert s1.request == {"city": "Paris"}
    assert s1.response == {"result": "Paris: sunny, 21C"}
    assert s1.tags["tool_use_id"] == s0.response["content"][0]["tool_use_id"]  # type: ignore[index]
    assert s1.depends_on == [s0.id]
    assert s2.depends_on == [s1.id]


def test_verbatim_fallback_links_tool_output_without_tool_use_id() -> None:
    client = relay.wrap(fake_anthropic_client(FakeLLM(default=text("ok"))))
    with relay.run("t"):
        report = get_weather("Lisbon")  # called by plain code, not via a tool_use block
        client.messages.create(
            model="m", max_tokens=8, messages=[{"role": "user", "content": f"Summarize: {report}"}]
        )
    tool_step, llm_step = last_run().steps
    assert "tool_use_id" not in tool_step.tags and tool_step.depends_on == []
    assert llm_step.depends_on == [tool_step.id]


async def test_tool_errors_are_recorded_and_re_raised_and_async_tools_work() -> None:
    async with relay.run("t"):
        with pytest.raises(ValueError, match="bad 3"):
            broken(3)
        assert await aget("Oslo") == "Oslo: cloudy"
    s0, s1 = last_run().steps
    assert (s0.name, s0.error and s0.error.message, s0.response) == ("explode", "bad 3", None)
    assert s0.tags["side_effects"] == "true"
    assert (s1.name, s1.response) == ("aget", {"result": "Oslo: cloudy"})
    info = tool_info(broken)
    assert info is not None and info.side_effects and info.name == "explode"


def test_run_status_output_and_error() -> None:
    with relay.run("good", input={"q": 1}, labels={"team": "x"}) as rec:
        rec.set_output("fine")
    good = last_run()
    assert (good.status, good.output, good.input, good.labels) == (
        "ok",
        {"return": "fine"},
        {"q": 1},
        {"team": "x"},
    )
    assert good.ended_at is not None and good.env["agentrelay"] == relay.__version__

    with pytest.raises(RuntimeError), relay.run("bad"):
        raise RuntimeError("boom")
    bad = last_run()
    assert bad.status == "error"
    assert bad.output == {"error": {"type": "RuntimeError", "message": "boom"}}


@relay.entrypoint
def entry(task: str) -> str:
    with relay.run("entry-run", input={"task": task}):
        get_weather("Rome")
    return f"done {task}"


def test_entrypoint_records_path_and_return_value() -> None:
    assert entry("x") == "done x"
    run = last_run()
    assert run.entrypoint == f"{__name__}:entry"
    assert run.output == {"return": "done x"}


def test_calls_outside_a_run_go_to_an_implicit_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentrelay.sdk.context import reset_state

    monkeypatch.setenv("AGENTRELAY_RUN_NAME", "my_script")
    get_weather("Paris")
    get_weather("Rome")
    reset_state()  # what atexit does at interpreter shutdown
    run = last_run()
    assert run.name == "my_script" and run.status == "unknown"
    assert [s.request["city"] for s in run.steps] == ["Paris", "Rome"]


def test_disabled_sdk_records_nothing(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> None:
    monkeypatch.setenv("AGENTRELAY_DISABLED", "1")
    raw = fake_anthropic_client(llm())
    assert relay.wrap(raw) is raw
    with relay.run("t") as rec:
        assert get_weather("Paris") == "Paris: sunny, 21C"
        rec.set_output("x")
    assert not isolated_home.exists()


def test_disk_errors_never_reach_user_code(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def disk_full(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    for method in ("start_run", "append_step", "finish_run", "save_run"):
        monkeypatch.setattr(RunStore, method, disk_full)
    client = relay.wrap(fake_anthropic_client(llm()))
    with caplog.at_level(logging.WARNING, logger="agentrelay"):
        with relay.run("t"):
            answer = agent_loop(client)
        assert entry("y") == "done y"
    assert answer == DEFAULT.text()
    assert "No space left on device" in caplog.text
