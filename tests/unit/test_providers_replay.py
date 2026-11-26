from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentbisect.providers.anthropic import AnthropicProvider
from agentbisect.providers.base import (
    LLMRequest,
    Message,
    ProviderError,
    ToolSpec,
    text_block,
)
from agentbisect.providers.fake import FakeLLM, Rule, text, tool_use
from agentbisect.providers.fake_http import fake_anthropic_client, fake_openai_client
from agentbisect.providers.openai import OpenAIProvider
from agentbisect.providers.pricing import UNKNOWN_PRICE, estimate_cost, price_for
from agentbisect.providers.registry import ProviderSetupError, make_provider
from agentbisect.replay.decision import (
    NotAnLLMStep,
    RequestOverrides,
    build_request,
    find_step,
    replay_decision,
)
from agentbisect.schema import Run, Step, request_hash

RULES = [Rule(when_all=["weather"], respond=tool_use("get_weather", {"city": "Paris"}))]
REQ = LLMRequest(
    model="fake-model",
    system="Be brief.",
    max_tokens=100,
    messages=[Message(role="user", content=[text_block("weather?")])],
    tools=[ToolSpec(name="get_weather", description="W", input_schema={})],
)


async def test_anthropic_provider_round_trips_through_the_real_sdk() -> None:
    client = fake_anthropic_client(FakeLLM(rules=RULES), async_client=True)
    provider = AnthropicProvider(client, prices={"fake-model": (3.0, 15.0)})
    resp = await provider.complete(REQ, seed=0)
    assert resp.tool_calls()[0].tool_input == {"city": "Paris"}
    assert provider.cost(resp, REQ) == pytest.approx(
        (resp.tokens_in * 3.0 + resp.tokens_out * 15.0) / 1e6
    )  # type: ignore[operator]


async def test_openai_provider_round_trips_through_the_real_sdk() -> None:
    client = fake_openai_client(FakeLLM(rules=RULES), async_client=True)
    resp = await OpenAIProvider(client).complete(REQ.model_copy(update={"model": "gpt-x"}))
    assert resp.tool_calls()[0].tool_name == "get_weather"


async def test_server_errors_become_retryable_provider_errors() -> None:
    client = fake_anthropic_client(FakeLLM(script=[]), async_client=True)  # -> HTTP 500
    with pytest.raises(ProviderError) as info:
        await AnthropicProvider(client).complete(REQ)
    assert info.value.retryable and info.value.status_code == 500


def test_pricing() -> None:
    assert price_for("fake") == ((1.0, 5.0), True)
    assert price_for("vendor-x-2026", {"vendor-x": (2.0, 8.0)}) == ((2.0, 8.0), True)
    assert price_for("mystery-model") == (UNKNOWN_PRICE, False)
    assert estimate_cost(REQ, (1.0, 5.0)) > 100 * 5.0 / 1e6  # charges the full max_tokens


def test_make_provider() -> None:
    fake = make_provider("fake", fake_rules="examples/file_cleanup_agent/fake_rules.py")
    assert isinstance(fake, FakeLLM) and fake.rules and fake.default.text() == "Cleanup complete."
    assert make_provider("anthropic").name == "anthropic"
    with pytest.raises(ProviderSetupError, match="--fake-rules"):
        make_provider("fake")
    with pytest.raises(ProviderSetupError, match="unknown provider"):
        make_provider("cohere")


def llm_step(req: LLMRequest) -> Step:
    now = datetime.now(UTC)
    d = req.to_dict()
    return Step(
        id="s0003",
        run_id="r",
        index=3,
        kind="llm_call",
        name=req.model,
        provider="anthropic",
        request=d,
        response=None,
        raw_request=None,
        raw_response=None,
        started_at=now,
        ended_at=now,
        request_hash=request_hash(d),
    )


async def test_replay_decision_with_overrides() -> None:
    step = llm_step(REQ)
    assert build_request(step) == REQ
    new_tools = [ToolSpec(name="get_weather", description="Weather by city", input_schema={})]
    over = RequestOverrides(model="other", temperature=0.0, system="New prompt", tools=new_tools)
    built = build_request(step, overrides=over)
    assert (built.model, built.temperature, built.system) == ("other", 0.0, "New prompt")
    assert built.tools == new_tools and built.messages == REQ.messages
    assert build_request(step) == REQ  # the recorded step is untouched
    resp = await replay_decision(step, provider=FakeLLM(rules=RULES), seed=0)
    assert resp.tool_calls()[0].tool_name == "get_weather"


def test_find_step_and_non_llm_steps() -> None:
    now = datetime.now(UTC)
    tool = Step(
        id="s0004",
        run_id="r",
        index=4,
        kind="tool_call",
        name="t",
        request={},
        response=None,
        raw_request=None,
        raw_response=None,
        started_at=now,
        ended_at=now,
        request_hash="h",
    )
    run = Run(
        id="r",
        name="n",
        entrypoint=None,
        input={},
        output=None,
        status="ok",
        steps=[llm_step(REQ), tool],
        started_at=now,
        ended_at=now,
        env={},
    )
    for ref in ("s0003", "3", "s3"):
        assert find_step(run, ref).id == "s0003"
    with pytest.raises(KeyError):
        find_step(run, "9")
    with pytest.raises(NotAnLLMStep):
        build_request(tool)
    assert text("x").text() == "x"
