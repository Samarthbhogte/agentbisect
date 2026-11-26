from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx2
import openai
import pytest

from agentbisect.providers.anthropic import AnthropicProvider, _provider_error
from agentbisect.providers.base import (
    LLMRequest,
    Message,
    ProviderError,
    ToolSpec,
    text_block,
)
from agentbisect.providers.fake import FakeLLM, Rule, text, tool_use
from agentbisect.providers.fake_http import fake_anthropic_client, fake_openai_client
from agentbisect.providers.gemini import BASE_URL, GeminiProvider
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


def test_make_gemini_provider_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderSetupError, match="GEMINI_API_KEY"):
        make_provider("gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = make_provider("gemini")
    assert isinstance(provider, GeminiProvider)
    assert (provider.name, provider.cache_namespace) == ("gemini", "gemini")
    assert str(provider.client.base_url) == BASE_URL


async def test_gemini_provider_never_sends_seed() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: Any) -> Any:
        sent.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "gemini-2.5-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url=BASE_URL,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    req = REQ.model_copy(update={"model": "gemini-2.5-flash"})
    resp = await GeminiProvider(client).complete(req, seed=7)
    assert resp.text() == "ok" and "seed" not in sent[0]
    await OpenAIProvider(client).complete(req, seed=7)
    assert sent[1]["seed"] == 7  # plain OpenAI still sends it


def test_rate_limit_wait_comes_from_header_or_google_retry_delay() -> None:
    def rate_limited(headers: dict[str, str], body: dict[str, Any]) -> openai.RateLimitError:
        request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
        response = httpx2.Response(429, headers=headers, json=body, request=request)
        return openai.RateLimitError("rate limited", response=response, body=body)

    google_body = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}],
        }
    }
    from_header = _provider_error(rate_limited({"retry-after": "12"}, {}))
    from_body = _provider_error(rate_limited({}, google_body))
    neither = _provider_error(rate_limited({}, {}))
    assert isinstance(from_header, ProviderError) and from_header.retry_after == 12.0
    assert isinstance(from_body, ProviderError) and from_body.retry_after == 37.0
    assert isinstance(neither, ProviderError) and neither.retry_after is None
    assert from_body.retryable and from_body.status_code == 429


def test_gemini_prices() -> None:
    assert price_for("gemini-2.5-flash") == ((0.30, 2.50), True)
    assert price_for("gemini-2.5-flash-lite") == ((0.10, 0.40), True)  # longest prefix wins


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
