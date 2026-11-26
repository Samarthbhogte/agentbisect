"""Real SDK clients backed by FakeLLM through an in-memory HTTP transport.

``fake_anthropic_client(llm)`` returns a genuine ``anthropic.Anthropic`` whose HTTP layer is an
``httpx2.MockTransport`` speaking the Messages API (JSON and server-sent events). The same goes for
``fake_openai_client`` with Chat Completions and Responses. Nothing touches the network, and the
whole SDK code path, including streaming, runs exactly as it would against the real API.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Iterator
from typing import Any

from agentbisect.providers import anthropic as anth
from agentbisect.providers import openai as oai
from agentbisect.providers.base import LLMRequest, LLMResponse
from agentbisect.providers.fake import FakeLLM

FAKE_KEY = "fake-key-not-a-secret"


def _sse(events: Iterator[tuple[str | None, dict[str, Any] | str]]) -> bytes:
    out = []
    for name, data in events:
        payload = data if isinstance(data, str) else json.dumps(data)
        out.append((f"event: {name}\n" if name else "") + f"data: {payload}\n\n")
    return "".join(out).encode()


class _FakeServer:
    """Turns HTTP requests into FakeLLM calls and back."""

    def __init__(self, llm: FakeLLM, seed: int, id_prefix: str) -> None:
        self.llm = llm
        self.seed = seed
        self._calls = itertools.count(1)
        self._prefix = id_prefix

    def complete(self, req: LLMRequest) -> tuple[LLMResponse, int]:
        n = next(self._calls)
        resp = self.llm.complete_sync(req, seed=self.seed)
        for i, block in enumerate(resp.tool_calls()):
            block.tool_use_id = f"{self._prefix}_{n:03d}_{i}"
        return resp, n


def _server_error(exc: Exception) -> Any:
    import httpx2

    message = f"fake server: {type(exc).__name__}: {exc}"
    return httpx2.Response(
        500, json={"type": "error", "error": {"type": "api_error", "message": message}}
    )


def _anthropic_handler(server: _FakeServer) -> Callable[[Any], Any]:
    import httpx2

    def handle(request: Any) -> Any:
        body = json.loads(request.content or b"{}")
        if not request.url.path.endswith("/messages"):
            return httpx2.Response(404, json={"error": f"fake server: {request.url.path}"})
        try:
            resp, n = server.complete(anth.request_from_native(body))
        except Exception as exc:  # surface FakeLLM failures as an API error, like a real server
            return _server_error(exc)
        native = anth.response_to_native(resp, model=body["model"], message_id=f"msg_fake_{n:03d}")
        if body.get("stream"):
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse(_anthropic_events(native)),
            )
        return httpx2.Response(200, json=native)

    return handle


def _anthropic_events(msg: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    start = {
        **msg,
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": msg["usage"]["input_tokens"], "output_tokens": 0},
    }
    yield "message_start", {"type": "message_start", "message": start}
    for i, block in enumerate(msg["content"]):
        if block["type"] == "text":
            yield (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            text = block["text"]
            for j in range(0, max(len(text), 1), 8):
                yield (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "text_delta", "text": text[j : j + 8]},
                    },
                )
        else:
            yield (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {**block, "input": {}},
                },
            )
            raw = json.dumps(block["input"])
            for j in range(0, len(raw), 10):
                yield (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": raw[j : j + 10]},
                    },
                )
        yield "content_block_stop", {"type": "content_block_stop", "index": i}
    yield (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": msg["usage"]["output_tokens"]},
        },
    )
    yield "message_stop", {"type": "message_stop"}


def _openai_handler(server: _FakeServer) -> Callable[[Any], Any]:
    import httpx2

    def handle(request: Any) -> Any:
        try:
            return _handle(request)
        except Exception as exc:  # surface FakeLLM failures as an API error, like a real server
            return _server_error(exc)

    def _handle(request: Any) -> Any:
        body = json.loads(request.content or b"{}")
        path = request.url.path
        if path.endswith("/chat/completions"):
            resp, n = server.complete(oai.chat_request_from_native(body))
            native = oai.chat_response_to_native(
                resp, model=body["model"], completion_id=f"chatcmpl_fake_{n:03d}"
            )
            events = _chat_chunks(native)
        elif path.endswith("/responses"):
            resp, n = server.complete(oai.responses_request_from_native(body))
            native = oai.responses_response_to_native(
                resp, model=body["model"], response_id=f"resp_fake_{n:03d}"
            )
            events = _responses_events(native)
        else:
            return httpx2.Response(404, json={"error": f"fake server: {path}"})
        if body.get("stream"):
            return httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse(events)
            )
        return httpx2.Response(200, json=native)

    return handle


def _chat_chunks(c: dict[str, Any]) -> Iterator[tuple[str | None, dict[str, Any] | str]]:
    base = {"id": c["id"], "object": "chat.completion.chunk", "created": 0, "model": c["model"]}
    choice = c["choices"][0]
    msg = choice["message"]

    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield None, chunk({"role": "assistant", "content": ""})
    text = msg.get("content") or ""
    for j in range(0, len(text), 8):
        yield None, chunk({"content": text[j : j + 8]})
    for i, call in enumerate(msg.get("tool_calls") or []):
        fn = call["function"]
        yield (
            None,
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": fn["name"], "arguments": ""},
                        }
                    ]
                }
            ),
        )
        for j in range(0, len(fn["arguments"]), 10):
            yield (
                None,
                chunk(
                    {
                        "tool_calls": [
                            {"index": i, "function": {"arguments": fn["arguments"][j : j + 10]}}
                        ]
                    }
                ),
            )
    yield None, chunk({}, choice["finish_reason"])
    yield None, {**base, "choices": [], "usage": c["usage"]}
    yield None, "[DONE]"


def _responses_events(r: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    in_progress = {**r, "status": "in_progress", "output": []}
    yield (
        "response.created",
        {"type": "response.created", "response": in_progress, "sequence_number": 0},
    )
    yield "response.completed", {"type": "response.completed", "response": r, "sequence_number": 1}


def fake_anthropic_client(llm: FakeLLM, *, seed: int = 0, async_client: bool = False) -> Any:
    """An `anthropic.Anthropic` (or `AsyncAnthropic`) answered by `llm`."""
    import anthropic
    import httpx2

    transport = httpx2.MockTransport(_anthropic_handler(_FakeServer(llm, seed, "toolu_fake")))
    if async_client:
        return anthropic.AsyncAnthropic(
            api_key=FAKE_KEY, max_retries=0, http_client=httpx2.AsyncClient(transport=transport)
        )
    return anthropic.Anthropic(
        api_key=FAKE_KEY, max_retries=0, http_client=httpx2.Client(transport=transport)
    )


def fake_openai_client(
    llm: FakeLLM, *, seed: int = 0, async_client: bool = False, base_url: str | None = None
) -> Any:
    """An `openai.OpenAI` (or `AsyncOpenAI`) answered by `llm`. `base_url` only changes the URL
    the SDK reports (e.g. Gemini's compatible endpoint); requests still go to `llm`."""
    import httpx2
    import openai

    transport = httpx2.MockTransport(_openai_handler(_FakeServer(llm, seed, "call_fake")))
    if async_client:
        return openai.AsyncOpenAI(
            api_key=FAKE_KEY,
            base_url=base_url,
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=transport),
        )
    return openai.OpenAI(
        api_key=FAKE_KEY,
        base_url=base_url,
        max_retries=0,
        http_client=httpx2.Client(transport=transport),
    )
