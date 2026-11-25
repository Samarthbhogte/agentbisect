"""OpenAI Chat Completions and Responses APIs <-> normalized format.

Normalization notes:

* System and developer messages are joined into ``system``; their position in the conversation
  is not kept.
* Consecutive messages with the same role are merged (tool messages become ``tool_result``
  blocks in a user message), matching the alternating structure the rest of AgentRelay assumes.
* ``max_tokens`` defaults to 4096 when the request sets no limit; ``extra["max_tokens_param"]``
  records which parameter (if any) carried it so the native request can be rebuilt.
"""

from __future__ import annotations

import json
from typing import Any

from agentrelay._json import to_jsonable
from agentrelay.providers.base import ContentBlock, LLMRequest, LLMResponse, Message, ToolSpec

PROVIDER = "openai"
TRANSPORT_KWARGS = frozenset({"stream", "extra_headers", "extra_query", "extra_body", "timeout"})
DEFAULT_MAX_TOKENS = 4096


def clean_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = to_jsonable(
        {k: v for k, v in kwargs.items() if k not in TRANSPORT_KWARGS}
    )
    return out


def _parse_args(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"_raw_arguments": arguments}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and "text" in p)


def _parts(content: Any) -> list[ContentBlock]:
    """User/assistant content (string or parts) -> blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [ContentBlock(type="text", text=content)]
    blocks = []
    for p in content:
        kind = p.get("type")
        if kind in ("text", "input_text", "output_text"):
            blocks.append(ContentBlock(type="text", text=p.get("text", "")))
        elif kind in ("image_url", "input_image"):
            blocks.append(ContentBlock(type="image", raw=p))
        else:
            blocks.append(ContentBlock(type="other", raw=p))
    return blocks


def _append(messages: list[Message], role: str, blocks: list[ContentBlock]) -> None:
    if not blocks:
        return
    if messages and messages[-1].role == role:
        messages[-1].content.extend(blocks)
    else:
        messages.append(Message(role=role, content=blocks))


def _max_tokens(k: dict[str, Any], *names: str) -> tuple[int, str | None]:
    for name in names:
        if k.get(name) is not None:
            return int(k[name]), name
    return DEFAULT_MAX_TOKENS, None


# ---- Chat Completions ---------------------------------------------------------------------
_CHAT_CORE = frozenset(
    {"model", "messages", "tools", "temperature", "max_tokens", "max_completion_tokens"}
)


def chat_request_from_native(kwargs: dict[str, Any]) -> LLMRequest:
    k = clean_kwargs(kwargs)
    system: list[str] = []
    messages: list[Message] = []
    for m in k["messages"]:
        role = m["role"]
        if role in ("system", "developer"):
            system.append(_text_of(m.get("content")))
        elif role == "tool":
            block = ContentBlock(
                type="tool_result",
                tool_use_id=m.get("tool_call_id"),
                text=_text_of(m.get("content")),
            )
            _append(messages, "user", [block])
        elif role == "assistant":
            blocks = _parts(m.get("content"))
            for call in m.get("tool_calls") or []:
                fn = call.get("function", {})
                blocks.append(
                    ContentBlock(
                        type="tool_use",
                        tool_use_id=call.get("id"),
                        tool_name=fn.get("name"),
                        tool_input=_parse_args(fn.get("arguments")),
                    )
                )
            _append(messages, "assistant", blocks)
        else:
            _append(messages, "user", _parts(m.get("content")))
    tools, other_tools = [], []
    for t in k.get("tools") or []:
        if t.get("type") == "function":
            fn = t["function"]
            tools.append(
                ToolSpec(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    input_schema=fn.get("parameters", {}),
                )
            )
        else:
            other_tools.append(t)
    max_tokens, param = _max_tokens(k, "max_completion_tokens", "max_tokens")
    extra = {key: v for key, v in k.items() if key not in _CHAT_CORE}
    extra["max_tokens_param"] = param
    if other_tools:
        extra["other_tools"] = other_tools
    return LLMRequest(
        model=k["model"],
        system="\n\n".join(system) if system else None,
        messages=messages,
        tools=tools,
        temperature=k.get("temperature"),
        max_tokens=max_tokens,
        extra=extra,
    )


def chat_request_to_native(req: LLMRequest) -> dict[str, Any]:
    extra = dict(req.extra)
    param = extra.pop("max_tokens_param", "max_completion_tokens")
    other_tools = extra.pop("other_tools", [])
    msgs: list[dict[str, Any]] = []
    if req.system is not None:
        msgs.append({"role": "system", "content": req.system})
    for m in req.messages:
        msgs.extend(_chat_message_to_native(m))
    out: dict[str, Any] = {"model": req.model, "messages": msgs}
    tools = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in req.tools
    ]
    if tools or other_tools:
        out["tools"] = tools + other_tools
    if req.temperature is not None:
        out["temperature"] = req.temperature
    if param:
        out[param] = req.max_tokens
    out.update(extra)
    return out


def _chat_message_to_native(m: Message) -> list[dict[str, Any]]:
    if m.role == "assistant":
        text = "".join(b.text or "" for b in m.content if b.type == "text")
        out: dict[str, Any] = {"role": "assistant", "content": text or None}
        calls = [
            {
                "id": b.tool_use_id,
                "type": "function",
                "function": {"name": b.tool_name, "arguments": json.dumps(b.tool_input or {})},
            }
            for b in m.content
            if b.type == "tool_use"
        ]
        if calls:
            out["tool_calls"] = calls
        return [out]
    natives: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    for b in m.content:
        if b.type == "tool_result":
            natives.append({"role": "tool", "tool_call_id": b.tool_use_id, "content": b.text or ""})
        elif b.type == "text":
            parts.append({"type": "text", "text": b.text or ""})
        elif b.raw is not None:
            parts.append(b.raw)
    if parts:
        only_text = len(parts) == 1 and parts[0].get("type") == "text"
        natives.append({"role": "user", "content": parts[0]["text"] if only_text else parts})
    return natives


def chat_response_from_native(completion: Any) -> LLMResponse:
    c = to_jsonable(completion)
    choice = (c.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    blocks = _parts(msg.get("content"))
    for call in msg.get("tool_calls") or []:
        fn = call.get("function", {})
        blocks.append(
            ContentBlock(
                type="tool_use",
                tool_use_id=call.get("id"),
                tool_name=fn.get("name"),
                tool_input=_parse_args(fn.get("arguments")),
            )
        )
    usage = c.get("usage") or {}
    return LLMResponse(
        content=blocks,
        stop_reason=choice.get("finish_reason"),
        tokens_in=usage.get("prompt_tokens"),
        tokens_out=usage.get("completion_tokens"),
    )


def chat_response_to_native(resp: LLMResponse, *, model: str, completion_id: str) -> dict[str, Any]:
    text = resp.text()
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    calls = [
        {
            "id": b.tool_use_id,
            "type": "function",
            "function": {"name": b.tool_name, "arguments": json.dumps(b.tool_input or {})},
        }
        for b in resp.tool_calls()
    ]
    if calls:
        message["tool_calls"] = calls
    tin, tout = resp.tokens_in or 0, resp.tokens_out or 0
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}
        ],
        "usage": {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout},
    }


class ChatStreamAggregator:
    """Rebuilds a ChatCompletion from `stream=True` chunks."""

    def __init__(self) -> None:
        self.base: dict[str, Any] = {}
        self.content: list[str] = []
        self.calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None

    def feed(self, chunk: Any) -> None:
        c = to_jsonable(chunk)
        self.base = self.base or {k: c.get(k) for k in ("id", "model", "created")}
        if c.get("usage"):
            self.usage = c["usage"]
        for choice in c.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.content.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                slot = self.calls.setdefault(
                    tc.get("index", 0), {"id": None, "name": "", "arguments": ""}
                )
                fn = tc.get("function") or {}
                slot["id"] = tc.get("id") or slot["id"]
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def result(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": "".join(self.content) or None}
        if self.calls:
            message["tool_calls"] = [
                {
                    "id": s["id"],
                    "type": "function",
                    "function": {"name": s["name"], "arguments": s["arguments"]},
                }
                for _, s in sorted(self.calls.items())
            ]
        return {
            **self.base,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason}],
            "usage": self.usage,
        }


# ---- Responses API ------------------------------------------------------------------------
_RESP_CORE = frozenset(
    {"model", "input", "instructions", "tools", "temperature", "max_output_tokens"}
)


def responses_request_from_native(kwargs: dict[str, Any]) -> LLMRequest:
    k = clean_kwargs(kwargs)
    system = [k["instructions"]] if k.get("instructions") else []
    messages: list[Message] = []
    items = k.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    for item in items or []:
        kind = item.get("type", "message")
        if kind == "function_call":
            _append(
                messages,
                "assistant",
                [
                    ContentBlock(
                        type="tool_use",
                        tool_use_id=item.get("call_id"),
                        tool_name=item.get("name"),
                        tool_input=_parse_args(item.get("arguments")),
                    )
                ],
            )
        elif kind == "function_call_output":
            _append(
                messages,
                "user",
                [
                    ContentBlock(
                        type="tool_result",
                        tool_use_id=item.get("call_id"),
                        text=_text_of(item.get("output")),
                    )
                ],
            )
        elif kind == "message" and item.get("role") in ("system", "developer"):
            system.append(_text_of(item.get("content")))
        elif kind == "message":
            role = "assistant" if item.get("role") == "assistant" else "user"
            _append(messages, role, _parts(item.get("content")))
        else:
            _append(messages, "assistant", [ContentBlock(type="other", raw=item)])
    tools, other_tools = [], []
    for t in k.get("tools") or []:
        if t.get("type") == "function":
            tools.append(
                ToolSpec(
                    name=t["name"],
                    description=t.get("description") or "",
                    input_schema=t.get("parameters") or {},
                )
            )
        else:
            other_tools.append(t)
    max_tokens, param = _max_tokens(k, "max_output_tokens")
    extra = {key: v for key, v in k.items() if key not in _RESP_CORE}
    extra["max_tokens_param"] = param
    if other_tools:
        extra["other_tools"] = other_tools
    return LLMRequest(
        model=k["model"],
        system="\n\n".join(system) if system else None,
        messages=messages,
        tools=tools,
        temperature=k.get("temperature"),
        max_tokens=max_tokens,
        extra=extra,
    )


def responses_response_from_native(response: Any) -> LLMResponse:
    r = to_jsonable(response)
    blocks: list[ContentBlock] = []
    for item in r.get("output") or []:
        if item.get("type") == "message":
            blocks.extend(_parts(item.get("content")))
        elif item.get("type") == "function_call":
            blocks.append(
                ContentBlock(
                    type="tool_use",
                    tool_use_id=item.get("call_id"),
                    tool_name=item.get("name"),
                    tool_input=_parse_args(item.get("arguments")),
                )
            )
        else:
            blocks.append(ContentBlock(type="other", raw=item))
    usage = r.get("usage") or {}
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else r.get("status")
    return LLMResponse(
        content=blocks,
        stop_reason=stop,
        tokens_in=usage.get("input_tokens"),
        tokens_out=usage.get("output_tokens"),
    )


def responses_response_to_native(
    resp: LLMResponse, *, model: str, response_id: str
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    text = resp.text()
    if text:
        output.append(
            {
                "type": "message",
                "id": f"{response_id}_msg",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for i, b in enumerate(resp.tool_calls()):
        output.append(
            {
                "type": "function_call",
                "id": f"{response_id}_fc{i}",
                "call_id": b.tool_use_id,
                "name": b.tool_name,
                "arguments": json.dumps(b.tool_input or {}),
                "status": "completed",
            }
        )
    tin, tout = resp.tokens_in or 0, resp.tokens_out or 0
    return {
        "id": response_id,
        "object": "response",
        "created_at": 0,
        "model": model,
        "status": "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": tin,
            "output_tokens": tout,
            "total_tokens": tin + tout,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


class ResponsesStreamAggregator:
    """Keeps the final response from a Responses API event stream."""

    _FINAL = frozenset({"response.completed", "response.incomplete", "response.failed"})

    def __init__(self) -> None:
        self.response: dict[str, Any] = {}

    def feed(self, event: Any) -> None:
        e = to_jsonable(event)
        if e.get("type") in self._FINAL or (
            e.get("type") == "response.created" and not self.response
        ):
            self.response = e.get("response") or self.response

    def result(self) -> dict[str, Any]:
        return self.response


# ---- provider -----------------------------------------------------------------------------
class OpenAIProvider:
    """Sends normalized requests to OpenAI Chat Completions (async client)."""

    name = PROVIDER

    def __init__(
        self, client: Any = None, *, prices: dict[str, tuple[float, float]] | None = None
    ) -> None:
        self._client = client
        self._prices = prices or {}

    @property
    def client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(max_retries=0)
        return self._client

    @property
    def cache_namespace(self) -> str:
        return PROVIDER

    async def complete(self, req: LLMRequest, *, seed: int | None = None) -> LLMResponse:
        native = chat_request_to_native(req)
        if seed is not None and "seed" not in native:
            native["seed"] = seed  # best-effort determinism, supported by Chat Completions
        try:
            completion = await self.client.chat.completions.create(**native)
        except Exception as exc:
            from agentrelay.providers.anthropic import _provider_error

            raise _provider_error(exc) from exc
        return chat_response_from_native(completion)

    def price(self, model: str) -> tuple[float, float]:
        from agentrelay.providers.pricing import price_for

        return price_for(model, self._prices)[0]

    def estimate_cost(self, req: LLMRequest) -> float:
        from agentrelay.providers.pricing import estimate_cost

        return estimate_cost(req, self.price(req.model))

    def cost(self, resp: LLMResponse, req: LLMRequest) -> float | None:
        from agentrelay.providers.pricing import response_cost

        return response_cost(resp, self.price(req.model))
