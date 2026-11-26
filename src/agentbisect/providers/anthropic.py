"""Anthropic Messages API <-> normalized format.

Blocks the normalized format has no field for (thinking, documents, server tools, ...) are kept
verbatim as ``type="other"`` (or ``"image"``) blocks with the original payload in ``raw``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agentbisect._json import to_jsonable
from agentbisect.providers.base import ContentBlock, LLMRequest, LLMResponse, Message, ToolSpec

PROVIDER = "anthropic"
# Request kwargs that never reach the trace: transport options and HTTP headers (keys).
TRANSPORT_KWARGS = frozenset({"stream", "extra_headers", "extra_query", "extra_body", "timeout"})
_CORE = frozenset({"model", "system", "messages", "tools", "temperature", "max_tokens"})


def clean_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Provider-native request without transport options or headers, JSON-ready."""
    out: dict[str, Any] = to_jsonable(
        {k: v for k, v in kwargs.items() if k not in TRANSPORT_KWARGS}
    )
    return out


# ---- request ------------------------------------------------------------------------------
def request_from_native(kwargs: dict[str, Any]) -> LLMRequest:
    k = clean_kwargs(kwargs)
    tools, server_tools = [], []
    for t in k.get("tools") or []:
        if "input_schema" in t:
            tools.append(
                ToolSpec(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t["input_schema"],
                )
            )
        else:
            server_tools.append(t)
    extra = {key: v for key, v in k.items() if key not in _CORE}
    if server_tools:
        extra["server_tools"] = server_tools
    return LLMRequest(
        model=k["model"],
        system=_system_text(k.get("system")),
        messages=[Message(role=m["role"], content=_blocks(m["content"])) for m in k["messages"]],
        tools=tools,
        temperature=k.get("temperature"),
        max_tokens=k["max_tokens"],
        extra=extra,
    )


def _system_text(system: Any) -> str | None:
    if system is None or isinstance(system, str):
        return system
    return "\n\n".join(b.get("text", "") for b in system if b.get("type") == "text")


def _blocks(content: Any) -> list[ContentBlock]:
    if isinstance(content, str):
        return [ContentBlock(type="text", text=content)]
    return [_block(b) for b in content]


def _block(b: dict[str, Any]) -> ContentBlock:
    kind = b.get("type")
    if kind == "text":
        return ContentBlock(type="text", text=b.get("text", ""))
    if kind == "tool_use":
        return ContentBlock(
            type="tool_use",
            tool_use_id=b.get("id"),
            tool_name=b.get("name"),
            tool_input=b.get("input") or {},
        )
    if kind == "tool_result":
        content = b.get("content")
        if content is None or isinstance(content, str):
            text, raw = content or "", None
        else:
            text = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
            raw = b if any(p.get("type") != "text" for p in content) else None
        return ContentBlock(
            type="tool_result",
            tool_use_id=b.get("tool_use_id"),
            text=text,
            is_error=b.get("is_error"),
            raw=raw,
        )
    if kind == "image":
        return ContentBlock(type="image", raw=b)
    return ContentBlock(type="other", raw=b)


def request_to_native(req: LLMRequest) -> dict[str, Any]:
    """Inverse of `request_from_native` (used to send normalized requests to the API)."""
    extra = dict(req.extra)
    server_tools = extra.pop("server_tools", [])
    out: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "messages": [
            {"role": m.role, "content": [_block_to_native(b) for b in m.content]}
            for m in req.messages
        ],
    }
    if req.system is not None:
        out["system"] = req.system
    tools = [t.model_dump() for t in req.tools] + server_tools
    if tools:
        out["tools"] = tools
    if req.temperature is not None:
        out["temperature"] = req.temperature
    out.update(extra)
    return out


def _block_to_native(b: ContentBlock) -> dict[str, Any]:
    if b.type == "text":
        return {"type": "text", "text": b.text or ""}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.tool_use_id, "name": b.tool_name, "input": b.tool_input}
    if b.type == "tool_result":
        if b.raw is not None:
            return b.raw
        out: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": b.tool_use_id,
            "content": b.text or "",
        }
        if b.is_error is not None:
            out["is_error"] = b.is_error
        return out
    return b.raw or {}


# ---- response -----------------------------------------------------------------------------
def response_from_native(message: Any) -> LLMResponse:
    m = to_jsonable(message)
    blocks = []
    for b in m.get("content") or []:
        if b.get("type") == "text":
            blocks.append(ContentBlock(type="text", text=b.get("text", "")))
        elif b.get("type") == "tool_use":
            blocks.append(
                ContentBlock(
                    type="tool_use",
                    tool_use_id=b.get("id"),
                    tool_name=b.get("name"),
                    tool_input=b.get("input") or {},
                )
            )
        else:
            blocks.append(ContentBlock(type="other", raw=b))
    usage = m.get("usage") or {}
    return LLMResponse(
        content=blocks,
        stop_reason=m.get("stop_reason"),
        tokens_in=usage.get("input_tokens"),
        tokens_out=usage.get("output_tokens"),
    )


def response_to_native(resp: LLMResponse, *, model: str, message_id: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for b in resp.content:
        if b.type == "text":
            content.append({"type": "text", "text": b.text or ""})
        elif b.type == "tool_use":
            content.append(
                {
                    "type": "tool_use",
                    "id": b.tool_use_id,
                    "name": b.tool_name,
                    "input": b.tool_input or {},
                }
            )
        elif b.raw is not None:
            content.append(b.raw)
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": resp.stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": resp.tokens_in or 0, "output_tokens": resp.tokens_out or 0},
    }


class StreamAggregator:
    """Rebuilds the final Message from `stream=True` events."""

    def __init__(self) -> None:
        self.message: dict[str, Any] = {"content": [], "usage": {}}
        self._json_parts: dict[int, list[str]] = {}

    def feed(self, event: Any) -> None:
        e = to_jsonable(event)
        kind = e.get("type")
        if kind == "message_start":
            self.message = {**e["message"], "content": []}
        elif kind == "content_block_start":
            self.message["content"].append(dict(e["content_block"]))
        elif kind == "content_block_delta":
            self._delta(e["index"], e["delta"])
        elif kind == "content_block_stop":
            self._close_block(e["index"])
        elif kind == "message_delta":
            self.message.update({k: v for k, v in e.get("delta", {}).items() if v is not None})
            usage = {k: v for k, v in (e.get("usage") or {}).items() if v is not None}
            self.message.setdefault("usage", {}).update(usage)

    def _delta(self, index: int, delta: dict[str, Any]) -> None:
        block = self.message["content"][index]
        if delta.get("type") == "text_delta":
            block["text"] = block.get("text", "") + delta.get("text", "")
        elif delta.get("type") == "input_json_delta":
            self._json_parts.setdefault(index, []).append(delta.get("partial_json", ""))
        elif delta.get("type") == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")

    def _close_block(self, index: int) -> None:
        parts = self._json_parts.pop(index, None)
        if parts is not None:
            joined = "".join(parts)
            self.message["content"][index]["input"] = json.loads(joined) if joined else {}

    def result(self) -> dict[str, Any]:
        for index in list(self._json_parts):
            self._close_block(index)
        return self.message


# ---- provider -----------------------------------------------------------------------------
class AnthropicProvider:
    """Sends normalized requests to the Anthropic Messages API (async client)."""

    name = PROVIDER

    def __init__(
        self, client: Any = None, *, prices: dict[str, tuple[float, float]] | None = None
    ) -> None:
        self._client = client
        self._prices = prices or {}

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(max_retries=0)
        return self._client

    @property
    def cache_namespace(self) -> str:
        return PROVIDER

    async def complete(self, req: LLMRequest, *, seed: int | None = None) -> LLMResponse:
        try:
            message = await self.client.messages.create(**request_to_native(req))
        except Exception as exc:
            raise _provider_error(exc) from exc
        return response_from_native(message)

    def price(self, model: str) -> tuple[float, float]:
        from agentbisect.providers.pricing import price_for

        return price_for(model, self._prices)[0]

    def estimate_cost(self, req: LLMRequest) -> float:
        from agentbisect.providers.pricing import estimate_cost

        return estimate_cost(req, self.price(req.model))

    def cost(self, resp: LLMResponse, req: LLMRequest) -> float | None:
        from agentbisect.providers.pricing import response_cost

        return response_cost(resp, self.price(req.model))


def _provider_error(exc: Exception) -> Exception:
    from agentbisect.providers.base import ProviderError

    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    retryable = (isinstance(status, int) and (status == 429 or status >= 500)) or name in (
        "APIConnectionError",
        "APITimeoutError",
    )
    if status is None and not retryable:
        return exc
    return ProviderError(
        f"{name}: {_short(exc)}",
        retryable=retryable,
        status_code=status,
        retry_after=retry_after_seconds(exc),
    )


_RETRY_DELAY = re.compile(r"""["']retryDelay["']\s*:\s*["'](\d+(?:\.\d+)?)s["']""")


def retry_after_seconds(exc: Exception) -> float | None:
    """How long the provider asked us to wait: the `Retry-After` header, else a Google-style
    `retryDelay` in the error body."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    value = headers.get("retry-after") if headers is not None else None
    if value is not None:
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
    match = _RETRY_DELAY.search(f"{exc} {getattr(exc, 'body', '')}")
    return float(match.group(1)) if match else None


def _short(exc: Exception, limit: int = 300) -> str:
    text = " ".join(str(exc).split())
    detail = _body_message(getattr(exc, "body", None))
    if detail and detail not in text:
        text = f"{text} - {' '.join(detail.split())}"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _body_message(body: Any) -> str | None:
    """The human-readable message in an API error body (`{"error": {"message": ...}}`, possibly
    wrapped in a list, as Google returns it)."""
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        inner = body.get("error", body)
        message = inner.get("message") if isinstance(inner, dict) else None
        return str(message) if message else None
    return None
