"""Normalized LLM format and the provider protocol.

Everything outside ``providers/`` and ``sdk/wrap_*.py`` uses these types; adapters translate to
and from provider-native formats.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ContentBlock(BaseModel):
    type: Literal["text", "tool_use", "tool_result", "image", "other"]
    text: str | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    is_error: bool | None = None
    raw: dict[str, Any] | None = None


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentBlock]


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class LLMRequest(BaseModel):
    model: str
    system: str | None
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict; this is what ``Step.request`` stores for LLM calls."""
        return self.model_dump(mode="json")


class LLMResponse(BaseModel):
    content: list[ContentBlock]
    stop_reason: str | None
    tokens_in: int | None = None
    tokens_out: int | None = None

    def tool_calls(self) -> list[ContentBlock]:
        return [b for b in self.content if b.type == "tool_use"]

    def text(self) -> str:
        return "".join(b.text or "" for b in self.content if b.type == "text")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ProviderError(Exception):
    """A provider call failed. `retryable` is True for rate limits, 5xx and connection errors."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after = retry_after  # seconds the provider asked us to wait, if it said


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def complete(self, req: LLMRequest, *, seed: int | None = None) -> LLMResponse: ...

    def estimate_cost(self, req: LLMRequest) -> float: ...


def text_block(text: str) -> ContentBlock:
    return ContentBlock(type="text", text=text)


def text(content: str, *, stop_reason: str = "end_turn") -> LLMResponse:
    """A response containing one text block."""
    return LLMResponse(content=[text_block(content)], stop_reason=stop_reason)


def tool_use(
    name: str, tool_input: dict[str, Any], *, tool_use_id: str = "toolu_fake"
) -> LLMResponse:
    """A response that calls one tool."""
    block = ContentBlock(
        type="tool_use", tool_use_id=tool_use_id, tool_name=name, tool_input=tool_input
    )
    return LLMResponse(content=[block], stop_reason="tool_use")


def render_context(req: LLMRequest) -> str:
    """Flatten everything the model would read into one string (system, tools, messages)."""
    parts: list[str] = []
    if req.system:
        parts.append(req.system)
    for tool in req.tools:
        parts.append(f"{tool.name}: {tool.description}")
    for msg in req.messages:
        for block in msg.content:
            if block.text:
                parts.append(block.text)
            if block.tool_name:
                parts.append(block.tool_name)
            if block.tool_input is not None:
                parts.append(str(block.tool_input))
    return "\n".join(parts)
