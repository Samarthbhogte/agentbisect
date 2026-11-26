"""Shared logic for intercepting one LLM method: call it, then record request and response."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agentrelay._json import to_jsonable
from agentrelay.providers.base import LLMRequest, LLMResponse
from agentrelay.sdk.context import current_recorder, current_replay, disabled, now, safe
from agentrelay.sdk.streams import Aggregator, AsyncStreamProxy, StreamProxy


@dataclass(frozen=True)
class Adapter:
    """How to translate one provider method's native request and response."""

    provider: str
    request_from_native: Callable[[dict[str, Any]], LLMRequest]
    response_from_native: Callable[[Any], LLMResponse]
    clean_kwargs: Callable[[dict[str, Any]], dict[str, Any]]
    new_aggregator: Callable[[], Aggregator]


def _record(
    adapter: Adapter,
    kwargs: dict[str, Any],
    native: Any,
    error: BaseException | None,
    started: datetime,
    tags: dict[str, str] | None = None,
) -> None:
    rec = current_recorder()
    if rec is None:
        return
    raw_response = to_jsonable(native) if native is not None else None
    rec.record_llm(
        provider=adapter.provider,
        request=adapter.request_from_native(kwargs),
        response=adapter.response_from_native(raw_response) if raw_response else None,
        raw_request=adapter.clean_kwargs(kwargs),
        raw_response=raw_response,
        error=error,
        started_at=started,
        ended_at=now(),
        tags=tags,
    )


def call(
    adapter: Adapter,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    root: Any = None,
    path: tuple[str, ...] = (),
) -> Any:
    """Call `fn` and record it. Works for sync methods and for methods returning awaitables.

    During program-level replay the controller decides whether the call is served from the
    recorded cassette (through a copy of the client) or goes live, and may raise
    `ReplayDivergence`.
    """
    if disabled():
        return fn(*args, **kwargs)
    tags: dict[str, str] | None = None
    controller = current_replay()
    if controller is not None:
        fn, tags = controller.route_llm(adapter, fn, kwargs, root=root, path=path)
    started = now()
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        safe(_record, adapter, kwargs, None, exc, started, tags)
        raise
    if inspect.isawaitable(result):
        return _finish_async(adapter, result, kwargs, started, tags)
    return _finish(adapter, result, kwargs, started, tags)


def _finish(
    adapter: Adapter,
    result: Any,
    kwargs: dict[str, Any],
    started: datetime,
    tags: dict[str, str] | None,
) -> Any:
    if kwargs.get("stream"):
        return StreamProxy(
            result,
            adapter.new_aggregator(),
            lambda final: _record(adapter, kwargs, final, None, started, tags),
        )
    safe(_record, adapter, kwargs, result, None, started, tags)
    return result


async def _finish_async(
    adapter: Adapter,
    pending: Any,
    kwargs: dict[str, Any],
    started: datetime,
    tags: dict[str, str] | None,
) -> Any:
    try:
        result = await pending
    except Exception as exc:
        safe(_record, adapter, kwargs, None, exc, started, tags)
        raise
    if kwargs.get("stream"):
        return AsyncStreamProxy(
            result,
            adapter.new_aggregator(),
            lambda final: _record(adapter, kwargs, final, None, started, tags),
        )
    safe(_record, adapter, kwargs, result, None, started, tags)
    return result
