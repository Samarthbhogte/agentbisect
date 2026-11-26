"""`@relay.tool`: record a tool's arguments, result, errors and timing."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from agentrelay.sdk.context import current_recorder, current_replay, disabled, now, safe

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class ToolInfo:
    name: str
    side_effects: bool


def tool_info(fn: Any) -> ToolInfo | None:
    info = getattr(fn, "__agentrelay_tool__", None)
    return info if isinstance(info, ToolInfo) else None


def _bind(sig: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        bound = sig.bind_partial(*args, **kwargs)
    except TypeError:
        return {"args": list(args), "kwargs": kwargs}
    return dict(bound.arguments)


def _record(
    info: ToolInfo,
    sig: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    error: BaseException | None,
    started: Any,
    tags: dict[str, str] | None = None,
) -> None:
    rec = current_recorder()
    if rec is not None:
        rec.record_tool(
            name=info.name,
            args=_bind(sig, args, kwargs),
            result=result,
            error=error,
            started_at=started,
            ended_at=now(),
            side_effects=info.side_effects,
            extra_tags=tags,
        )


def _replayed(
    info: ToolInfo, sig: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[bool, Any, dict[str, str] | None]:
    """(served, result, tags). Raises the recorded error, `ToolNotInCassette` or
    `SideEffectRefused` as the replay controller decides."""
    controller = current_replay()
    if controller is None:
        return False, None, None
    return controller.route_tool(info, _bind(sig, args, kwargs))  # type: ignore[no-any-return]


def _decorate(fn: F, name: str | None, side_effects: bool) -> F:
    info = ToolInfo(name=name or fn.__name__, side_effects=side_effects)
    sig = inspect.signature(fn)

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if disabled():
                return await fn(*args, **kwargs)
            started = now()
            try:
                served, result, tags = _replayed(info, sig, args, kwargs)
                if not served:
                    result = await fn(*args, **kwargs)
            except Exception as exc:
                safe(_record, info, sig, args, kwargs, None, exc, started, _replay_tags())
                raise
            safe(_record, info, sig, args, kwargs, result, None, started, tags)
            return result

        async_wrapper.__agentrelay_tool__ = info  # type: ignore[attr-defined]
        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if disabled():
            return fn(*args, **kwargs)
        started = now()
        try:
            served, result, tags = _replayed(info, sig, args, kwargs)
            if not served:
                result = fn(*args, **kwargs)
        except Exception as exc:
            safe(_record, info, sig, args, kwargs, None, exc, started, _replay_tags())
            raise
        safe(_record, info, sig, args, kwargs, result, None, started, tags)
        return result

    wrapper.__agentrelay_tool__ = info  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def _replay_tags() -> dict[str, str] | None:
    return {"replay": "refused-or-error"} if current_replay() is not None else None


@overload
def tool(fn: F, /) -> F: ...
@overload
def tool(*, name: str | None = None, side_effects: bool = False) -> Callable[[F], F]: ...


def tool(fn: Any = None, /, *, name: str | None = None, side_effects: bool = False) -> Any:
    """Record calls to a tool. Use as `@relay.tool` or `@relay.tool(side_effects=True)`.

    Tools with `side_effects=True` are never run live during replay unless explicitly allowed.
    """
    if fn is not None:
        return _decorate(fn, name, side_effects)
    return lambda f: _decorate(f, name, side_effects)
