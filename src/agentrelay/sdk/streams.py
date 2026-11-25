"""Stream pass-through: hand every chunk to the caller unchanged, record the final response
when the stream ends (exhausted, closed, or its context manager exits)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from agentrelay.sdk.context import safe


class Aggregator(Protocol):
    def feed(self, event: Any) -> None: ...
    def result(self) -> dict[str, Any]: ...


class _Base:
    def __init__(
        self, inner: Any, agg: Aggregator, on_done: Callable[[dict[str, Any]], None]
    ) -> None:
        self._inner = inner
        self._agg = agg
        self._on_done = on_done
        self._done = False

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            safe(lambda: self._on_done(self._agg.result()))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class StreamProxy(_Base):
    def __init__(
        self, inner: Any, agg: Aggregator, on_done: Callable[[dict[str, Any]], None]
    ) -> None:
        super().__init__(inner, agg, on_done)
        self._it: Any = None

    def __iter__(self) -> StreamProxy:
        return self

    def __next__(self) -> Any:
        if self._it is None:
            self._it = iter(self._inner)
        try:
            event = next(self._it)
        except StopIteration:
            self._finish()
            raise
        safe(self._agg.feed, event)
        return event

    def close(self) -> None:
        self._finish()
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> StreamProxy:
        enter = getattr(self._inner, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, *exc: Any) -> Any:
        self._finish()
        exit_ = getattr(self._inner, "__exit__", None)
        return exit_(*exc) if callable(exit_) else None


class AsyncStreamProxy(_Base):
    def __init__(
        self, inner: Any, agg: Aggregator, on_done: Callable[[dict[str, Any]], None]
    ) -> None:
        super().__init__(inner, agg, on_done)
        self._it: Any = None

    def __aiter__(self) -> AsyncStreamProxy:
        return self

    async def __anext__(self) -> Any:
        if self._it is None:
            self._it = self._inner.__aiter__()
        try:
            event = await self._it.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        safe(self._agg.feed, event)
        return event

    async def close(self) -> None:
        self._finish()
        close = getattr(self._inner, "close", None)
        if callable(close):
            await close()

    async def __aenter__(self) -> AsyncStreamProxy:
        enter = getattr(self._inner, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        self._finish()
        exit_ = getattr(self._inner, "__aexit__", None)
        return await exit_(*exc) if callable(exit_) else None
