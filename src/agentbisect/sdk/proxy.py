"""Transparent attribute-delegating proxy with a few intercepted methods."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentbisect.sdk.intercept import Adapter, call


class Proxy:
    """Delegates every attribute to `inner`, except:

    * names in `children` return another `Proxy` (e.g. `client.messages`), and
    * names in `methods` return a function that records the call with an `Adapter`.

    `root` is the wrapped client and `path` the attribute path from it to `inner`; replay uses
    them to call the same method on a copy of the client that serves recorded responses.
    """

    __slots__ = ("_children", "_inner", "_methods", "_path", "_root")

    def __init__(
        self,
        inner: Any,
        *,
        children: dict[str, Callable[[Any], Proxy]] | None = None,
        methods: dict[str, Adapter] | None = None,
        root: Any = None,
        path: tuple[str, ...] = (),
    ) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_children", children or {})
        object.__setattr__(self, "_methods", methods or {})
        object.__setattr__(self, "_root", inner if root is None else root)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._inner, name)
        if name in self._children:
            return self._children[name](target)
        if name in self._methods:
            adapter = self._methods[name]
            root, path = self._root, (*self._path, name)

            def recorded(*args: Any, **kwargs: Any) -> Any:
                return call(adapter, target, args, kwargs, root=root, path=path)

            return recorded
        return target

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)

    def __repr__(self) -> str:
        return f"agentbisect.wrap({self._inner!r})"

    def __enter__(self) -> Any:
        self._inner.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._inner.__exit__(*exc)

    async def __aenter__(self) -> Any:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._inner.__aexit__(*exc)
