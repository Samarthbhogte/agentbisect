"""Serve recorded responses through a real SDK client (program-level replay).

`cassette_client(root, serve)` copies the user's Anthropic or OpenAI client with an in-memory
HTTP layer. Every request it receives is answered by `serve()`, which returns the recorded,
provider-native response (or None for a recorded error). Streaming requests get the response
re-encoded as the provider's server-sent events, so the SDK parses everything exactly as it did
when the run was recorded.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from agentbisect.providers.fake_http import (
    _anthropic_events,
    _chat_chunks,
    _responses_events,
    _server_error,
    _sse,
)

Serve = Callable[[], dict[str, Any] | None]


def _handler(provider: str, serve: Serve) -> Callable[[Any], Any]:
    import httpx2

    def handle(request: Any) -> Any:
        body = json.loads(request.content or b"{}")
        native = serve()
        if native is None:
            return _server_error(RuntimeError("the recorded call failed"))
        if not body.get("stream"):
            return httpx2.Response(200, json=native)
        path = request.url.path
        if provider == "anthropic":
            events: Any = _anthropic_events(native)
        elif path.endswith("/responses"):
            events = _responses_events(native)
        else:
            events = _chat_chunks(native)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse(events)
        )

    return handle


def cassette_client(root: Any, provider: str, serve: Serve) -> Any:
    """A copy of `root` (sync or async) whose HTTP layer answers with `serve()`."""
    import httpx2

    transport = httpx2.MockTransport(_handler(provider, serve))
    is_async = type(root).__name__.startswith("Async")
    http_client = (
        httpx2.AsyncClient(transport=transport) if is_async else httpx2.Client(transport=transport)
    )
    return root.with_options(http_client=http_client, max_retries=0)
