"""`relay.wrap(client)`: pick the right proxy for an LLM client."""

from __future__ import annotations

from typing import Any

from agentbisect.sdk.context import disabled


def wrap(client: Any) -> Any:
    """Return a transparent proxy that records LLM calls made through `client`.

    Supports `anthropic.Anthropic`, `anthropic.AsyncAnthropic`, `openai.OpenAI` and
    `openai.AsyncOpenAI`. With `AGENTBISECT_DISABLED=1` the client is returned unchanged.
    """
    if disabled():
        return client
    module = type(client).__module__.split(".")[0]
    if module == "anthropic":
        from agentbisect.sdk.wrap_anthropic import wrap_anthropic

        return wrap_anthropic(client)
    if module == "openai":
        from agentbisect.sdk.wrap_openai import wrap_openai

        return wrap_openai(client)
    raise TypeError(
        f"agentbisect.wrap: unsupported client {type(client).__module__}.{type(client).__name__}; "
        "expected an anthropic or openai client"
    )
