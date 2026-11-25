"""Proxy for `anthropic.Anthropic` / `AsyncAnthropic`: records `messages.create`."""

from __future__ import annotations

from typing import Any

from agentrelay.providers import anthropic as anth
from agentrelay.sdk.intercept import Adapter
from agentrelay.sdk.proxy import Proxy

ADAPTER = Adapter(
    provider=anth.PROVIDER,
    request_from_native=anth.request_from_native,
    response_from_native=anth.response_from_native,
    clean_kwargs=anth.clean_kwargs,
    new_aggregator=anth.StreamAggregator,
)


def wrap_anthropic(client: Any) -> Proxy:
    return Proxy(
        client,
        children={
            "messages": lambda messages: Proxy(messages, methods={"create": ADAPTER}),
        },
    )
