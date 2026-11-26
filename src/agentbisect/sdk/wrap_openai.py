"""Proxy for `openai.OpenAI` / `AsyncOpenAI`: records `chat.completions.create` and
`responses.create`."""

from __future__ import annotations

from typing import Any

from agentbisect.providers import openai as oai
from agentbisect.sdk.intercept import Adapter
from agentbisect.sdk.proxy import Proxy

CHAT = Adapter(
    provider=oai.PROVIDER,
    request_from_native=oai.chat_request_from_native,
    response_from_native=oai.chat_response_from_native,
    clean_kwargs=oai.clean_kwargs,
    new_aggregator=oai.ChatStreamAggregator,
)
RESPONSES = Adapter(
    provider=oai.PROVIDER,
    request_from_native=oai.responses_request_from_native,
    response_from_native=oai.responses_response_from_native,
    clean_kwargs=oai.clean_kwargs,
    new_aggregator=oai.ResponsesStreamAggregator,
)


def wrap_openai(client: Any) -> Proxy:
    def completions(c: Any) -> Proxy:
        return Proxy(c, methods={"create": CHAT}, root=client, path=("chat", "completions"))

    def chat(c: Any) -> Proxy:
        return Proxy(c, children={"completions": completions}, root=client, path=("chat",))

    def responses(r: Any) -> Proxy:
        return Proxy(r, methods={"create": RESPONSES}, root=client, path=("responses",))

    return Proxy(client, children={"chat": chat, "responses": responses})
