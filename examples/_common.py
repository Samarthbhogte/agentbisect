"""Shared pieces of the example agents: client setup, the tool-use loop, and a small CLI."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from typing import Any

import agentrelay as relay
from agentrelay.providers.base import LLMResponse
from agentrelay.providers.fake import FakeLLM, Rule

DEFAULT_MODEL = os.environ.get("AGENTRELAY_MODEL", "fake-model")
MAX_TURNS = 8


def make_client(provider: str, rules: Sequence[Rule], default: LLMResponse, seed: int) -> Any:
    """A wrapped Anthropic client. `fake` answers with FakeLLM rules, fully offline."""
    if provider == "fake":
        from agentrelay.providers.fake_http import fake_anthropic_client

        llm = FakeLLM(rules=rules, default=default)
        return relay.wrap(fake_anthropic_client(llm, seed=seed))
    if provider == "anthropic":
        import anthropic

        return relay.wrap(anthropic.Anthropic())
    raise ValueError(f"unknown provider {provider!r}; use 'anthropic' or 'fake'")


def run_agent(
    client: Any,
    *,
    model: str,
    system: str,
    tools: list[dict[str, Any]],
    tool_funcs: dict[str, Callable[..., str]],
    task: str,
) -> str:
    """A plain Anthropic tool-use loop: call the model, run requested tools, repeat."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    for _ in range(MAX_TURNS):
        resp = client.messages.create(
            model=model, max_tokens=1024, system=system, tools=tools, messages=messages
        )
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            return "".join(b.text for b in resp.content if b.type == "text")
        results = []
        for block in uses:
            try:
                output, is_error = tool_funcs[block.name](**block.input), False
            except Exception as exc:
                output, is_error = f"Error: {exc}", True
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})
    return "Stopped: too many turns."


def cli(main: Callable[..., str], default_task: str, argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=default_task)
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "fake"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0, help="FakeLLM seed (fake provider only)")
    parser.add_argument("--fixed", action="store_true", help="Use the documented fix")
    args = parser.parse_args(argv)
    print(
        main(
            task=args.task,
            provider=args.provider,
            model=args.model,
            seed=args.seed,
            fixed=args.fixed,
        )
    )
