"""Shared pieces of the example agents: client setup, the tool-use loop, and a small CLI."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from typing import Any

import agentbisect as relay
from agentbisect.providers.base import LLMResponse
from agentbisect.providers.fake import FakeLLM, Rule

# The model id comes from --model or AGENTBISECT_MODEL. With --provider fake any name works.
PLACEHOLDER_MODEL = "fake-model"
DEFAULT_MODEL = os.environ.get("AGENTBISECT_MODEL", PLACEHOLDER_MODEL)
MAX_TURNS = 8
# Room for a reasoning model's thinking tokens as well as the visible answer.
CHAT_MAX_TOKENS = 4096


PROVIDERS = ("anthropic", "gemini", "fake")


def make_client(provider: str, rules: Sequence[Rule], default: LLMResponse, seed: int) -> Any:
    """A wrapped client. `fake` answers with FakeLLM rules, fully offline; `gemini` is an OpenAI
    client pointed at Gemini's compatible endpoint (key from GEMINI_API_KEY)."""
    if provider == "fake":
        from agentbisect.providers.fake_http import fake_anthropic_client

        llm = FakeLLM(rules=rules, default=default)
        return relay.wrap(fake_anthropic_client(llm, seed=seed))
    if provider == "anthropic":
        import anthropic

        return relay.wrap(anthropic.Anthropic())
    if provider == "gemini":
        import openai

        from agentbisect.providers.gemini import API_KEY_ENV, BASE_URL

        return relay.wrap(openai.OpenAI(base_url=BASE_URL, api_key=os.environ.get(API_KEY_ENV)))
    raise ValueError(f"unknown provider {provider!r}; use one of {', '.join(PROVIDERS)}")


def run_agent(
    client: Any,
    *,
    model: str,
    system: str,
    tools: list[dict[str, Any]],
    tool_funcs: dict[str, Callable[..., str]],
    task: str,
) -> str:
    """A plain tool-use loop: call the model, run requested tools, repeat.

    Speaks the Anthropic Messages format, or OpenAI Chat Completions when `client` is an OpenAI
    client (as for Gemini). `tools` are always given in the Anthropic shape.
    """
    if hasattr(client, "chat"):
        return _run_chat_agent(
            client, model=model, system=system, tools=tools, tool_funcs=tool_funcs, task=task
        )
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


def _call_tool(tool_funcs: dict[str, Callable[..., str]], name: str, args: dict[str, Any]) -> str:
    try:
        return tool_funcs[name](**args)
    except Exception as exc:
        return f"Error: {exc}"


def _run_chat_agent(
    client: Any,
    *,
    model: str,
    system: str,
    tools: list[dict[str, Any]],
    tool_funcs: dict[str, Callable[..., str]],
    task: str,
) -> str:
    """The same loop in the OpenAI Chat Completions format."""
    chat_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    for _ in range(MAX_TURNS):
        resp = client.chat.completions.create(
            model=model, max_completion_tokens=CHAT_MAX_TOKENS, messages=messages, tools=chat_tools
        )
        msg = resp.choices[0].message
        # exclude_none keeps provider extras (e.g. Gemini thought signatures) for the next turn
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content or ""
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            output = _call_tool(tool_funcs, call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
    return "Stopped: too many turns."


def cli(main: Callable[..., str], default_task: str, argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=default_task)
    parser.add_argument("--provider", default="anthropic", choices=list(PROVIDERS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0, help="FakeLLM seed (fake provider only)")
    parser.add_argument("--fixed", action="store_true", help="Use the documented fix")
    args = parser.parse_args(argv)
    if args.provider != "fake" and args.model == PLACEHOLDER_MODEL:
        parser.error(
            f"--provider {args.provider} needs a real model: pass --model <model id> "
            "or set AGENTBISECT_MODEL"
        )
    print(
        main(
            task=args.task,
            provider=args.provider,
            model=args.model,
            seed=args.seed,
            fixed=args.fixed,
        )
    )
