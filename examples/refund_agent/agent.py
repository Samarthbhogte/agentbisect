"""Refund agent: answers refund questions from the policy knowledge base.

Bug: `search_policy` sometimes returns an outdated policy, and the agent quotes a 60-day refund
window instead of 30.

    python -m examples.refund_agent.agent --provider fake --seed 7
"""

from __future__ import annotations

from typing import Any

import agentrelay as relay

from .._common import DEFAULT_MODEL, cli, make_client, run_agent
from .fake_rules import DEFAULT, RULES
from .tools import get_order, search_policy

SYSTEM = """You are a customer support assistant for an online electronics store.

Answer refund questions using the store's policy documents. Look up the customer's order first,
then search the policy knowledge base, then give a short, friendly answer."""

SYSTEM_FIXED = (
    SYSTEM
    + "\n\nPolicies change over time.\n"
    + "If documents conflict, trust the document with the latest effective date."
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_order",
        "description": "Look up an order by its id.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Search store policies. Returns five documents.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]
TASK = "Hi, I'd like a refund for order A1001. Am I still eligible?"


@relay.entrypoint
def main(
    task: str = TASK,
    provider: str = "anthropic",
    model: str = DEFAULT_MODEL,
    seed: int = 0,
    fixed: bool = False,
) -> str:
    client = make_client(provider, RULES, DEFAULT, seed)
    run_input = {"task": task, "provider": provider, "model": model, "seed": seed, "fixed": fixed}
    with relay.run("refund", input=run_input) as run:
        answer = run_agent(
            client,
            model=model,
            system=SYSTEM_FIXED if fixed else SYSTEM,
            tools=TOOLS,
            tool_funcs={"get_order": get_order, "search_policy": search_policy},
            task=task,
        )
        run.set_output({"answer": answer})
    return answer


if __name__ == "__main__":
    cli(main, TASK)
