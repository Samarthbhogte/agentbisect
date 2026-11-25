"""Model prices in USD per million tokens, used for budgets and cost reports.

The built-in table is intentionally small: prices change, and a wrong price silently breaks
budgets. Set real prices in `agentrelay.toml` under `[prices]`. A model with no known price is
charged the conservative `UNKNOWN_PRICE`, so a budget can be exceeded only on the safe side.
"""

from __future__ import annotations

from agentrelay.providers.base import LLMRequest, LLMResponse, render_context

PRICES: dict[str, tuple[float, float]] = {
    "fake": (1.0, 5.0),
}
UNKNOWN_PRICE: tuple[float, float] = (15.0, 75.0)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def price_for(
    model: str, overrides: dict[str, tuple[float, float]] | None = None
) -> tuple[tuple[float, float], bool]:
    """((input, output) $/Mtok, known?). Exact id first, then the longest matching prefix."""
    table = {**PRICES, **(overrides or {})}
    if model in table:
        return table[model], True
    prefixes = [k for k in table if model.startswith(k)]
    if prefixes:
        return table[max(prefixes, key=len)], True
    return UNKNOWN_PRICE, False


def estimate_cost(req: LLMRequest, prices: tuple[float, float]) -> float:
    """Upper-bound estimate: approximate input tokens plus the full `max_tokens` of output."""
    return (approx_tokens(render_context(req)) * prices[0] + req.max_tokens * prices[1]) / 1e6


def response_cost(resp: LLMResponse, prices: tuple[float, float]) -> float | None:
    if resp.tokens_in is None or resp.tokens_out is None:
        return None
    return (resp.tokens_in * prices[0] + resp.tokens_out * prices[1]) / 1e6
