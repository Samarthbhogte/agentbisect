"""FakeLLM: an offline, reproducible stand-in for a real model.

Two modes:

* **Scripted:** returns a fixed list of responses in order.
* **Rules:** stochastic trigger rules. A rule fires when every ``when_all`` substring is in the
  rendered context, no ``unless_any`` substring is, and a seeded coin flip with
  ``probability`` comes up. The first rule that fires wins; otherwise the default response is
  returned. Matching is case-insensitive.

Randomness is derived from ``sha256(canonical(request) + seed)``, so each ``(request, seed)``
pair always gives the same answer, in any process.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from pydantic import BaseModel, Field

from agentrelay.providers.base import (
    LLMRequest,
    LLMResponse,
    render_context,
    text,
    text_block,
    tool_use,
)
from agentrelay.schema import canonical_json, sha256_hex

__all__ = ["FakeLLM", "FakeScriptExhausted", "Rule", "text", "text_block", "tool_use"]


class FakeScriptExhausted(RuntimeError):
    """A scripted FakeLLM was called more times than it has responses."""


class Rule(BaseModel):
    when_all: list[str] = Field(default_factory=list)
    unless_any: list[str] = Field(default_factory=list)
    respond: LLMResponse
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    name: str | None = None

    def matches(self, context: str) -> bool:
        """``context`` must already be lower-cased."""
        if not all(s.lower() in context for s in self.when_all):
            return False
        return not any(s.lower() in context for s in self.unless_any)


def rng_for(req: LLMRequest, seed: int | None) -> random.Random:
    digest = sha256_hex(canonical_json(req.to_dict()) + f"|seed={seed or 0}")
    return random.Random(int(digest[:16], 16))


class FakeLLM:
    """Implements the ``LLMProvider`` protocol without any network access."""

    def __init__(
        self,
        *,
        script: Sequence[LLMResponse] | None = None,
        rules: Sequence[Rule] | None = None,
        default: LLMResponse | None = None,
        price_in_per_mtok: float = 1.0,
        price_out_per_mtok: float = 5.0,
        name: str = "fake",
    ) -> None:
        if script is not None and rules:
            raise ValueError("FakeLLM takes either `script` or `rules`, not both")
        self.name = name
        self._script = list(script) if script is not None else None
        self._pos = 0
        self.rules = list(rules or [])
        self.default = default or text("Done.")
        self.price_in_per_mtok = price_in_per_mtok
        self.price_out_per_mtok = price_out_per_mtok
        self.calls = 0

    async def complete(self, req: LLMRequest, *, seed: int | None = None) -> LLMResponse:
        return self.complete_sync(req, seed=seed)

    def complete_sync(self, req: LLMRequest, *, seed: int | None = None) -> LLMResponse:
        self.calls += 1
        chosen = self._next_scripted() if self._script is not None else self.choose(req, seed)
        return self._with_usage(req, chosen)

    def choose(self, req: LLMRequest, seed: int | None) -> LLMResponse:
        """Pick a response by rules. Pure: does not count as a call."""
        rng = rng_for(req, seed)
        context = render_context(req).lower()
        for rule in self.rules:
            if rule.matches(context) and rng.random() < rule.probability:
                return rule.respond
        return self.default

    def estimate_cost(self, req: LLMRequest) -> float:
        tokens_in = _approx_tokens(render_context(req))
        return self._price(tokens_in, req.max_tokens)

    def cost(self, resp: LLMResponse) -> float:
        return self._price(resp.tokens_in or 0, resp.tokens_out or 0)

    def _next_scripted(self) -> LLMResponse:
        assert self._script is not None
        if self._pos >= len(self._script):
            raise FakeScriptExhausted(f"script has only {len(self._script)} responses")
        resp = self._script[self._pos]
        self._pos += 1
        return resp

    def _with_usage(self, req: LLMRequest, resp: LLMResponse) -> LLMResponse:
        out_text = resp.text() + "".join(f"{b.tool_name}{b.tool_input}" for b in resp.tool_calls())
        return resp.model_copy(
            deep=True,
            update={
                "tokens_in": _approx_tokens(render_context(req)),
                "tokens_out": _approx_tokens(out_text),
            },
        )

    def _price(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.price_in_per_mtok + tokens_out * self.price_out_per_mtok) / 1e6


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)
