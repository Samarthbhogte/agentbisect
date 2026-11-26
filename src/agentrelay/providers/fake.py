"""FakeLLM: an offline, reproducible stand-in for a real model.

Two modes:

* **Scripted:** returns a fixed list of responses in order.
* **Rules:** stochastic trigger rules. A rule fires when every ``when_all`` substring is in the
  rendered context, no ``unless_any`` substring is, the conversation turn (the number of
  assistant messages so far) is within ``min_turn``..``max_turn``, and a seeded coin flip with
  ``probability`` comes up. The first rule that fires wins; otherwise the default response is
  returned. Matching is case-insensitive.

Randomness is derived from ``sha256(canonical(request) + seed)``, so each ``(request, seed)``
pair always gives the same answer, in any process.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any

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


# Added to every seed. Program-level replay sets it per trial so live continuations of a recorded
# run draw fresh samples, the way a real model at temperature > 0 would.
SEED_OFFSET: ContextVar[int] = ContextVar("agentrelay_fake_seed_offset", default=0)


class FakeScriptExhausted(RuntimeError):
    """A scripted FakeLLM was called more times than it has responses."""


class Rule(BaseModel):
    when_all: list[str] = Field(default_factory=list)
    unless_any: list[str] = Field(default_factory=list)
    respond: LLMResponse
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    name: str | None = None
    min_turn: int = 0
    max_turn: int | None = None

    def matches(self, context: str, turn: int = 0) -> bool:
        """``context`` must already be lower-cased; ``turn`` = assistant messages so far."""
        if turn < self.min_turn or (self.max_turn is not None and turn > self.max_turn):
            return False
        if not all(s.lower() in context for s in self.when_all):
            return False
        return not any(s.lower() in context for s in self.unless_any)


def rng_for(req: LLMRequest, seed: int | None) -> random.Random:
    effective = (seed or 0) + SEED_OFFSET.get()
    digest = sha256_hex(canonical_json(req.to_dict()) + f"|seed={effective}")
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
        turn = sum(1 for m in req.messages if m.role == "assistant")
        for rule in self.rules:
            if rule.matches(context, turn) and rng.random() < rule.probability:
                return rule.respond
        return self.default

    def estimate_cost(self, req: LLMRequest) -> float:
        tokens_in = _approx_tokens(render_context(req))
        return self._price(tokens_in, req.max_tokens)

    def cost(self, resp: LLMResponse, req: LLMRequest | None = None) -> float:
        return self._price(resp.tokens_in or 0, resp.tokens_out or 0)

    @property
    def cache_namespace(self) -> str:
        """Identifies this rule set, so cached answers from different rules never mix."""
        if self._script is not None:
            return "fake:script:" + sha256_hex(canonical_json([r.to_dict() for r in self._script]))
        rules = [r.model_dump(mode="json") for r in self.rules]
        return "fake:" + sha256_hex(canonical_json([rules, self.default.to_dict()]))

    @classmethod
    def from_rules_file(cls, ref: str, **kwargs: Any) -> FakeLLM:
        """Load `RULES` (and optional `DEFAULT`) from a Python file such as `fake_rules.py`."""
        from agentrelay._loader import load_module

        module = load_module(ref)
        rules = getattr(module, "RULES", None)
        if rules is None:
            raise AttributeError(f"{ref} must define RULES (a list of Rule)")
        return cls(rules=rules, default=getattr(module, "DEFAULT", None), **kwargs)

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
