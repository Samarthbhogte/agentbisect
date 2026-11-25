"""Attribution: which units of the minimal context cause the failure (BLUEPRINT 6.9).

For each removable unit `u` of the minimal set `M` (chunks rather than their tool result):

1. necessity in the minimal context: p(M without u) vs p(M)
2. necessity in the original context: p(Full without u) vs p0 (the real check)
3. counterfactual (tool results and chunks): replace u with a neutral rewrite and re-measure

attribution(u) = p0 - p(Full without u), with a Newcombe interval built from the two Wilson
intervals. `u` is a culprit when the two Wilson intervals do not overlap. Several culprits are an
"and" interaction (the failure needs them together). With no single culprit, pairs within M
(|M| <= 6) are tested; a pair whose joint removal fixes it is an "or" interaction (either is
enough).
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, Field

from agentrelay.algo.shrink import ShrinkResult
from agentrelay.context.neutralize import neutralize, removed_text
from agentrelay.context.units import recompose, substitute, unit_text
from agentrelay.providers.base import LLMRequest
from agentrelay.schema import ContextUnit, RateEstimate
from agentrelay.stats.runner import TrialRunner

Rewriter = Callable[[str], Awaitable[str]]
SubstitutionSource = Literal["provided", "generated", "heuristic"]
FULL_TEXT_CHARS = 2000


class ExplainConfig(BaseModel):
    baseline_trials: int = 20
    trials: int = 16
    max_pair_units: int = 6


class Counterfactual(BaseModel):
    source: SubstitutionSource
    text: str
    removed: str | None = None
    estimate: RateEstimate | None = None
    attribution: float | None = None
    note: str


class UnitAttribution(BaseModel):
    rank: int
    unit_id: str
    kind: str
    tokens: int
    text: str
    without_full: RateEstimate
    without_minimal: RateEstimate | None = None
    attribution: float
    attribution_low: float
    attribution_high: float
    culprit: bool
    counterfactual: Counterfactual | None = None


class Interaction(BaseModel):
    kind: Literal["and", "or"]
    units: list[str]
    estimate: RateEstimate | None = None
    message: str


ExplainStatus = Literal["ok", "no_culprit", "partial", "too_rare", "not_failing"]


class ExplainResult(BaseModel):
    status: ExplainStatus
    message: str
    fingerprint: str
    baseline: RateEstimate
    minimal_estimate: RateEstimate | None = None
    attributions: list[UnitAttribution] = Field(default_factory=list)
    interactions: list[Interaction] = Field(default_factory=list)
    not_removable: list[str] = Field(default_factory=list)
    shrink: ShrinkResult
    trials: int = 0
    cached_trials: int = 0
    provider_calls: int = 0
    cost_usd: float = 0.0
    run_id: str | None = None
    step_id: str | None = None
    model: str | None = None
    oracle: str | None = None
    oracle_noisy: bool = False
    elapsed_s: float | None = None

    @property
    def culprits(self) -> list[UnitAttribution]:
        return [a for a in self.attributions if a.culprit]


class _BudgetStop(Exception):
    pass


def newcombe(p1: RateEstimate, p2: RateEstimate) -> tuple[float, float, float]:
    """Difference p1 - p2 with Newcombe's hybrid score interval from two Wilson intervals."""
    d = p1.p_hat - p2.p_hat
    low = d - math.sqrt((p1.p_hat - p1.ci_low) ** 2 + (p2.ci_high - p2.p_hat) ** 2)
    high = d + math.sqrt((p1.ci_high - p1.p_hat) ** 2 + (p2.p_hat - p2.ci_low) ** 2)
    return d, max(-1.0, low), min(1.0, high)


def explained_units(shrink: ShrinkResult) -> list[ContextUnit]:
    """Removable units of the minimal set, at the finest level (chunks instead of their parent)."""
    minimal = set(shrink.minimal_ids)
    parents_with_chunks = {u.parent for u in shrink.units if u.parent and u.id in minimal}
    return [
        u
        for u in shrink.units
        if u.id in minimal and u.removable and u.id not in parents_with_chunks
    ]


def _closure(units: list[ContextUnit], drop: set[str]) -> set[str]:
    """`drop` plus group mates and (transitively) children."""
    groups = {u.group for u in units if u.id in drop and u.group}
    out = set(drop) | {u.id for u in units if u.group in groups}
    changed = True
    while changed:
        children = {u.id for u in units if u.parent in out} - out
        out |= children
        changed = bool(children)
    return out


def _label(u: ContextUnit, req: LLMRequest) -> str:
    """`msg.0 (task)`, `msg.1 (tool call)`, ... for units that are always kept."""
    if u.kind != "message":
        return u.id
    role = req.messages[int(u.id.split(".")[1])].role
    return f"{u.id} (task)" if role == "user" else f"{u.id} (tool call)"


async def explain(
    req: LLMRequest,
    *,
    runner: TrialRunner,
    shrink: ShrinkResult,
    config: ExplainConfig | None = None,
    substitutions: dict[str, str] | None = None,
    rewriter: Rewriter | None = None,
) -> ExplainResult:
    cfg = config or ExplainConfig()
    fp = shrink.fingerprint
    threshold = shrink.threshold if shrink.threshold is not None else 0.5
    subs = substitutions or {}
    units = shrink.units
    everything = {u.id for u in units}
    minimal = set(shrink.minimal_ids)

    async def measure(r: LLMRequest, trials: int) -> RateEstimate:
        est = await runner.failure_rate(
            r,
            fingerprint=fp,
            min_trials=trials,
            max_trials=trials,
            threshold=threshold,
            early_stop=False,
        )
        if est.partial:
            raise _BudgetStop
        return est

    base = ExplainResult(
        status="ok",
        message="",
        fingerprint=fp,
        baseline=shrink.baseline,
        shrink=shrink,
        not_removable=[_label(u, req) for u in units if not u.removable and u.id in minimal],
    )

    def finish(result: ExplainResult) -> ExplainResult:
        s = runner.stats
        return result.model_copy(
            update={
                "trials": s.trials,
                "cached_trials": s.cache_hits,
                "provider_calls": s.provider_calls,
                "cost_usd": s.cost_usd,
            }
        )

    if shrink.status in ("too_rare", "not_failing"):
        return finish(base.model_copy(update={"status": shrink.status, "message": shrink.message}))
    attributions: list[UnitAttribution] = []
    try:
        p0 = await measure(req, cfg.baseline_trials)
        base = base.model_copy(update={"baseline": p0})
        minimal_est = await measure(recompose(req, minimal, shrink.removal), cfg.trials)
        base = base.model_copy(update={"minimal_estimate": minimal_est})
        for u in explained_units(shrink):
            attributions.append(
                await _attribute(
                    req, u, units, everything, minimal, p0, shrink, cfg, subs, rewriter, measure
                )
            )
        interactions = await _interactions(
            req, attributions, units, everything, p0, shrink, cfg, measure
        )
    except _BudgetStop:
        ranked = _rank(attributions)
        return finish(
            base.model_copy(
                update={
                    "status": "partial",
                    "attributions": ranked,
                    "message": f"budget of ${runner.budget.max_cost:.2f} reached; showing "
                    "the units measured so far. Raise --max-cost to finish.",
                }
            )
        )
    ranked = _rank(attributions)
    culprits = [a for a in ranked if a.culprit]
    if culprits or interactions:
        message = (
            f"{len(culprits)} culprit{'s' if len(culprits) != 1 else ''} found"
            if culprits
            else "no single culprit; see the interaction"
        )
        status: ExplainStatus = "ok"
    else:
        status, message = (
            "no_culprit",
            ("no unit's removal clearly changes the failure rate; try more --trials"),
        )
    return finish(
        base.model_copy(
            update={
                "status": status,
                "message": message,
                "attributions": ranked,
                "interactions": interactions,
            }
        )
    )


async def _attribute(
    req: LLMRequest,
    u: ContextUnit,
    units: list[ContextUnit],
    everything: set[str],
    minimal: set[str],
    p0: RateEstimate,
    shrink: ShrinkResult,
    cfg: ExplainConfig,
    subs: dict[str, str],
    rewriter: Rewriter | None,
    measure: Callable[[LLMRequest, int], Awaitable[RateEstimate]],
) -> UnitAttribution:
    drop = _closure(units, {u.id})
    full_wo = await measure(recompose(req, everything - drop, shrink.removal), cfg.trials)
    min_wo = await measure(recompose(req, minimal - drop, shrink.removal), cfg.trials)
    d, low, high = newcombe(p0, full_wo)
    counterfactual = None
    if u.kind in ("tool_result", "chunk"):
        counterfactual = await _counterfactual(req, u, p0, cfg, subs, rewriter, measure)
    return UnitAttribution(
        rank=0,
        unit_id=u.id,
        kind=u.kind,
        tokens=u.tokens,
        text=unit_text(req, u)[:FULL_TEXT_CHARS],
        without_full=full_wo,
        without_minimal=min_wo,
        attribution=d,
        attribution_low=low,
        attribution_high=high,
        culprit=full_wo.ci_high < p0.ci_low,
        counterfactual=counterfactual,
    )


async def _counterfactual(
    req: LLMRequest,
    u: ContextUnit,
    p0: RateEstimate,
    cfg: ExplainConfig,
    subs: dict[str, str],
    rewriter: Rewriter | None,
    measure: Callable[[LLMRequest, int], Awaitable[RateEstimate]],
) -> Counterfactual:
    original = unit_text(req, u)
    source: SubstitutionSource
    if u.id in subs:
        source, neutral, removed = "provided", subs[u.id], None
    elif rewriter is not None:
        source, neutral, removed = "generated", await rewriter(original), None
    else:
        neutral = neutralize(original)
        source, removed = "heuristic", removed_text(original, neutral) or None
    if neutral.strip() == original.strip():
        return Counterfactual(
            source=source,
            text=neutral[:FULL_TEXT_CHARS],
            note=(
                "the neutral rewrite is identical: the text contains no instructions, so the "
                "failure comes from its content (facts), not from instructions in it"
            ),
        )
    est = await measure(substitute(req, u.id, neutral), cfg.trials)
    effect = p0.p_hat - est.p_hat
    note = (
        "the neutral rewrite fixes it: the instructions in this text cause the failure"
        if est.ci_high < p0.ci_low
        else "the neutral rewrite does not clearly change the failure rate"
    )
    return Counterfactual(
        source=source,
        text=neutral[:FULL_TEXT_CHARS],
        removed=removed,
        estimate=est,
        attribution=effect,
        note=note,
    )


async def _interactions(
    req: LLMRequest,
    attributions: list[UnitAttribution],
    units: list[ContextUnit],
    everything: set[str],
    p0: RateEstimate,
    shrink: ShrinkResult,
    cfg: ExplainConfig,
    measure: Callable[[LLMRequest, int], Awaitable[RateEstimate]],
) -> list[Interaction]:
    culprits = [a.unit_id for a in attributions if a.culprit]
    if len(culprits) >= 2:
        names = " and ".join(culprits)
        return [
            Interaction(
                kind="and",
                units=culprits,
                message=(f"failure needs {names} together: removing any one of them fixes it"),
            )
        ]
    ids = [a.unit_id for a in attributions]
    if culprits or not 1 < len(ids) <= cfg.max_pair_units:
        return []
    found: list[Interaction] = []
    for a, b in combinations(ids, 2):
        drop = _closure(units, {a, b})
        est = await measure(recompose(req, everything - drop, shrink.removal), cfg.trials)
        if est.ci_high < p0.ci_low:
            found.append(
                Interaction(
                    kind="or",
                    units=[a, b],
                    estimate=est,
                    message=(
                        f"either {a} or {b} is enough to cause the failure: removing one alone "
                        "does not fix it, removing both does"
                    ),
                )
            )
    return found


def _rank(attributions: list[UnitAttribution]) -> list[UnitAttribution]:
    order = sorted(
        enumerate(attributions), key=lambda ia: (not ia[1].culprit, -ia[1].attribution, ia[0])
    )
    return [a.model_copy(update={"rank": r}) for r, (_, a) in enumerate(order, start=1)]


def rewrite_request(text: str, model: str) -> LLMRequest:
    """The one LLM call that produces a "generated" neutral rewrite."""
    from agentrelay.providers.base import Message, text_block

    prompt = (
        "Rewrite the text below so it keeps every fact but contains no instructions, "
        "requests or commands addressed to anyone. Return only the rewritten text.\n\n"
        f"<text>\n{text}\n</text>"
    )
    return LLMRequest(
        model=model,
        system=None,
        max_tokens=max(256, len(text) // 2),
        temperature=0.0,
        messages=[Message(role="user", content=[text_block(prompt)])],
    )
