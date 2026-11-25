"""Shrink one decision to a minimal failing context (BLUEPRINT 6.6 baseline, 6.8 ddmin).

1. Baseline: measure p0 on the original request. If p0 < 0.2 the failure is too rare to diagnose.
2. ddmin with `still_fails(ids)` = recompose -> trial runner (threshold max(0.2, p0/2)).
3. Confirm the minimal set with more trials. If it no longer fails, back off to the last candidate
   that confirms, and flag the result as unstable.

The budget can run out at any point; the result then carries the best candidate found so far.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from agentrelay.algo.ddmin import DDMinStep, ddmin
from agentrelay.context.units import Removal, all_ids, decompose, recompose, total_tokens
from agentrelay.providers.base import LLMRequest
from agentrelay.schema import ContextUnit, RateEstimate
from agentrelay.stats.estimate import default_threshold
from agentrelay.stats.runner import TrialRunner

Status = Literal["ok", "unstable", "too_rare", "partial", "not_failing"]


class ShrinkConfig(BaseModel):
    baseline_trials: int = 10
    min_trials: int = 3
    max_trials: int = 8
    confirm_trials: int = 16
    min_p0: float = 0.2
    removal: Removal = "placeholder"


class ShrinkResult(BaseModel):
    status: Status
    message: str
    fingerprint: str
    removal: Removal
    units: list[ContextUnit]
    original_tokens: int
    removable_units: int
    minimal_ids: list[str]
    minimal_removable_ids: list[str]
    minimal_tokens: int
    baseline: RateEstimate
    threshold: float | None = None
    minimal_estimate: RateEstimate | None = None
    minimal_request: dict[str, object] | None = None
    candidates_tested: int = 0
    trials: int = 0
    cached_trials: int = 0
    provider_calls: int = 0
    cost_usd: float = 0.0
    slippage: dict[str, int] = Field(default_factory=dict)
    log: list[DDMinStep] = Field(default_factory=list)
    run_id: str | None = None
    step_id: str | None = None
    oracle: str | None = None

    @property
    def reduction(self) -> float:
        return 1 - self.minimal_tokens / self.original_tokens if self.original_tokens else 0.0


class _BudgetStop(Exception):
    pass


async def shrink(
    req: LLMRequest, *, runner: TrialRunner, fingerprint: str, config: ShrinkConfig | None = None
) -> ShrinkResult:
    cfg = config or ShrinkConfig()
    units = decompose(req, cfg.removal)
    removable = {u.id for u in units if u.removable}
    slippage: Counter[str] = Counter()

    async def measure(
        r: LLMRequest, *, trials: int | None = None, threshold: float = 0.5
    ) -> RateEstimate:
        if trials is None:
            est = await runner.failure_rate(
                r,
                fingerprint=fingerprint,
                min_trials=cfg.min_trials,
                max_trials=cfg.max_trials,
                threshold=threshold,
            )
        else:
            est = await runner.failure_rate(
                r,
                fingerprint=fingerprint,
                min_trials=trials,
                max_trials=trials,
                threshold=threshold,
                early_stop=False,
            )
        slippage.update(est.slippage)
        return est

    def result(
        status: Status,
        message: str,
        minimal: frozenset[str],
        baseline: RateEstimate,
        *,
        threshold: float | None = None,
        minimal_estimate: RateEstimate | None = None,
        candidates_tested: int = 0,
        log: list[DDMinStep] | None = None,
        minimal_request: dict[str, object] | None = None,
    ) -> ShrinkResult:
        stats = runner.stats
        return ShrinkResult(
            status=status,
            message=message,
            fingerprint=fingerprint,
            removal=cfg.removal,
            units=units,
            original_tokens=total_tokens(units),
            removable_units=len(removable),
            minimal_ids=sorted(minimal),
            minimal_removable_ids=sorted(minimal & removable),
            minimal_tokens=total_tokens(units, minimal),
            baseline=baseline,
            threshold=threshold,
            minimal_estimate=minimal_estimate,
            minimal_request=minimal_request,
            candidates_tested=candidates_tested,
            trials=stats.trials,
            cached_trials=stats.cache_hits,
            provider_calls=stats.provider_calls,
            cost_usd=stats.cost_usd,
            slippage=dict(slippage),
            log=log or [],
        )

    everything = all_ids(units)
    baseline = await measure(req, trials=cfg.baseline_trials)
    if baseline.partial:
        return result(
            "partial", baseline.note or "budget exhausted during the baseline", everything, baseline
        )
    p0 = baseline.p_hat
    if baseline.failures == 0:
        return result(
            "not_failing",
            f"the request never failed in {baseline.trials} trials "
            f"with fingerprint {fingerprint!r}; nothing to shrink",
            everything,
            baseline,
        )
    if p0 < cfg.min_p0:
        return result(
            "too_rare",
            f"baseline failure rate {p0:.0%} is below {cfg.min_p0:.0%}: too "
            "rare to diagnose reliably at this budget. Use more --trials or a lower "
            "temperature.",
            everything,
            baseline,
        )
    threshold = default_threshold(p0)
    best = [everything]

    async def still_fails(keep: frozenset[str]) -> bool:
        est = await measure(recompose(req, keep, cfg.removal), threshold=threshold)
        if est.partial:
            raise _BudgetStop
        return est.still_fails()

    def accept(keep: frozenset[str]) -> None:
        best.append(keep)

    try:
        found = await ddmin(units, still_fails, on_accept=accept)
    except _BudgetStop:
        current = best[-1]
        return result(
            "partial",
            "budget exhausted during the search; showing the smallest "
            "failing context found so far (unconfirmed)",
            current,
            baseline,
            threshold=threshold,
            minimal_request=recompose(req, current, cfg.removal).to_dict(),
        )
    minimal = frozenset(found.minimal_ids)
    candidates = [minimal, *(frozenset(a) for a in reversed(found.accepted)), everything]
    status: Status = "ok"
    message = "minimal failing context found and confirmed"
    confirmed: RateEstimate | None = None
    for candidate in dict.fromkeys(candidates):
        est = await measure(
            recompose(req, candidate, cfg.removal), trials=cfg.confirm_trials, threshold=threshold
        )
        if est.partial:
            return result(
                "partial",
                "budget exhausted while confirming the minimal context",
                candidate,
                baseline,
                threshold=threshold,
                minimal_estimate=est,
                candidates_tested=found.candidates_tested,
                log=found.log,
                minimal_request=recompose(req, candidate, cfg.removal).to_dict(),
            )
        if est.still_fails():
            minimal, confirmed = candidate, est
            break
        status = "unstable"
        message = (
            "the smallest candidate did not fail on confirmation; backed off to the last "
            "candidate that did"
        )
    return result(
        status,
        message,
        minimal,
        baseline,
        threshold=threshold,
        minimal_estimate=confirmed,
        candidates_tested=found.candidates_tested,
        log=found.log,
        minimal_request=recompose(req, minimal, cfg.removal).to_dict(),
    )
