"""Wilson score intervals and the sequential stopping rule (BLUEPRINT 6.6)."""

from __future__ import annotations

import math

from agentbisect.schema import Decision, RateEstimate

Z_95 = 1.959963984540054


def wilson(failures: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. No trials -> (0, 1)."""
    if trials <= 0:
        return 0.0, 1.0
    if not 0 <= failures <= trials:
        raise ValueError(f"failures must be in [0, {trials}], got {failures}")
    p = failures / trials
    z2 = z * z
    denom = 1 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denom
    low = 0.0 if failures == 0 else max(0.0, centre - half)
    high = 1.0 if failures == trials else min(1.0, centre + half)
    return low, high


def sequential_decision(
    failures: int, trials: int, *, threshold: float, min_trials: int, max_trials: int
) -> Decision | None:
    """Decision after `trials` trials, or None to keep sampling.

    From `min_trials` on: CI entirely at/above the threshold -> fails; entirely below -> passes.
    At `max_trials` without a decision -> undecided (callers fall back to the point estimate).
    """
    if trials < min_trials:
        return None
    low, high = wilson(failures, trials)
    if low >= threshold:
        return "fails"
    if high < threshold:
        return "passes"
    if trials >= max_trials:
        return "undecided"
    return None


def make_estimate(
    failures: int,
    trials: int,
    *,
    decided: Decision,
    threshold: float,
    slippage: dict[str, int] | None = None,
    partial: bool = False,
    note: str | None = None,
) -> RateEstimate:
    low, high = wilson(failures, trials)
    return RateEstimate(
        failures=failures,
        trials=trials,
        p_hat=failures / trials if trials else 0.0,
        ci_low=low,
        ci_high=high,
        slippage=dict(slippage or {}),
        decided=decided,
        threshold=threshold,
        partial=partial,
        note=note,
    )


def default_threshold(p0: float) -> float:
    """A candidate "still fails" if it fails at least half as often as the original."""
    return max(0.2, 0.5 * p0)
