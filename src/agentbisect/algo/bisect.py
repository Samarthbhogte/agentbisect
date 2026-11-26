"""First bad step (BLUEPRINT 6.10).

f(k) is the failure rate when steps < k are replayed from the recording and everything from step k
on runs live. f(n_steps) replays the whole run, so it reproduces the recorded failure. Binary
search finds the smallest k whose f(k) still fails, assuming f is roughly monotone; every measured
f(k) is kept so a non-monotone curve is visible in the report.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel

from agentbisect.schema import RateEstimate

BisectStatus = Literal["ok", "fails_from_start", "not_reproduced"]


class CurvePoint(BaseModel):
    k: int
    estimate: RateEstimate


class BisectResult(BaseModel):
    status: BisectStatus
    first_bad: int | None
    last_good: int | None
    n_steps: int
    curve: list[CurvePoint]
    monotone: bool


async def bisect(
    n_steps: int, measure: Callable[[int], Awaitable[RateEstimate]], *, full_curve: bool = False
) -> BisectResult:
    points: dict[int, RateEstimate] = {}

    async def f(k: int) -> RateEstimate:
        if k not in points:
            points[k] = await measure(k)
        return points[k]

    def result(status: BisectStatus, first_bad: int | None, last_good: int | None) -> BisectResult:
        curve = [CurvePoint(k=k, estimate=points[k]) for k in sorted(points)]
        fails = [p.estimate.still_fails() for p in curve]
        monotone = all(not a or b for a, b in pairwise(fails))
        return BisectResult(
            status=status,
            first_bad=first_bad,
            last_good=last_good,
            n_steps=n_steps,
            curve=curve,
            monotone=monotone,
        )

    if not (await f(n_steps)).still_fails():
        return result("not_reproduced", None, None)
    if (await f(0)).still_fails():
        first: int | None = 0
        status: BisectStatus = "fails_from_start"
        good: int | None = None
    else:
        lo, hi = 0, n_steps  # f(lo) passes, f(hi) fails
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if (await f(mid)).still_fails():
                hi = mid
            else:
                lo = mid
        first, good, status = hi, lo, "ok"
    if full_curve:
        for k in range(n_steps + 1):
            await f(k)
    return result(status, first, good)
