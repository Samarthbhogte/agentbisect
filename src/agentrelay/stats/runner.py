"""TrialRunner: repeated trials with a response cache, a cost budget, concurrency and retries.

Trial *i* always uses ``seed = sample_idx = i``, so re-running the same request is served from the
cache and costs nothing.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from agentrelay.providers.base import LLMProvider, LLMRequest, LLMResponse
from agentrelay.schema import Failure, RateEstimate
from agentrelay.stats.estimate import make_estimate, sequential_decision

DecisionJudge = Callable[[LLMResponse], Awaitable[Failure | None]]
ANY_FINGERPRINT = "*"
DEFAULT_MAX_COST = 2.00


class ResponseCache(Protocol):
    def get(self, key: str, sample_idx: int) -> LLMResponse | None: ...
    def put(self, key: str, sample_idx: int, response: LLMResponse) -> LLMResponse: ...


class BudgetExhausted(Exception):
    """Raised internally when the next trial would exceed the budget."""


@dataclass
class Budget:
    """Spending cap for one command. `spent` includes only real provider calls."""

    max_cost: float = DEFAULT_MAX_COST
    spent: float = 0.0
    exhausted: bool = False

    def can_afford(self, estimate: float) -> bool:
        return self.spent + estimate <= self.max_cost + 1e-12

    def charge(self, amount: float) -> None:
        self.spent += amount


@dataclass
class RunnerStats:
    trials: int = 0
    cache_hits: int = 0
    provider_calls: int = 0
    retries: int = 0
    cost_usd: float = 0.0


@dataclass
class _Trial:
    idx: int
    failure: Failure | None


@dataclass
class _Tally:
    failures: int = 0
    trials: int = 0
    slippage: Counter[str] = field(default_factory=Counter)


def is_retryable(exc: BaseException) -> bool:
    """An explicit `retryable` flag wins; otherwise HTTP 429 and 5xx are retried."""
    flag = getattr(exc, "retryable", None)
    if isinstance(flag, bool):
        return flag
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


class TrialRunner:
    def __init__(
        self,
        provider: LLMProvider,
        judge: DecisionJudge,
        *,
        cache: ResponseCache | None,
        budget: Budget,
        concurrency: int = 4,
        cache_namespace: str = "",
        max_retries: int = 5,
        backoff_base: float = 0.5,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.provider = provider
        self.judge = judge
        self.cache = cache
        self.budget = budget
        self.cache_namespace = cache_namespace
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._concurrency = max(1, concurrency)
        self.stats = RunnerStats()

    def cache_key(self, req: LLMRequest) -> str:
        from agentrelay.store.cache import trial_cache_key

        return trial_cache_key(req, namespace=self.cache_namespace)

    async def sample(self, req: LLMRequest, sample_idx: int) -> LLMResponse:
        """One response for `(req, sample_idx)`: from the cache, or one (budgeted) provider call."""
        key = self.cache_key(req)
        if self.cache is not None:
            cached = self.cache.get(key, sample_idx)
            if cached is not None:
                self.stats.cache_hits += 1
                return cached
        async with self._semaphore:
            estimate = self.provider.estimate_cost(req)
            if not self.budget.can_afford(estimate):
                self.budget.exhausted = True
                raise BudgetExhausted
            self.budget.charge(estimate)  # reserve now so concurrent trials can't overspend
            resp = await self._call_with_retries(req, sample_idx)
            actual = _actual_cost(self.provider, req, resp, estimate)
            self.budget.charge(actual - estimate)
            self.stats.cost_usd += actual
            self.stats.provider_calls += 1
        if self.cache is not None:
            resp = self.cache.put(key, sample_idx, resp)
        return resp

    async def _call_with_retries(self, req: LLMRequest, seed: int) -> LLMResponse:
        attempt = 0
        while True:
            try:
                return await self.provider.complete(req, seed=seed)
            except Exception as exc:
                if attempt >= self.max_retries or not is_retryable(exc):
                    raise
                attempt += 1
                self.stats.retries += 1
                delay = self.backoff_base * 2 ** (attempt - 1)
                await self._sleep(delay + random.uniform(0, delay))

    async def _trial(self, req: LLMRequest, idx: int) -> _Trial:
        resp = await self.sample(req, idx)
        return _Trial(idx, await self.judge(resp))

    async def failure_rate(
        self,
        req: LLMRequest,
        *,
        fingerprint: str,
        min_trials: int = 3,
        max_trials: int = 8,
        threshold: float = 0.5,
        early_stop: bool = True,
    ) -> RateEstimate:
        """Estimate how often `req` fails with `fingerprint` (`"*"` = any fingerprint).

        Trials run in batches of `concurrency` but are judged in index order, so the stopping
        point is the same as running them one at a time.
        """
        min_trials = min(min_trials, max_trials)
        tally = _Tally()
        next_idx = 0
        while next_idx < max_trials:
            batch = range(next_idx, min(next_idx + self._concurrency, max_trials))
            results = await asyncio.gather(
                *(self._trial(req, i) for i in batch), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BudgetExhausted):
                    return self._partial(tally, threshold, max_trials)
                if isinstance(result, BaseException):
                    raise result
                self._count(tally, result.failure, fingerprint)
                decision = sequential_decision(
                    tally.failures,
                    tally.trials,
                    threshold=threshold,
                    min_trials=min_trials if early_stop else max_trials,
                    max_trials=max_trials,
                )
                if decision is not None:
                    return make_estimate(
                        tally.failures,
                        tally.trials,
                        decided=decision,
                        threshold=threshold,
                        slippage=dict(tally.slippage),
                    )
            next_idx = batch.stop
        return make_estimate(
            tally.failures,
            tally.trials,
            decided="undecided",
            threshold=threshold,
            slippage=dict(tally.slippage),
        )

    def _count(self, tally: _Tally, failure: Failure | None, fingerprint: str) -> None:
        tally.trials += 1
        self.stats.trials += 1
        if failure is None:
            return
        if fingerprint in (ANY_FINGERPRINT, failure.fingerprint):
            tally.failures += 1
        else:
            tally.slippage[failure.fingerprint] += 1

    def _partial(self, tally: _Tally, threshold: float, planned: int) -> RateEstimate:
        note = (
            f"budget of ${self.budget.max_cost:.2f} reached after {tally.trials} of {planned} "
            f"trials (spent ${self.budget.spent:.4f}); result is partial. Raise --max-cost to "
            "finish."
        )
        return make_estimate(
            tally.failures,
            tally.trials,
            decided="undecided",
            threshold=threshold,
            slippage=dict(tally.slippage),
            partial=True,
            note=note,
        )


def _actual_cost(
    provider: LLMProvider, req: LLMRequest, resp: LLMResponse, estimate: float
) -> float:
    """`provider.cost(resp, req)` when the provider reports usage; otherwise the estimate."""
    cost_fn = getattr(provider, "cost", None)
    if callable(cost_fn):
        value = cost_fn(resp, req)
        if isinstance(value, int | float):
            return float(value)
    return estimate
