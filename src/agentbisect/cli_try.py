"""`relay try`: re-run one recorded decision N times and report its failure rate."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from agentbisect.config import Config, load_config
from agentbisect.oracles.api import Oracle, acheck, check, load_oracle
from agentbisect.providers.base import LLMProvider, LLMRequest, LLMResponse
from agentbisect.providers.registry import PROVIDERS, cache_namespace, make_provider
from agentbisect.replay.decision import RequestOverrides, build_request, find_step, request_of
from agentbisect.schema import Failure, RateEstimate, Run, Step
from agentbisect.stats.runner import ANY_FINGERPRINT, Budget, RunnerStats, TrialRunner
from agentbisect.store.cache import TrialCache
from agentbisect.store.runs import RunStore


class TryError(Exception):
    """A user-facing problem with the arguments (bad step, wrong oracle level, ...)."""


@dataclass
class ProviderOptions:
    provider: str | None = None
    fake_rules: str | None = None
    max_cost: float = 2.0
    concurrency: int = 4


@dataclass
class TryResult:
    run: Run
    step: Step
    request: LLMRequest
    oracle: Oracle | None
    fingerprint: str
    recorded_failure: Failure | None
    estimate: RateEstimate
    stats: RunnerStats
    budget: Budget
    responses: Counter[str] = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run.id,
            "step": self.step.id,
            "model": self.request.model,
            "oracle": self.oracle.name if self.oracle else None,
            "noisy_oracle": bool(self.oracle and self.oracle.noisy),
            "fingerprint": self.fingerprint,
            "recorded_failure": self.recorded_failure.model_dump()
            if self.recorded_failure
            else None,
            "estimate": self.estimate.model_dump(),
            "responses": [{"summary": s, "count": n} for s, n in self.responses.most_common()],
            "stats": vars(self.stats),
            "budget": {"max_cost": self.budget.max_cost, "spent": self.budget.spent},
        }


def summarize(resp: LLMResponse, width: int = 80) -> str:
    import json

    parts = [
        f"→ {c.tool_name}({json.dumps(c.tool_input, sort_keys=True)})" for c in resp.tool_calls()
    ]
    text = " ".join(resp.text().split())
    if text:
        parts.insert(0, text)
    out = " ".join(parts) or "(empty)"
    return out if len(out) <= width else out[: width - 1] + "…"


def resolve_provider(step: Step, opts: ProviderOptions, cfg: Config) -> LLMProvider:
    name = opts.provider or step.provider
    if name not in PROVIDERS:
        raise TryError(
            f"step {step.id} was recorded with provider {step.provider!r}; "
            f"pass --provider {'|'.join(PROVIDERS)}"
        )
    return make_provider(name, fake_rules=opts.fake_rules, prices=cfg.prices)


def resolve_step(run: Run, ref: str, oracle: Oracle | None) -> Step:
    if ref != "auto":
        try:
            return find_step(run, ref)
        except KeyError as exc:
            raise TryError(str(exc.args[0])) from exc
    if oracle is None or oracle.level != "decision":
        raise TryError("--step auto needs a decision-level --oracle")
    for step in run.steps:
        if step.kind != "llm_call" or step.response is None:
            continue
        if check(oracle, LLMResponse.model_validate(step.response), request_of(step)):
            return step
    raise TryError(f"no LLM call in run {run.id} fails oracle {oracle.name!r}")


def run_try(
    run_ref: str,
    *,
    step_ref: str,
    oracle_ref: str | None,
    fingerprint: str | None,
    trials: int,
    overrides: RequestOverrides,
    opts: ProviderOptions,
    store: RunStore | None = None,
) -> TryResult:
    store = store or RunStore()
    cfg = load_config(store.home)
    run = store.load(run_ref)
    oracle = load_oracle(oracle_ref, config_oracles=cfg.oracles) if oracle_ref else None
    if oracle is not None and oracle.level != "decision":
        raise TryError(
            f"oracle {oracle.name!r} judges whole runs; `relay try` re-runs a single "
            "decision and needs a decision-level oracle"
        )
    step = resolve_step(run, step_ref, oracle)
    try:
        req = build_request(step, overrides=overrides)
    except ValueError as exc:
        raise TryError(str(exc)) from exc
    provider = resolve_provider(step, opts, cfg)

    recorded = None
    if oracle is not None and step.response is not None:
        recorded = (
            check(oracle, LLMResponse.model_validate(step.response), request_of(step))
            if not _is_async(oracle)
            else None
        )
    target = fingerprint or (recorded.fingerprint if recorded else ANY_FINGERPRINT)

    responses: Counter[str] = Counter()

    async def judge(resp: LLMResponse) -> Failure | None:
        responses[summarize(resp)] += 1
        return await acheck(oracle, resp, req) if oracle else None

    budget = Budget(max_cost=opts.max_cost)
    runner = TrialRunner(
        provider,
        judge,
        cache=TrialCache(store.home / "cache.db", redactor=store.redactor),
        budget=budget,
        concurrency=opts.concurrency,
        cache_namespace=cache_namespace(provider),
    )
    estimate = asyncio.run(
        runner.failure_rate(
            req,
            fingerprint=target,
            min_trials=trials,
            max_trials=trials,
            threshold=0.5,
            early_stop=False,
        )
    )
    return TryResult(
        run=run,
        step=step,
        request=req,
        oracle=oracle,
        fingerprint=target,
        recorded_failure=recorded,
        estimate=estimate,
        stats=runner.stats,
        budget=budget,
        responses=responses,
    )


def _is_async(o: Oracle) -> bool:
    import inspect

    return inspect.iscoroutinefunction(o.fn)
