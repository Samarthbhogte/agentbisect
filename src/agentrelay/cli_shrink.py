"""`relay shrink`: find the smallest context that still reproduces a failing decision."""

from __future__ import annotations

import asyncio
from pathlib import Path

from agentrelay.algo.shrink import ShrinkConfig, ShrinkResult, shrink
from agentrelay.cli_try import ProviderOptions, TryError, resolve_provider, resolve_step
from agentrelay.config import load_config
from agentrelay.context.units import Removal
from agentrelay.oracles.api import acheck, check, load_oracle
from agentrelay.providers.base import LLMResponse
from agentrelay.providers.registry import cache_namespace
from agentrelay.replay.decision import request_of
from agentrelay.schema import Failure
from agentrelay.stats.runner import Budget, TrialRunner
from agentrelay.store.cache import TrialCache
from agentrelay.store.runs import RunStore


def run_shrink(
    run_ref: str,
    *,
    step_ref: str,
    oracle_ref: str,
    fingerprint: str | None,
    trials: int,
    removal: Removal,
    opts: ProviderOptions,
    store: RunStore | None = None,
) -> tuple[ShrinkResult, Path]:
    store = store or RunStore()
    cfg = load_config(store.home)
    run = store.load(run_ref)
    oracle = load_oracle(oracle_ref, config_oracles=cfg.oracles)
    if oracle.level != "decision":
        raise TryError(
            f"oracle {oracle.name!r} judges whole runs; `relay shrink` works on one "
            "decision and needs a decision-level oracle"
        )
    step = resolve_step(run, step_ref, oracle)
    try:
        req = request_of(step)
    except ValueError as exc:
        raise TryError(str(exc)) from exc
    target = fingerprint
    if target is None and step.response is not None:
        recorded = check(oracle, LLMResponse.model_validate(step.response), req)
        target = recorded.fingerprint if recorded else None
    if target is None:
        raise TryError(
            f"the recorded response of {step.id} passes the oracle, so there is no "
            "failure to shrink; pass --fingerprint to name the failure to look for"
        )
    provider = resolve_provider(step, opts, cfg)

    async def judge(resp: LLMResponse) -> Failure | None:
        return await acheck(oracle, resp, req)

    runner = TrialRunner(
        provider,
        judge,
        cache=TrialCache(store.home / "cache.db", redactor=store.redactor),
        budget=Budget(max_cost=opts.max_cost),
        concurrency=opts.concurrency,
        cache_namespace=cache_namespace(provider),
    )
    result = asyncio.run(
        shrink(
            req,
            runner=runner,
            fingerprint=target,
            config=ShrinkConfig(baseline_trials=trials, removal=removal),
        )
    )
    result = result.model_copy(update={"run_id": run.id, "step_id": step.id, "oracle": oracle.name})
    path = store.save_artifact(run.id, f"shrink-{step.id}.json", result)
    return result, path
