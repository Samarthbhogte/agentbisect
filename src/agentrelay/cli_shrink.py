"""`relay shrink` and the setup shared with `relay explain`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from agentrelay.algo.shrink import ShrinkConfig, ShrinkResult, shrink
from agentrelay.cli_try import ProviderOptions, TryError, resolve_provider, resolve_step
from agentrelay.config import load_config
from agentrelay.context.units import Removal
from agentrelay.oracles.api import Oracle, acheck, check, load_oracle
from agentrelay.providers.base import LLMProvider, LLMRequest, LLMResponse
from agentrelay.providers.fake import FakeLLM
from agentrelay.providers.registry import cache_namespace
from agentrelay.replay.decision import request_of
from agentrelay.schema import Failure, Run, Step
from agentrelay.stats.runner import Budget, TrialRunner
from agentrelay.store.cache import TrialCache
from agentrelay.store.runs import RunStore


@dataclass
class Decision:
    """Everything needed to run trials on one recorded decision."""

    store: RunStore
    run: Run
    step: Step
    request: LLMRequest
    oracle: Oracle
    fingerprint: str
    provider: LLMProvider
    runner: TrialRunner

    @property
    def is_fake(self) -> bool:
        return isinstance(self.provider, FakeLLM)


def prepare_decision(
    run_ref: str,
    *,
    step_ref: str,
    oracle_ref: str,
    fingerprint: str | None,
    opts: ProviderOptions,
    store: RunStore | None = None,
    command: str = "shrink",
) -> Decision:
    store = store or RunStore()
    cfg = load_config(store.home)
    run = store.load(run_ref)
    oracle = load_oracle(oracle_ref, config_oracles=cfg.oracles)
    if oracle.level != "decision":
        raise TryError(
            f"oracle {oracle.name!r} judges whole runs; `relay {command}` works on one "
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
            "failure to analyse; pass --fingerprint to name the failure to look for"
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
    return Decision(store, run, step, req, oracle, target, provider, runner)


def shrink_decision(d: Decision, *, trials: int, removal: Removal) -> tuple[ShrinkResult, Path]:
    result = asyncio.run(
        shrink(
            d.request,
            runner=d.runner,
            fingerprint=d.fingerprint,
            config=ShrinkConfig(baseline_trials=trials, removal=removal),
        )
    )
    result = result.model_copy(
        update={"run_id": d.run.id, "step_id": d.step.id, "oracle": d.oracle.name}
    )
    path = d.store.save_artifact(d.run.id, f"shrink-{d.step.id}.json", result)
    return result, path


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
    d = prepare_decision(
        run_ref,
        step_ref=step_ref,
        oracle_ref=oracle_ref,
        fingerprint=fingerprint,
        opts=opts,
        store=store,
    )
    return shrink_decision(d, trials=trials, removal=removal)


def saved_shrink(d: Decision, removal: Removal) -> ShrinkResult | None:
    """A previous shrink of this step with the same fingerprint and removal mode, if any."""
    try:
        data = d.store.load_artifact(d.run.id, f"shrink-{d.step.id}.json")
    except (FileNotFoundError, KeyError):
        return None
    data.pop("schema", None)
    try:
        result = ShrinkResult.model_validate(data)
    except ValueError:
        return None
    if (result.fingerprint, result.removal) != (d.fingerprint, removal):
        return None
    return result if result.status in ("ok", "unstable") else None
