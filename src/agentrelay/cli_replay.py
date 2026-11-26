"""`relay replay` and `relay bisect` plumbing."""

from __future__ import annotations

import asyncio
from collections import Counter

from pydantic import BaseModel, Field

from agentrelay.algo.bisect import BisectResult, bisect
from agentrelay.config import load_config
from agentrelay.oracles.api import OracleError, acheck, load_oracle
from agentrelay.replay.program import ReplayResult, ToolsMode, replay_program
from agentrelay.schema import RateEstimate, Run, Step
from agentrelay.stats.estimate import make_estimate, sequential_decision
from agentrelay.store.runs import RunStore


def parse_step_index(run: Run, ref: str) -> int:
    """`s0003`, `3` -> 3; also accepts the number of steps (replay everything)."""
    digits = ref[1:] if ref.startswith("s") else ref
    if not digits.isdigit():
        raise ValueError(f"bad step reference {ref!r}: use s0003 or 3")
    k = int(digits)
    if not 0 <= k <= len(run.steps):
        raise ValueError(f"step {ref} is outside run {run.id} (0..{len(run.steps)})")
    return k


class BisectReport(BaseModel):
    run_id: str
    oracle: str
    fingerprint: str
    tools: ToolsMode
    threshold: float
    result: BisectResult
    message: str
    first_bad_step: dict[str, str] | None = None
    explain_step: str | None = None
    trials: int = 0
    errors: dict[str, int] = Field(default_factory=dict)


def describe_step(step: Step) -> str:
    if step.kind == "tool_call":
        result = str((step.response or {}).get("result", ""))
        snippet = " ".join(result.split())[:120]
        args = ", ".join(f"{k}={v!r}" for k, v in step.request.items())
        return f"`{step.name}({args})` returned: {snippet}…"
    if step.kind == "llm_call":
        parts = []
        for b in (step.response or {}).get("content") or []:
            if b.get("type") == "tool_use":
                parts.append(f"called {b.get('tool_name')}({b.get('tool_input')})")
            elif b.get("type") == "text":
                parts.append(" ".join((b.get("text") or "").split())[:120])
        return "the model " + ("; ".join(parts) or "answered")
    return step.name


def run_bisect(
    run_ref: str,
    *,
    oracle_ref: str,
    tools: ToolsMode,
    max_trials: int,
    threshold: float,
    full_curve: bool,
    allow_live_side_effects: bool,
    store: RunStore | None = None,
) -> BisectReport:
    store = store or RunStore()
    cfg = load_config(store.home)
    run = store.load(run_ref)
    oracle = load_oracle(oracle_ref, config_oracles=cfg.oracles)
    if oracle.level != "run":
        raise OracleError(
            f"oracle {oracle.name!r} judges single decisions; bisect replays the "
            "whole run and needs a run-level oracle (@oracle(..., level='run'))"
        )
    recorded = asyncio.run(acheck(oracle, run))
    if recorded is None:
        raise OracleError(f"the recorded run passes oracle {oracle.name!r}; nothing to bisect")
    fingerprint = recorded.fingerprint
    errors: Counter[str] = Counter()
    trials_run = [0]

    async def measure(k: int) -> RateEstimate:
        failures = slipped = 0
        slippage: Counter[str] = Counter()
        for i in range(max_trials):
            res: ReplayResult = replay_program(
                run,
                mode="fork",
                fork_at=k,
                tools=tools,
                allow_live_side_effects=allow_live_side_effects,
                seed=i + 1,
                persist=False,
            )
            trials_run[0] += 1
            if res.error:
                errors[res.error.split(":")[0]] += 1
            failure = await acheck(oracle, res.run) if res.run is not None else None
            if failure is not None and failure.fingerprint == fingerprint:
                failures += 1
            elif failure is not None:
                slipped += 1
                slippage[failure.fingerprint] += 1
            decision = sequential_decision(
                failures,
                i + 1,
                threshold=threshold,
                min_trials=min(3, max_trials),
                max_trials=max_trials,
            )
            if decision is not None:
                return make_estimate(
                    failures, i + 1, decided=decision, threshold=threshold, slippage=dict(slippage)
                )
        return make_estimate(
            failures, max_trials, decided="undecided", threshold=threshold, slippage=dict(slippage)
        )

    result = asyncio.run(bisect(len(run.steps), measure, full_curve=full_curve))
    report = BisectReport(
        run_id=run.id,
        oracle=oracle.name,
        fingerprint=fingerprint,
        tools=tools,
        threshold=threshold,
        result=result,
        message="",
        trials=trials_run[0],
        errors=dict(errors),
    )
    return report.model_copy(update=_narrative(run, result))


def _pct(est: RateEstimate) -> str:
    return f"{100 * est.p_hat:.0f}%"


def _narrative(run: Run, r: BisectResult) -> dict[str, object]:
    by_k = {p.k: p.estimate for p in r.curve}
    if r.status == "not_reproduced":
        return {
            "message": "replaying the whole recorded run does not reproduce the failure with "
            "this oracle, so there is nothing to bisect (if replays raised errors they are "
            "listed below; `relay replay` shows the details)"
        }
    assert r.first_bad is not None
    k = r.first_bad
    bad_step = run.steps[k] if k < len(run.steps) else None
    explain = next((s.id for s in run.steps[k:] if s.kind == "llm_call"), None)
    what = describe_step(bad_step) if bad_step else "the end of the run"
    if r.status == "fails_from_start":
        message = (
            f"The run fails {_pct(by_k[0])} of the time even when everything runs live "
            "from the start: the task itself fails, no single step is to blame."
        )
    else:
        good = r.last_good if r.last_good is not None else 0
        good_id = run.steps[good].id if good < len(run.steps) else "the end"
        bad_id = bad_step.id if bad_step else "the end"
        message = (
            f"Until {good_id} the run is recoverable (fails {_pct(by_k[good])} when "
            f"continued live). From {bad_id} it fails {_pct(by_k[k])}. {bad_id} is the "
            f"first bad step: {what}"
        )
    first = (
        {"id": bad_step.id, "kind": bad_step.kind, "name": bad_step.name, "description": what}
        if bad_step
        else None
    )
    return {"message": message, "first_bad_step": first, "explain_step": explain}
