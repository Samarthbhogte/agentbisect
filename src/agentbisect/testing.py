"""Run regression scenarios emitted by `relay emit-test` (BLUEPRINT 6.11).

A scenario is the minimal failing request of a diagnosed decision, its oracle and fingerprint, and
the failure rate a fixed agent must stay under. `run_scenario` re-runs it `trials` times and passes
when the failure rate is at most `max_failure_rate`.

The **current** system prompt and tool specs are swapped in when configured (`system_prompt` /
`tools` in `agentbisect.toml`, or the references stored in the scenario), so after you fix the
prompt the test checks the fix. Without them the recorded request is replayed as is
(useful when upgrading models).

Provider: `AGENTBISECT_PROVIDER` (`anthropic`, `openai`, `gemini`, `fake`), else the provider the
step was recorded with. Without its API key the test is skipped, unless
`AGENTBISECT_REQUIRE_LIVE=1`.
`AGENTBISECT_PROVIDER=fake` uses `AGENTBISECT_FAKE_RULES` (or the rules file stored at emit time).
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentbisect.schema import RateEstimate

ScenarioStatus = Literal["pass", "regression", "skipped", "error"]
KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class ScenarioSkipped(Exception):
    """The scenario cannot run here (no API key); not a failure."""


class Scenario(BaseModel):
    schema_: str = Field(default="agentbisect/v1", alias="schema")
    kind: Literal["scenario"] = "scenario"
    name: str
    run_id: str
    step_id: str
    oracle: str
    fingerprint: str
    request: dict[str, Any]
    culprits: list[str] = Field(default_factory=list)
    trials: int = 10
    max_failure_rate: float = 0.1
    baseline: RateEstimate | None = None
    minimal_estimate: RateEstimate | None = None
    recorded_provider: str | None = None
    fake_rules: str | None = None
    system_prompt: str | None = None
    tools: str | None = None
    created_at: str | None = None
    agentbisect: str | None = None

    model_config = {"populate_by_name": True}


class ScenarioResult(BaseModel):
    name: str
    path: str
    status: ScenarioStatus
    passed: bool
    estimate: RateEstimate | None = None
    max_failure_rate: float
    trials: int
    provider: str | None = None
    model: str | None = None
    overrides: list[str] = Field(default_factory=list)
    message: str = ""
    provider_calls: int = 0
    cached_trials: int = 0
    cost_usd: float = 0.0

    def summary(self) -> str:
        if self.estimate is None:
            return f"{self.name}: {self.status}: {self.message}"
        e = self.estimate
        rate = (
            f"{100 * e.p_hat:.0f}% [{100 * e.ci_low:.0f}%-{100 * e.ci_high:.0f}%] "
            f"({e.failures}/{e.trials})"
        )
        verdict = "within" if self.passed else "ABOVE"
        extra = f"; overrides: {', '.join(self.overrides)}" if self.overrides else ""
        return (
            f"{self.name}: failure rate {rate} is {verdict} the allowed "
            f"{100 * self.max_failure_rate:.0f}% on {self.provider}/{self.model}{extra}. "
            f"{self.message}"
        ).strip()


# ---- locating things ----------------------------------------------------------------------
def project_root(start: Path) -> Path:
    """Nearest parent with `agentbisect.toml`, else with `pyproject.toml`, else `start`."""
    start = start.resolve()
    for marker in ("agentbisect.toml", "pyproject.toml"):
        for directory in (start, *start.parents):
            if (directory / marker).is_file():
                return directory
    return start


def _resolve_ref(ref: str, root: Path) -> str:
    """`file.py:NAME` relative to the cwd or the project root; `module:NAME` importable from
    the project root."""
    target, _, attr = ref.rpartition(":")
    if target.endswith(".py"):
        path = Path(target)
        if not path.is_absolute() and not path.exists() and (root / path).exists():
            return f"{root / path}:{attr}"
        return ref
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return ref


def _resolve_file(ref: str, root: Path) -> str:
    path = Path(ref)
    if not path.is_absolute() and not path.exists() and (root / path).exists():
        return str(root / path)
    return ref


def _caller_dir() -> Path:
    here = Path(__file__).resolve()
    for frame in inspect.stack()[1:]:
        if Path(frame.filename).resolve() != here:
            return Path(frame.filename).resolve().parent
    return Path.cwd()


def scenario_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return ((base or _caller_dir()) / p).resolve()


def load_scenario(path: Path) -> Scenario:
    import json

    return Scenario.model_validate(json.loads(path.read_text(encoding="utf-8")))


# ---- overrides ----------------------------------------------------------------------------
def _tool_specs(value: Any) -> list[Any]:
    from agentbisect.providers.base import ToolSpec

    specs = []
    for t in value:
        if isinstance(t, ToolSpec):
            specs.append(t)
        elif isinstance(t, dict) and t.get("type") == "function" and "function" in t:
            fn = t["function"]
            specs.append(
                ToolSpec(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    input_schema=fn.get("parameters", {}),
                )
            )
        else:
            specs.append(
                ToolSpec(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("input_schema", t.get("parameters", {})),
                )
            )
    return specs


def apply_overrides(
    request: Any, *, system_ref: str | None, tools_ref: str | None, model: str | None, root: Path
) -> tuple[Any, list[str]]:
    from agentbisect._loader import load_attr

    applied: list[str] = []
    update: dict[str, Any] = {}
    if system_ref:
        value = load_attr(_resolve_ref(system_ref, root))
        if not isinstance(value, str):
            raise TypeError(f"system_prompt {system_ref!r} is not a string")
        update["system"] = value
        applied.append(f"system={system_ref}")
    if tools_ref:
        update["tools"] = _tool_specs(load_attr(_resolve_ref(tools_ref, root)))
        applied.append(f"tools={tools_ref}")
    if model:
        update["model"] = model
        applied.append(f"model={model}")
    return (request.model_copy(update=update, deep=True) if update else request), applied


# ---- running ------------------------------------------------------------------------------
def evaluate_scenario(
    path: str | Path,
    *,
    trials: int | None = None,
    max_failure_rate: float | None = None,
    provider: str | None = None,
    fake_rules: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    tools: str | None = None,
    max_cost: float | None = None,
    base: Path | None = None,
) -> ScenarioResult:
    """Run one scenario and return its result (never raises for skips or regressions)."""
    from agentbisect.config import load_config
    from agentbisect.oracles.api import acheck, load_oracle
    from agentbisect.providers.base import LLMRequest, LLMResponse
    from agentbisect.providers.registry import cache_namespace, make_provider
    from agentbisect.schema import Failure
    from agentbisect.stats.runner import Budget, TrialRunner
    from agentbisect.store.cache import TrialCache
    from agentbisect.store.runs import resolve_home

    file = scenario_path(path, base)
    sc = load_scenario(file)
    n = trials or sc.trials
    limit = sc.max_failure_rate if max_failure_rate is None else max_failure_rate
    root = project_root(file.parent)
    cfg = load_config(root / ".agentbisect")
    name = provider or os.environ.get("AGENTBISECT_PROVIDER") or sc.recorded_provider or "anthropic"

    def result(status: ScenarioStatus, message: str, **kw: Any) -> ScenarioResult:
        return ScenarioResult(
            name=sc.name,
            path=str(file),
            status=status,
            passed=status in ("pass", "skipped"),
            max_failure_rate=limit,
            trials=n,
            provider=name,
            message=message,
            **kw,
        )

    if name in KEY_ENV and not os.environ.get(KEY_ENV[name]):
        message = f"{KEY_ENV[name]} is not set, so the {name} scenario cannot run"
        if os.environ.get("AGENTBISECT_REQUIRE_LIVE") == "1":
            return result("error", message + " (AGENTBISECT_REQUIRE_LIVE=1)")
        return result("skipped", message + "; set AGENTBISECT_PROVIDER=fake to run offline")
    rules = fake_rules or os.environ.get("AGENTBISECT_FAKE_RULES") or sc.fake_rules
    try:
        llm = make_provider(
            name, fake_rules=_resolve_file(rules, root) if rules else None, prices=cfg.prices
        )
        request, applied = apply_overrides(
            LLMRequest.model_validate(sc.request),
            system_ref=system_prompt or sc.system_prompt or cfg.system_prompt,
            tools_ref=tools or sc.tools or cfg.tools,
            model=model or os.environ.get("AGENTBISECT_MODEL"),
            root=root,
        )
        oracle = load_oracle(_resolve_ref(sc.oracle, root), config_oracles=cfg.oracles)
    except Exception as exc:
        return result("error", f"cannot set up scenario: {type(exc).__name__}: {exc}")

    async def judge(resp: LLMResponse) -> Failure | None:
        return await acheck(oracle, resp, request)

    home = resolve_home(root)
    runner = TrialRunner(
        llm,
        judge,
        cache=TrialCache(home / "cache.db"),
        budget=Budget(
            max_cost=max_cost
            if max_cost is not None
            else float(os.environ.get("AGENTBISECT_MAX_COST", "1.0"))
        ),
        cache_namespace=cache_namespace(llm),
    )
    try:
        est = asyncio.run(
            runner.failure_rate(
                request,
                fingerprint=sc.fingerprint,
                min_trials=n,
                max_trials=n,
                threshold=max(limit, 1e-9),
                early_stop=False,
            )
        )
    except Exception as exc:
        return result(
            "error", f"{type(exc).__name__}: {exc}", model=request.model, overrides=applied
        )
    stats = dict(
        provider_calls=runner.stats.provider_calls,
        cached_trials=runner.stats.cache_hits,
        cost_usd=runner.stats.cost_usd,
    )
    if est.partial:
        return result(
            "error",
            est.note or "budget exhausted",
            estimate=est,
            model=request.model,
            overrides=applied,
            **stats,
        )
    ok = est.p_hat <= limit
    message = (
        "the failure no longer reproduces"
        if ok
        else "the failure still reproduces: the bug is not fixed (or has come back)"
    )
    return result(
        "pass" if ok else "regression",
        message,
        estimate=est,
        model=request.model,
        overrides=applied,
        **stats,
    )


def run_scenario(
    path: str | Path,
    *,
    trials: int | None = None,
    max_failure_rate: float | None = None,
    **kwargs: Any,
) -> ScenarioResult:
    """Pytest entry point used by generated tests. Skips (`pytest.skip`) when it cannot run;
    raises on setup errors; returns the result for `assert result.passed, result.summary()`."""
    res = evaluate_scenario(
        path, trials=trials, max_failure_rate=max_failure_rate, base=_caller_dir(), **kwargs
    )
    if res.status == "skipped":
        if "pytest" in sys.modules:
            import pytest

            pytest.skip(res.message)
        raise ScenarioSkipped(res.message)
    if res.status == "error":
        raise RuntimeError(f"scenario {res.name} could not run: {res.message}")
    return res
