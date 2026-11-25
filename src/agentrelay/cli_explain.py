"""`relay explain`: rank the units of the minimal context by how much they cause the failure."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from agentrelay.algo.explain import (
    ExplainConfig,
    ExplainResult,
    Rewriter,
    explain,
    rewrite_request,
)
from agentrelay.cli_shrink import Decision, prepare_decision, saved_shrink, shrink_decision
from agentrelay.cli_try import ProviderOptions, TryError
from agentrelay.context.units import Removal


def parse_substitutions(items: list[str]) -> dict[str, str]:
    """`UNIT=FILE` pairs -> {unit id: file contents}."""
    out: dict[str, str] = {}
    for item in items:
        unit_id, sep, path = item.partition("=")
        if not sep or not unit_id or not path:
            raise TryError(f"--substitute expects UNIT=FILE, got {item!r}")
        try:
            out[unit_id] = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise TryError(f"cannot read substitute for {unit_id}: {exc}") from exc
    return out


def run_explain(
    run_ref: str,
    *,
    step_ref: str,
    oracle_ref: str,
    fingerprint: str | None,
    trials: int,
    removal: Removal,
    substitutes: list[str],
    opts: ProviderOptions,
    html: bool = False,
) -> tuple[ExplainResult, Path, Path | None]:
    started = time.monotonic()
    subs = parse_substitutions(substitutes)
    d = prepare_decision(
        run_ref,
        step_ref=step_ref,
        oracle_ref=oracle_ref,
        fingerprint=fingerprint,
        opts=opts,
        command="explain",
    )
    shrink_result = saved_shrink(d, removal)
    if shrink_result is None:
        shrink_result, _ = shrink_decision(d, trials=10, removal=removal)
    unknown = set(subs) - {u.id for u in shrink_result.units}
    if unknown:
        raise TryError(f"--substitute names unknown units: {', '.join(sorted(unknown))}")
    result = asyncio.run(
        explain(
            d.request,
            runner=d.runner,
            shrink=shrink_result,
            config=ExplainConfig(trials=trials),
            substitutions=subs,
            rewriter=None if d.is_fake else _rewriter(d),
        )
    )
    result = result.model_copy(
        update={
            "run_id": d.run.id,
            "step_id": d.step.id,
            "model": d.request.model,
            "oracle": d.oracle.name,
            "oracle_noisy": d.oracle.noisy,
            "elapsed_s": round(time.monotonic() - started, 2),
        }
    )
    path = d.store.save_artifact(d.run.id, f"explain-{d.step.id}.json", result)
    report = None
    if html:
        from agentrelay.report.html import render_report

        report = d.store.save_artifact(d.run.id, "report.html", render_report(result, d.request))
    return result, path, report


def _rewriter(d: Decision) -> Rewriter:
    async def rewrite(text: str) -> str:
        return (await d.runner.sample(rewrite_request(text, d.request.model), 0)).text()

    return rewrite
