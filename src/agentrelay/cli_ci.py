"""`relay ci`: run all emitted scenarios, report, and set the exit code."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

from agentrelay.testing import ScenarioResult, evaluate_scenario

ICON = {
    "pass": "✅ pass",
    "regression": "❌ regression",
    "error": "⚠️ error",
    "skipped": "⏭️ skipped",
}


class CIOutcome(BaseModel):
    path: str
    results: list[ScenarioResult]
    exit_code: int
    headline: str
    report_path: str
    cost_usd: float


def find_scenarios(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("**/scenarios/*.json")) or sorted(path.glob("*.json"))


def markdown(o: CIOutcome) -> str:
    lines = [
        "## AgentRelay regression scenarios",
        "",
        "| Scenario | Failure rate | Allowed | Result |",
        "|---|---|---|---|",
    ]
    for r in o.results:
        rate = (
            f"{100 * r.estimate.p_hat:.0f}% ({r.estimate.failures}/{r.estimate.trials})"
            if r.estimate
            else "-"
        )
        lines.append(
            f"| `{r.name}` | {rate} | {100 * r.max_failure_rate:.0f}% | {ICON[r.status]} |"
        )
    lines += ["", f"**{o.headline}**", ""]
    for r in o.results:
        if r.status in ("regression", "error", "skipped"):
            lines.append(f"- `{r.name}`: {r.message}")
    return "\n".join(lines) + "\n"


def run_ci(
    path: Path,
    *,
    trials: int | None,
    provider: str | None,
    fake_rules: str | None,
    max_cost: float,
    report_path: Path,
) -> CIOutcome:
    results: list[ScenarioResult] = []
    remaining = max_cost
    for scenario in find_scenarios(path):
        res = evaluate_scenario(
            scenario,
            trials=trials,
            provider=provider,
            fake_rules=fake_rules,
            max_cost=max(remaining, 0.0),
        )
        remaining -= res.cost_usd
        results.append(res)
    counts = {
        s: sum(r.status == s for r in results) for s in ("pass", "regression", "error", "skipped")
    }
    failed = counts["regression"] + counts["error"]
    cost = max_cost - remaining
    headline = (
        f"{counts['regression']} regression(s), {counts['error']} error(s), "
        f"{counts['pass']} passed, {counts['skipped']} skipped; spent ${cost:.4f}"
    )
    outcome = CIOutcome(
        path=str(path),
        results=results,
        exit_code=1 if failed else 0,
        headline=headline,
        report_path=str(report_path),
        cost_usd=cost,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"schema": "agentrelay/v1", **outcome.model_dump(mode="json")}, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(markdown(outcome))
    return outcome
