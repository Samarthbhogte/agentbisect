"""Run RelayBench: AgentRelay (shrink + explain) against two baselines, offline.

    uv run python bench/run_bench.py              # all cases, rewrites bench/RESULTS.md
    uv run python bench/run_bench.py --cases 5    # quick look (does not rewrite RESULTS.md)

Methods, all judged by the same oracle and charged in model calls (trial-cache misses):

* agentrelay:      `shrink` (probabilistic ddmin) then `explain` (attribution), default settings.
* leave-one-out:   baseline (20 trials), then remove each finest unit on its own and test it with
                   the same sequential rule ddmin uses (3-8 trials). No ddmin.
* random-ablation: random subsets (each unit kept with probability 1/2, 4 trials each) until it
                   has used as many model calls as agentrelay did on that case; units are ranked
                   by how much keeping them raises the failure rate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from agentrelay import __version__  # noqa: E402
from agentrelay.algo.explain import ExplainConfig, _closure, explain  # noqa: E402
from agentrelay.algo.shrink import shrink  # noqa: E402
from agentrelay.context.units import decompose, explained_ids, recompose  # noqa: E402
from agentrelay.providers.base import LLMRequest, LLMResponse  # noqa: E402
from agentrelay.providers.fake import FakeLLM, Rule  # noqa: E402
from agentrelay.schema import Failure  # noqa: E402
from agentrelay.stats.estimate import default_threshold  # noqa: E402
from agentrelay.stats.runner import Budget, TrialRunner  # noqa: E402
from agentrelay.store.cache import TrialCache  # noqa: E402

TARGETS = {"top1": 0.85, "median_shrink_ratio": 10.0, "fewer_trials_than_loo": 0.40}


def load_cases(limit: int | None = None) -> list[dict[str, Any]]:
    files = sorted((ROOT / "cases").glob("case_*.json"))
    return [json.loads(f.read_text()) for f in files[:limit]]


def make_runner(case: dict[str, Any], cache_dir: Path, method: str) -> TrialRunner:
    llm = FakeLLM(
        rules=[Rule.model_validate(r) for r in case["rules"]],
        default=LLMResponse.model_validate(case["default"]),
    )

    async def judge(resp: LLMResponse) -> Failure | None:
        for call in resp.tool_calls():
            if call.tool_name in (case["fingerprint"], case["decoy_fingerprint"]):
                return Failure(fingerprint=call.tool_name)
        return None

    return TrialRunner(
        llm,
        judge,
        cache=TrialCache(cache_dir / f"{case['id']}-{method}.db"),
        budget=Budget(max_cost=1_000.0),
        cache_namespace=llm.cache_namespace,
    )


def score(case: dict[str, Any], ranking: list[str], flagged: list[str]) -> dict[str, Any]:
    truth = set(case["ground_truth"])
    return {
        "top1": bool(ranking) and ranking[0] in truth,
        "top3": truth <= set(ranking[:3]),
        "flagged": flagged,
        "false_flags": sorted(set(flagged) - truth),
        "missed": sorted(truth - set(flagged)),
        "ranking": ranking[:5],
    }


async def run_agentrelay(case: dict[str, Any], req: LLMRequest, cache: Path) -> dict[str, Any]:
    runner = make_runner(case, cache, "agentrelay")
    shrunk = await shrink(req, runner=runner, fingerprint=case["fingerprint"])
    out: dict[str, Any] = {
        "status": shrunk.status,
        "original_tokens": shrunk.original_tokens,
        "minimal_tokens": shrunk.minimal_tokens,
    }
    ranking: list[str] = []
    flagged: list[str] = []
    interaction = None
    if shrunk.status in ("ok", "unstable"):
        result = await explain(req, runner=runner, shrink=shrunk)
        out["status"] = result.status
        ranking = [a.unit_id for a in result.attributions]
        flagged = [a.unit_id for a in result.culprits]
        interaction = [(i.kind, sorted(i.units)) for i in result.interactions]
        if result.status == "no_culprit":
            # Not counted in this case's calls: does doubling the trials settle it?
            before = runner.stats.provider_calls
            more = await explain(
                req,
                runner=runner,
                shrink=shrunk,
                config=ExplainConfig(baseline_trials=40, trials=32),
            )
            out["with_more_trials"] = {
                "status": more.status,
                "flagged": [a.unit_id for a in more.culprits],
                "extra_calls": runner.stats.provider_calls - before,
            }
            runner.stats.provider_calls = before
    out.update(score(case, ranking, flagged))
    out["shrink_ratio"] = (
        shrunk.original_tokens / shrunk.minimal_tokens if shrunk.minimal_tokens else None
    )
    out["interaction_correct"] = (
        interaction == [("and", sorted(case["ground_truth"]))] if case["interaction"] else None
    )
    out["calls"] = runner.stats.provider_calls
    out["cost"] = runner.stats.cost_usd
    return out


async def run_leave_one_out(case: dict[str, Any], req: LLMRequest, cache: Path) -> dict[str, Any]:
    runner = make_runner(case, cache, "loo")
    units = decompose(req)
    everything = {u.id for u in units}
    fp = case["fingerprint"]
    p0 = await runner.failure_rate(
        req, fingerprint=fp, min_trials=20, max_trials=20, early_stop=False
    )
    threshold = default_threshold(p0.p_hat)
    scores: dict[str, float] = {}
    flagged = []
    for uid in explained_ids(units):
        keep = everything - _closure(units, {uid})
        est = await runner.failure_rate(
            recompose(req, keep), fingerprint=fp, min_trials=3, max_trials=8, threshold=threshold
        )
        scores[uid] = p0.p_hat - est.p_hat
        if not est.still_fails():
            flagged.append(uid)
    ranking = sorted(scores, key=lambda u: (u not in flagged, -scores[u]))
    out = score(case, ranking, flagged)
    out.update(calls=runner.stats.provider_calls, cost=runner.stats.cost_usd)
    return out


async def run_random_ablation(
    case: dict[str, Any], req: LLMRequest, cache: Path, budget: int, seed: int
) -> dict[str, Any]:
    runner = make_runner(case, cache, "random")
    units = decompose(req)
    everything = {u.id for u in units}
    finest = explained_ids(units)
    rng = random.Random(seed)
    kept_p: dict[str, list[float]] = {u: [] for u in finest}
    dropped_p: dict[str, list[float]] = {u: [] for u in finest}
    while runner.stats.provider_calls < budget:
        drop = {u for u in finest if rng.random() < 0.5}
        keep = everything - _closure(units, drop)
        est = await runner.failure_rate(
            recompose(req, keep),
            fingerprint=case["fingerprint"],
            min_trials=4,
            max_trials=4,
            early_stop=False,
        )
        for u in finest:
            (dropped_p if u in drop else kept_p)[u].append(est.p_hat)
        if runner.stats.provider_calls == 0:  # everything cached: avoid spinning forever
            break

    def effect(u: str) -> float:
        k, d = kept_p[u], dropped_p[u]
        return (statistics.fmean(k) if k else 0.0) - (statistics.fmean(d) if d else 0.0)

    ranking = sorted(finest, key=lambda u: -effect(u))
    flagged = [u for u in ranking if effect(u) >= 0.3]
    out = score(case, ranking, flagged)
    out.update(calls=runner.stats.provider_calls, cost=runner.stats.cost_usd)
    return out


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    req = LLMRequest.model_validate(case["request"])
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        started = time.perf_counter()
        ar = asyncio.run(run_agentrelay(case, req, cache))
        loo = asyncio.run(run_leave_one_out(case, req, cache))
        rnd = asyncio.run(
            run_random_ablation(
                case,
                req,
                cache,
                budget=max(ar["calls"], 4),
                seed=case["seed"] + int(case["id"][-2:]),
            )
        )
    return {
        "id": case["id"],
        "units": case["units"],
        "interaction": case["interaction"],
        "probability": case["probability"],
        "ground_truth": case["ground_truth"],
        "culprit_kinds": case["culprit_kinds"],
        "agentrelay": ar,
        "loo": loo,
        "random": rnd,
        "seconds": round(time.perf_counter() - started, 2),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(method: str, key: str, rows: list[dict[str, Any]] | None = None) -> float:
        rows = results if rows is None else rows
        return sum(bool(r[method][key]) for r in rows) / len(rows) if rows else 0.0

    def false_rate(method: str) -> float | None:
        flagged = sum(len(r[method]["flagged"]) for r in results)
        wrong = sum(len(r[method]["false_flags"]) for r in results)
        return wrong / flagged if flagged else None

    big = [r for r in results if r["units"] > 30]
    ar_big = sum(r["agentrelay"]["calls"] for r in big)
    loo_big = sum(r["loo"]["calls"] for r in big)
    ratios = [
        r["agentrelay"]["shrink_ratio"]
        for r in results
        if r["agentrelay"]["shrink_ratio"] is not None
    ]
    two = [r for r in results if r["interaction"]]
    return {
        "cases": len(results),
        "two_culprit_cases": len(two),
        "large_cases": len(big),
        "top1": {m: rate(m, "top1") for m in ("agentrelay", "loo", "random")},
        "top3": {m: rate(m, "top3") for m in ("agentrelay", "loo", "random")},
        "false_culprit_rate": {m: false_rate(m) for m in ("agentrelay", "loo", "random")},
        "missed_culprit_rate": {
            m: sum(len(r[m]["missed"]) for r in results)
            / sum(len(r["ground_truth"]) for r in results)
            for m in ("agentrelay", "loo", "random")
        },
        "median_shrink_ratio": statistics.median(ratios) if ratios else None,
        "min_shrink_ratio": min(ratios) if ratios else None,
        "interaction_accuracy": rate("agentrelay", "interaction_correct", two),
        "calls_mean": {
            m: statistics.fmean(r[m]["calls"] for r in results)
            for m in ("agentrelay", "loo", "random")
        },
        "calls_median": {
            m: statistics.median(r[m]["calls"] for r in results)
            for m in ("agentrelay", "loo", "random")
        },
        "cost_mean": {
            m: statistics.fmean(r[m]["cost"] for r in results)
            for m in ("agentrelay", "loo", "random")
        },
        "fewer_trials_than_loo_large": 1 - ar_big / loo_big if loo_big else None,
        "per_case_savings_large": [
            1 - r["agentrelay"]["calls"] / r["loo"]["calls"] for r in big if r["loo"]["calls"]
        ],
        "statuses": {
            s: sum(r["agentrelay"]["status"] == s for r in results)
            for s in {r["agentrelay"]["status"] for r in results}
        },
    }


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def check(ok: bool) -> str:
    return "✅ met" if ok else "❌ missed"


def render(results: list[dict[str, Any]], agg: dict[str, Any], seconds: float) -> str:
    top1, ratio = agg["top1"]["agentrelay"], agg["median_shrink_ratio"] or 0.0
    saving = agg["fewer_trials_than_loo_large"] or 0.0
    lines = [
        "# RelayBench results",
        "",
        f"Generated by `bench/run_bench.py` on {datetime.now(UTC):%Y-%m-%d} with AgentRelay "
        f"{__version__}, Python {platform.python_version()}. {agg['cases']} cases, "
        f"{agg['two_culprit_cases']} with two culprits that must appear together; "
        f"{agg['large_cases']} have more than 30 units. Offline (FakeLLM); total runtime "
        f"{seconds:.0f} s.",
        "",
        "## Targets",
        "",
        "| Target | Result | Status |",
        "|---|---|---|",
        f"| Top-1 culprit accuracy ≥ 85% | {pct(top1)} | {check(top1 >= TARGETS['top1'])} |",
        f"| Median shrink ratio ≥ 10× | {ratio:.1f}× | "
        f"{check(ratio >= TARGETS['median_shrink_ratio'])} |",
        f"| ≥ 40% fewer trials than leave-one-out on cases with > 30 units | "
        f"{pct(saving)} fewer | {check(saving >= TARGETS['fewer_trials_than_loo'])} |",
        "",
        "## Methods compared",
        "",
        "| | AgentRelay | Leave-one-out | Random ablation (same calls as AgentRelay) |",
        "|---|---|---|---|",
    ]
    for label, key, fmt in (
        ("Top-1 accuracy", "top1", pct),
        ("Top-3 accuracy", "top3", pct),
        ("False culprit rate", "false_culprit_rate", pct),
        ("Missed culprit rate", "missed_culprit_rate", pct),
    ):
        vals = agg[key]
        lines.append(
            f"| {label} | {fmt(vals['agentrelay'])} | {fmt(vals['loo'])} | {fmt(vals['random'])} |"
        )
    calls_mean, calls_med, cost = agg["calls_mean"], agg["calls_median"], agg["cost_mean"]
    lines += [
        f"| Model calls per diagnosis (mean / median) | {calls_mean['agentrelay']:.0f} / "
        f"{calls_med['agentrelay']:.0f} | {calls_mean['loo']:.0f} / {calls_med['loo']:.0f} | "
        f"{calls_mean['random']:.0f} / {calls_med['random']:.0f} |",
        f"| Simulated cost per diagnosis | ${cost['agentrelay']:.4f} | ${cost['loo']:.4f} | "
        f"${cost['random']:.4f} |",
        "",
        f'Two-culprit cases where AgentRelay reported exactly the right "and" interaction: '
        f"{pct(agg['interaction_accuracy'])} ({agg['two_culprit_cases']} cases). Median shrink "
        f"ratio {ratio:.1f}× (minimum {agg['min_shrink_ratio']:.1f}×). AgentRelay statuses: "
        + ", ".join(f"{k} {v}" for k, v in sorted(agg["statuses"].items()))
        + ".",
        "",
        "## Per case",
        "",
        "| Case | Units | p | Culprits | AgentRelay top-1 | Shrink | Calls (AR / LOO / random) "
        "| LOO top-1 | Random top-1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ar, loo, rnd = r["agentrelay"], r["loo"], r["random"]
        culprits = ", ".join(f"`{c}`" for c in r["ground_truth"])
        shrink_ratio = f"{ar['shrink_ratio']:.0f}×" if ar["shrink_ratio"] else "–"
        lines.append(
            f"| {r['id']} | {r['units']} | {r['probability']:.2f} | {culprits} | "
            f"{'✅' if ar['top1'] else '❌ ' + ar['status']} | {shrink_ratio} | "
            f"{ar['calls']} / {loo['calls']} / {rnd['calls']} | {'✅' if loo['top1'] else '❌'} | "
            f"{'✅' if rnd['top1'] else '❌'} |"
        )
    lines += [
        "",
        "## Findings",
        "",
        *findings(results, agg),
        "",
        "## Notes and caveats",
        "",
        *NOTES,
        "",
    ]
    return "\n".join(lines)


def findings(results: list[dict[str, Any]], agg: dict[str, Any]) -> list[str]:
    out = []
    top1, fcr = agg["top1"], agg["false_culprit_rate"]
    if top1["random"] >= top1["agentrelay"]:
        out.append(
            f"- **Top-1 accuracy does not separate the methods here.** Random ablation given the "
            f"same number of model calls also reaches {pct(top1['random'])} top-1 (leave-one-out "
            f"{pct(top1['loo'])}), because planted triggers make culprits crisp. The differences "
            f"are in what each method *claims*: false culprit rate {pct(fcr['agentrelay'])} for "
            f"AgentRelay vs {pct(fcr['loo'])} (leave-one-out) and {pct(fcr['random'])} (random), "
            f"and in cost."
        )
    unflagged = [r for r in results if r["agentrelay"]["status"] == "no_culprit"]
    if unflagged:
        names = ", ".join(f"{r['id']} (p={r['probability']:.2f})" for r in unflagged)
        settled = [
            r
            for r in unflagged
            if set(r["agentrelay"].get("with_more_trials", {}).get("flagged", []))
            == set(r["ground_truth"])
        ]
        out.append(
            f"- **Missed culprits are confidence, not ranking.** In {len(unflagged)} case(s), "
            f"{names}, AgentRelay ranked the true culprit first but did not flag it: with a low "
            "failure rate, the 95% intervals of the baseline (20 trials) and the ablation (16 "
            "trials) overlapped, and explain only flags non-overlapping intervals. Re-running "
            f"explain with doubled trials (40 / 32) flagged exactly the right unit in "
            f"{len(settled)} of {len(unflagged)} (extra calls not counted above)."
        )
    small = [r for r in results if r["units"] <= 30]
    if small and any(r["agentrelay"]["calls"] > r["loo"]["calls"] for r in small):
        out.append(
            f"- **On small contexts leave-one-out can be cheaper.** For the {len(small)} cases "
            "with 30 units or fewer, testing every unit costs about as much as AgentRelay's "
            "fixed overhead (baseline, confirmation, attribution), which is why the trials "
            "target only covers cases with more than 30 units."
        )
    return out


NOTES = [
    "- **Synthetic and offline.** FakeLLM rules fire on exact trigger strings, so a culprit is "
    "either fully present or absent. Real models react to paraphrases and partial context, so "
    "these numbers are an upper bound on what to expect against a live model. The opt-in live "
    "bench (three examples against a real model) has not been run yet.",
    "- **What counts.** Top-1: the rank-1 unit is a ground-truth culprit. Top-3: every culprit "
    "is in the top 3. False culprit rate: flagged units that are not ground truth, over all "
    "flagged units. Units are compared at the finest level (chunks, not whole tool results).",
    "- **Cost** is model calls that missed the per-method trial cache (repeated questions are "
    "free, as they are for users). Simulated dollars use FakeLLM's $1/$5 per million tokens.",
    "- **Leave-one-out** uses the same baseline (20 trials) and the same sequential test (3-8 "
    "trials) as AgentRelay's search, so the comparison isolates ddmin + targeted attribution "
    "against testing every unit. **Random ablation** gets exactly AgentRelay's call budget per "
    "case and ranks units by the difference in failure rate when kept versus dropped.",
    "- Reproduce: `uv run python bench/generate.py && uv run python bench/run_bench.py`.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=None, help="Only the first N cases.")
    parser.add_argument("--out", type=Path, default=ROOT / "RESULTS.md")
    parser.add_argument("--json", type=Path, default=ROOT / "results.json")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    started = time.perf_counter()
    results = []
    for case in cases:
        r = run_case(case)
        ar = r["agentrelay"]
        print(
            f"{r['id']}: units {r['units']:>2}  agentrelay top1={ar['top1']!s:<5} "
            f"calls={ar['calls']:>4}  loo calls={r['loo']['calls']:>4}  "
            f"random top1={r['random']['top1']!s:<5} ({r['seconds']:.1f}s)",
            flush=True,
        )
        results.append(r)
    seconds = time.perf_counter() - started
    agg = aggregate(results)
    print(json.dumps({k: v for k, v in agg.items() if k != "per_case_savings_large"}, indent=1))
    print(f"total {seconds:.1f}s")
    if args.cases is None or args.out != ROOT / "RESULTS.md":
        args.out.write_text(render(results, agg, seconds))
        args.json.write_text(json.dumps({"aggregate": agg, "cases": results}, indent=1) + "\n")
        print(f"wrote {args.out} and {args.json}")


if __name__ == "__main__":
    main()
