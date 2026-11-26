"""RelayBench: reproducible cases with valid ground truth, and a quick end-to-end run."""

from __future__ import annotations

import json
import time
from pathlib import Path

from bench.generate import N_CASES, N_TWO_CULPRIT, generate
from bench.run_bench import aggregate, load_cases, render, run_case

from agentbisect.context.units import decompose, explained_ids, unit_text
from agentbisect.providers.base import LLMRequest

CASES = Path(__file__).resolve().parents[2] / "bench" / "cases"


def test_generator_is_reproducible_and_matches_the_committed_cases() -> None:
    first, second = generate(), generate()
    assert first == second
    assert len(first) == N_CASES and sum(c["interaction"] for c in first) == N_TWO_CULPRIT
    committed = [json.loads(p.read_text()) for p in sorted(CASES.glob("case_*.json"))]
    assert [c["id"] for c in committed] == [c["id"] for c in first]
    assert committed == json.loads(json.dumps(first))  # regenerate with bench/generate.py


def test_cases_follow_the_spec_and_ground_truth_marks_the_triggers() -> None:
    for case in load_cases():
        assert 10 <= case["units"] <= 80
        assert 0.4 <= case["probability"] <= 0.95
        req = LLMRequest.model_validate(case["request"])
        units = {u.id: u for u in decompose(req)}
        finest = set(explained_ids(list(units.values())))
        assert set(case["ground_truth"]) <= finest
        assert all(units[uid].removable for uid in case["ground_truth"])
        texts = {uid: unit_text(req, units[uid]) for uid in case["ground_truth"]}
        for code in case["triggers"]:
            holders = [uid for uid in finest if code in unit_text(req, units[uid])]
            assert len(holders) == 1 and holders[0] in texts, (case["id"], code)


def test_quick_bench_run(tmp_path: Path) -> None:
    started = time.perf_counter()
    results = [run_case(case) for case in load_cases(3)]
    elapsed = time.perf_counter() - started
    assert elapsed < 30  # the full 30-case bench must stay well under 3 minutes
    agg = aggregate(results)
    assert agg["top1"]["agentbisect"] == 1.0
    assert agg["false_culprit_rate"]["agentbisect"] == 0.0
    for r in results:
        assert r["agentbisect"]["calls"] > 0 and r["random"]["calls"] >= r["agentbisect"]["calls"]
        assert r["agentbisect"]["shrink_ratio"] and r["agentbisect"]["shrink_ratio"] > 1
    md = render(results, agg, elapsed)
    for section in (
        "## Targets",
        "## Methods compared",
        "## Per case",
        "## Findings",
        "## Notes and caveats",
    ):
        assert section in md
    assert "Top-1 culprit accuracy ≥ 85%" in md


def test_committed_results_report_meets_or_states_the_targets() -> None:
    md = (CASES.parent / "RESULTS.md").read_text()
    targets = md.split("## Targets")[1].split("##")[0]
    assert targets.count("✅ met") + targets.count("❌ missed") == 3
    data = json.loads((CASES.parent / "results.json").read_text())
    assert data["aggregate"]["cases"] == N_CASES
