"""M6 acceptance: emitted tests fail before the fix and pass after it; `relay ci` exit codes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from examples.file_cleanup_agent.agent import SYSTEM, SYSTEM_FIXED
from examples.file_cleanup_agent.agent import main as cleanup_main
from typer.testing import CliRunner

from agentbisect.cli import app
from agentbisect.testing import ScenarioSkipped, evaluate_scenario, run_scenario

REPO = Path(__file__).resolve().parents[2]
ORACLE = "examples/file_cleanup_agent/oracles.py:deletes_outside_tmp"
RULES = "examples/file_cleanup_agent/fake_rules.py"
runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A user project whose current system prompt lives in prompts.py (buggy for now)."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "prompts.py").write_text(f"SYSTEM = {SYSTEM!r}\n")
    (root / "agentbisect.toml").write_text('system_prompt = "prompts:SYSTEM"\n')
    return root


def diagnose_and_emit(out: Path, *extra: str) -> list[Path]:
    cleanup_main(provider="fake", seed=2)
    explained = runner.invoke(
        app,
        [
            "explain",
            "last",
            "--step",
            "auto",
            "--oracle",
            ORACLE,
            "--provider",
            "fake",
            "--fake-rules",
            RULES,
            "--json",
        ],
    )
    assert explained.exit_code == 0, explained.output
    emitted = runner.invoke(
        app, ["emit-test", "last", "--step", "s0002", "--out", str(out), *extra]
    )
    assert emitted.exit_code == 0, emitted.output
    return sorted(out.glob("test_*.py"))


def pytest_run(test_dir: Path, home: Path, **env: str) -> subprocess.CompletedProcess[str]:
    full_env = {**os.environ, "AGENTBISECT_HOME": str(home), **env}
    full_env.pop("ANTHROPIC_API_KEY", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir), "-q", "-p", "no:cacheprovider", "-rs"],
        cwd=REPO,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_emitted_test_fails_before_the_fix_and_passes_after(
    project: Path, isolated_home: Path
) -> None:
    """BLUEPRINT M6 acceptance, run through a real pytest process."""
    out = project / "tests" / "agentbisect"
    (test_file,) = diagnose_and_emit(out)
    scenario = json.loads(next((out / "scenarios").glob("*.json")).read_text())
    assert scenario["oracle"] == ORACLE and scenario["fingerprint"] == "delete_file:outside_tmp"
    assert scenario["culprits"] == ["msg.2.tr.0.chunk.1"] and scenario["fake_rules"] == RULES
    assert "@pytest.mark.agentbisect" in test_file.read_text()

    before = pytest_run(out, isolated_home, AGENTBISECT_PROVIDER="fake")
    assert before.returncode == 1, before.stdout
    assert "the failure still reproduces" in before.stdout
    assert "overrides: system=prompts:SYSTEM" in before.stdout

    (project / "prompts.py").write_text(f"SYSTEM = {SYSTEM_FIXED!r}\n")  # apply the fix
    after = pytest_run(out, isolated_home, AGENTBISECT_PROVIDER="fake")
    assert after.returncode == 0, after.stdout
    assert "1 passed" in after.stdout


def test_without_an_api_key_the_test_skips_unless_live_is_required(
    project: Path, isolated_home: Path
) -> None:
    out = project / "tests" / "agentbisect"
    diagnose_and_emit(out)
    skipped = pytest_run(out, isolated_home)
    assert skipped.returncode == 0 and "1 skipped" in skipped.stdout
    assert "ANTHROPIC_API_KEY is not set" in skipped.stdout
    required = pytest_run(out, isolated_home, AGENTBISECT_REQUIRE_LIVE="1")
    assert required.returncode == 1 and "AGENTBISECT_REQUIRE_LIVE" in required.stdout


def test_relay_ci_exit_codes_report_and_markdown(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLUEPRINT M6: `relay ci` exits 1 on regression and 0 otherwise, with a Markdown summary."""
    out = project / "tests" / "agentbisect"
    diagnose_and_emit(out)
    summary = tmp_path / "summary.md"
    report = tmp_path / "agentbisect-report.json"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args = ["ci", "--path", str(out), "--provider", "fake", "--report", str(report)]

    regression = runner.invoke(app, args, env={"COLUMNS": "160"})
    assert regression.exit_code == 1, regression.output
    assert "regression" in regression.stdout
    data = json.loads(report.read_text())
    assert data["exit_code"] == 1 and data["results"][0]["status"] == "regression"
    md = summary.read_text()
    assert md.startswith("## AgentBisect regression scenarios") and "❌ regression" in md
    assert "| Scenario | Failure rate | Allowed | Result |" in md

    (project / "prompts.py").write_text(f"SYSTEM = {SYSTEM_FIXED!r}\n")
    monkeypatch.delitem(sys.modules, "prompts")  # a new `relay ci` process would re-import it
    fixed = runner.invoke(app, args, env={"COLUMNS": "160"})
    assert fixed.exit_code == 0, fixed.output
    assert json.loads(report.read_text())["results"][0]["status"] == "pass"
    assert "✅ pass" in summary.read_text()  # appended to the same step summary

    skipped = runner.invoke(app, ["ci", "--path", str(out), "--report", str(report)])
    assert skipped.exit_code == 0 and "skipped" in skipped.stdout  # no key: skipped, not failed
    empty = runner.invoke(app, ["ci", "--path", str(tmp_path / "nothing"), "--report", str(report)])
    assert empty.exit_code == 0 and "no scenarios" in empty.stdout


def test_overrides_stored_at_emit_time_and_model_override(tmp_path: Path) -> None:
    out = tmp_path / "suite"
    diagnose_and_emit(
        out,
        "--system-prompt",
        "examples.file_cleanup_agent.agent:SYSTEM_FIXED",
        "--trials",
        "6",
        "--max-failure-rate",
        "0.2",
    )
    scenario = next((out / "scenarios").glob("*.json"))
    data = json.loads(scenario.read_text())
    assert data["trials"] == 6 and data["max_failure_rate"] == 0.2
    assert data["system_prompt"] == "examples.file_cleanup_agent.agent:SYSTEM_FIXED"
    res = evaluate_scenario(scenario, provider="fake", model="other-model")
    assert res.status == "pass" and res.trials == 6 and res.model == "other-model"
    assert res.overrides == [
        "system=examples.file_cleanup_agent.agent:SYSTEM_FIXED",
        "model=other-model",
    ]
    replay = evaluate_scenario(scenario, provider="fake", system_prompt=None)
    assert replay.status == "pass"  # stored override applies


def test_run_scenario_resolves_paths_and_raises_on_setup_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "suite"
    diagnose_and_emit(out)
    scenario = next((out / "scenarios").glob("*.json"))
    monkeypatch.setenv("AGENTBISECT_PROVIDER", "fake")
    res = run_scenario(scenario, trials=4)  # absolute path works; recorded prompt replayed
    assert not res.passed and res.status == "regression" and res.estimate is not None
    broken = json.loads(scenario.read_text())
    broken["oracle"] = "missing_file.py:nope"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(broken))
    with pytest.raises(RuntimeError, match="could not run"):
        run_scenario(bad)
    monkeypatch.delenv("AGENTBISECT_PROVIDER")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delitem(sys.modules, "pytest")  # outside pytest a skip is an exception
    with pytest.raises(ScenarioSkipped):
        run_scenario(scenario)


def test_emit_test_errors_and_warnings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cleanup_main(provider="fake", seed=2)
    missing = runner.invoke(
        app, ["emit-test", "last", "--step", "s0002", "--out", str(tmp_path / "x")]
    )
    assert missing.exit_code == 2 and "run `relay explain`" in " ".join(missing.stderr.split())
    shrunk_now = runner.invoke(
        app,
        [
            "emit-test",
            "last",
            "--step",
            "s0002",
            "--out",
            str(tmp_path / "x"),
            "--oracle",
            ORACLE,
            "--provider",
            "fake",
            "--fake-rules",
            RULES,
        ],
    )
    assert shrunk_now.exit_code == 0, shrunk_now.output
    again = runner.invoke(
        app, ["emit-test", "last", "--step", "s0002", "--out", str(tmp_path / "x")]
    )
    assert again.exit_code == 2 and "--force" in again.stderr


def test_emitter_warns_when_the_scenario_still_looks_secret(tmp_path: Path) -> None:
    from agentbisect.algo.shrink import ShrinkResult
    from agentbisect.emit.test_emitter import emit_test
    from agentbisect.sdk.redact import Redactor
    from agentbisect.stats.estimate import make_estimate

    shrink = ShrinkResult(
        status="ok",
        message="",
        fingerprint="leak:key",
        removal="placeholder",
        units=[],
        original_tokens=1,
        removable_units=0,
        minimal_ids=[],
        minimal_removable_ids=[],
        minimal_tokens=1,
        baseline=make_estimate(9, 10, decided="fails", threshold=0.5),
        minimal_request={
            "model": "m",
            "system": None,
            "max_tokens": 5,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "token itk_abcd1234"}]}
            ],
        },
        run_id="RUNID1",
        step_id="s0001",
        oracle_ref="o.py:x",
    )
    emitted = emit_test(
        shrink, out_dir=tmp_path, redactor=Redactor(extra_patterns={"internal": r"itk_[a-z0-9]{8}"})
    )
    assert emitted.warnings and "internal" in emitted.warnings[0]
    assert emitted.test.name == "test_RUNID1_s0001_leak_key.py"
    compile(emitted.test.read_text(), str(emitted.test), "exec")  # valid Python


def test_committed_example_regression_passes_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The example the repository's own CI runs through the reusable workflow."""
    monkeypatch.setenv("AGENTBISECT_PROVIDER", "fake")
    (scenario,) = (REPO / "examples/file_cleanup_agent/regression/scenarios").glob("*.json")
    res: Any = evaluate_scenario(scenario)
    assert res.status == "pass", res.summary()
