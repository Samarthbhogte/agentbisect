"""M3 acceptance through the CLI: `relay try`."""

from __future__ import annotations

import json
from typing import Any

import pytest
from examples.file_cleanup_agent.agent import main as cleanup_main
from typer.testing import CliRunner

from agentbisect.cli import app
from agentbisect.providers.fake import FakeLLM

runner = CliRunner()
ORACLE = "examples/file_cleanup_agent/oracles.py:deletes_outside_tmp"
RULES = "examples/file_cleanup_agent/fake_rules.py"
BASE = [
    "try",
    "last",
    "--step",
    "auto",
    "--oracle",
    ORACLE,
    "--provider",
    "fake",
    "--fake-rules",
    RULES,
]


def try_json(*extra: str) -> tuple[int, dict[str, Any]]:
    result = runner.invoke(app, [*BASE, *extra, "--json"])
    return result.exit_code, json.loads(result.stdout)


@pytest.fixture
def recorded() -> None:
    cleanup_main(provider="fake", seed=2)


@pytest.fixture
def counted_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts every FakeLLM call made in this process (the provider-side call counter)."""
    calls: list[int] = []
    original = FakeLLM.complete_sync

    def counting(self: FakeLLM, req: Any, *, seed: int | None = None) -> Any:
        calls.append(seed or 0)
        return original(self, req, seed=seed)

    monkeypatch.setattr(FakeLLM, "complete_sync", counting)
    return calls


def test_try_reports_the_failure_rate(recorded: None) -> None:
    code, out = try_json("--trials", "20")
    assert code == 0
    assert out["step"] == "s0002" and out["fingerprint"] == "delete_file:outside_tmp"
    assert out["recorded_failure"]["detail"] == "/home/user/project"
    est = out["estimate"]
    assert est["trials"] == 20 and 0.6 <= est["p_hat"] <= 0.95
    assert est["ci_low"] < est["p_hat"] < est["ci_high"]
    summaries = {r["summary"] for r in out["responses"]}
    assert '→ delete_file({"path": "/home/user/project"})' in summaries


def test_repeating_try_makes_zero_provider_calls(recorded: None, counted_calls: list[int]) -> None:
    code1, first = try_json("--trials", "12")
    calls_after_first = len(counted_calls)
    code2, second = try_json("--trials", "12")
    assert code1 == code2 == 0
    assert calls_after_first == 12 and len(counted_calls) == 12  # provider counter unchanged
    assert first["stats"]["provider_calls"] == 12
    assert second["stats"]["provider_calls"] == 0 and second["stats"]["cache_hits"] == 12
    assert second["stats"]["cost_usd"] == 0
    assert first["estimate"] == second["estimate"]
    code3, third = try_json("--trials", "15")  # 12 cached + 3 new
    assert code3 == 0 and third["stats"]["provider_calls"] == 3


def test_max_cost_stops_early_with_a_partial_result(recorded: None) -> None:
    result = runner.invoke(
        app, [*BASE, "--trials", "10", "--max-cost", "0.01"], env={"COLUMNS": "150"}
    )
    assert result.exit_code == 3, result.output
    assert "Partial result" in result.stdout and "budget of $0.01" in result.stdout
    code, out = try_json("--trials", "50", "--max-cost", "0.01")
    est = out["estimate"]
    assert code == 3 and est["partial"] and est["decided"] == "undecided"
    assert 0 < est["trials"] < 50
    assert out["budget"]["spent"] <= 0.01


def test_other_fingerprints_are_slippage(recorded: None) -> None:
    code, out = try_json("--trials", "20", "--fingerprint", "delete_file:other")
    est = out["estimate"]
    assert code == 0 and est["failures"] == 0
    assert est["slippage"] == {"delete_file:outside_tmp": out["responses"][0]["count"]}


def test_step_refs_overrides_and_no_oracle(recorded: None) -> None:
    by_number = runner.invoke(
        app,
        [
            "try",
            "last",
            "--step",
            "2",
            "--provider",
            "fake",
            "--fake-rules",
            RULES,
            "--trials",
            "6",
            "--json",
            "--model",
            "other-model",
        ],
    )
    assert by_number.exit_code == 0, by_number.output
    out = json.loads(by_number.stdout)
    assert out["step"] == "s0002" and out["model"] == "other-model" and out["oracle"] is None
    assert out["fingerprint"] == "*" and sum(r["count"] for r in out["responses"]) == 6
    plain = runner.invoke(
        app,
        [
            "try",
            "last",
            "--step",
            "2",
            "--provider",
            "fake",
            "--fake-rules",
            RULES,
            "--trials",
            "3",
        ],
    )
    assert "response distribution only" in plain.stdout


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--step", "auto", "--provider", "fake", "--fake-rules", RULES], "needs a decision-level"),
        (["--step", "1", "--provider", "fake", "--fake-rules", RULES], "not an LLM call"),
        (["--step", "s0099", "--provider", "fake", "--fake-rules", RULES], "has no step"),
        (["--step", "2", "--provider", "fake"], "--fake-rules"),
        (
            [
                "--step",
                "2",
                "--oracle",
                "examples/refund_agent/oracles.py:wrong_refund",
                "--provider",
                "fake",
                "--fake-rules",
                RULES,
            ],
            "judges whole runs",
        ),
    ],
)
def test_try_errors_are_clear(recorded: None, args: list[str], message: str) -> None:
    result = runner.invoke(app, ["try", "last", *args], env={"COLUMNS": "200"})
    assert result.exit_code == 2
    assert message in " ".join(result.stderr.split())


def test_oracle_errors_abort(recorded: None, tmp_path: Any) -> None:
    bad = tmp_path / "bad_oracles.py"
    bad.write_text("def boom(resp):\n    raise RuntimeError('oops')\n")
    result = runner.invoke(
        app,
        [
            "try",
            "last",
            "--step",
            "2",
            "--oracle",
            f"{bad}:boom",
            "--provider",
            "fake",
            "--fake-rules",
            RULES,
        ],
    )
    assert result.exit_code == 2
    assert "oracle error" in result.stderr and "RuntimeError: oops" in result.stderr
