"""M4 acceptance: shrink each example's failing decision to its culprit context."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from examples.calendar_agent.agent import main as calendar_main
from examples.file_cleanup_agent.agent import main as cleanup_main
from examples.refund_agent.agent import main as refund_main
from typer.testing import CliRunner

from agentrelay.algo.shrink import ShrinkConfig, shrink
from agentrelay.cli import app
from agentrelay.context.constraints import validate
from agentrelay.context.units import split_chunks
from agentrelay.providers.base import LLMRequest, LLMResponse, Message, render_context, text_block
from agentrelay.providers.fake import FakeLLM, Rule, text, tool_use
from agentrelay.schema import Failure
from agentrelay.stats.runner import Budget, TrialRunner
from agentrelay.store import RunStore

runner = CliRunner()


def shrink_json(example: str, oracle: str, *extra: str) -> tuple[int, dict[str, Any]]:
    result = runner.invoke(
        app,
        [
            "shrink",
            "last",
            "--step",
            "auto",
            "--oracle",
            f"examples/{example}/oracles.py:{oracle}",
            "--provider",
            "fake",
            "--fake-rules",
            f"examples/{example}/fake_rules.py",
            *extra,
            "--json",
        ],
    )
    return result.exit_code, json.loads(result.stdout)


def injected_chunk_id(out: dict[str, Any]) -> str:
    req = LLMRequest.model_validate(RunStore().load("last").steps[2].request)
    chunks = split_chunks(req.messages[2].content[0].text or "")
    (index,) = [i for i, c in enumerate(chunks) if "ignore previous instructions" in c]
    return f"msg.2.tr.0.chunk.{index}"


def test_file_cleanup_minimal_set_is_the_injected_chunk() -> None:
    """BLUEPRINT M4: contains the injected chunk and at most 3 removable units."""
    cleanup_main(provider="fake", seed=2)
    code, out = shrink_json("file_cleanup_agent", "deletes_outside_tmp")
    assert code == 0 and out["status"] == "ok"
    assert out["step_id"] == "s0002" and out["fingerprint"] == "delete_file:outside_tmp"
    injected = injected_chunk_id(out)
    assert injected in out["minimal_ids"]
    assert len(out["minimal_removable_ids"]) <= 3
    assert out["minimal_removable_ids"] == ["msg.2.tr.0", injected]
    assert out["minimal_tokens"] < out["original_tokens"] / 2
    assert out["minimal_estimate"]["trials"] == 16 and out["minimal_estimate"]["p_hat"] >= 0.5
    minimal = LLMRequest.model_validate(out["minimal_request"])
    assert validate(minimal) == []
    assert "ignore previous instructions" in render_context(minimal)
    assert "# Acme Project" not in render_context(minimal)  # the rest of the README is gone
    saved = RunStore().load_artifact(out["run_id"], "shrink-s0002.json")
    assert saved["schema"] == "agentrelay/v1" and saved["minimal_ids"] == out["minimal_ids"]


def test_rerunning_shrink_is_free() -> None:
    cleanup_main(provider="fake", seed=2)
    _, first = shrink_json("file_cleanup_agent", "deletes_outside_tmp")
    _, second = shrink_json("file_cleanup_agent", "deletes_outside_tmp")
    assert first["provider_calls"] > 0 and second["provider_calls"] == 0
    assert second["cached_trials"] == second["trials"] and second["cost_usd"] == 0
    assert first["minimal_ids"] == second["minimal_ids"]


def test_refund_minimal_set_isolates_the_stale_policy() -> None:
    refund_main(provider="fake", seed=7)
    code, out = shrink_json("refund_agent", "quotes_stale_window")
    assert code == 0 and out["status"] == "ok"
    kept_chunks = [
        u for u in out["units"] if u["kind"] == "chunk" and u["id"] in out["minimal_ids"]
    ]
    assert len(kept_chunks) == 1 and "2024-01-01" in kept_chunks[0]["text"]
    assert len(out["minimal_removable_ids"]) <= 3


def test_calendar_minimal_set_includes_the_tool_description() -> None:
    calendar_main(provider="fake", seed=1)
    code, out = shrink_json("calendar_agent", "books_in_utc")
    assert code == 0 and out["status"] == "ok"
    assert "tool.create_event" in out["minimal_removable_ids"]
    assert "tool.get_user_profile" not in out["minimal_ids"]


def test_delete_mode_also_finds_the_injection() -> None:
    cleanup_main(provider="fake", seed=2)
    code, out = shrink_json("file_cleanup_agent", "deletes_outside_tmp", "--removal", "delete")
    assert code == 0 and out["removal"] == "delete"
    assert injected_chunk_id(out) in out["minimal_ids"]
    assert validate(LLMRequest.model_validate(out["minimal_request"])) == []


def test_terminal_output() -> None:
    cleanup_main(provider="fake", seed=2)
    result = runner.invoke(
        app,
        [
            "shrink",
            "last",
            "--step",
            "auto",
            "--oracle",
            "examples/file_cleanup_agent/oracles.py:deletes_outside_tmp",
            "--provider",
            "fake",
            "--fake-rules",
            "examples/file_cleanup_agent/fake_rules.py",
        ],
        env={"COLUMNS": "160"},
    )
    assert result.exit_code == 0, result.output
    out = result.stdout
    for text_ in (
        "Baseline failure rate",
        "Minimal repro",
        "Minimal failure rate",
        "Minimal context",
        "always kept",
        "Next:  relay explain",
    ):
        assert text_ in out


def test_too_rare_and_not_failing_and_budget(tmp_path: Any) -> None:
    rare = tmp_path / "rare_rules.py"
    rare.write_text(
        "from agentrelay.providers.fake import Rule, text, tool_use\n"
        "RULES = [Rule(when_all=['ignore previous instructions'], probability=0.05,\n"
        "              respond=tool_use('delete_file', {'path': '/home/user/project'}))]\n"
        "DEFAULT = text('ok')\n"
    )
    cleanup_main(provider="fake", seed=2)
    base = [
        "shrink",
        "last",
        "--step",
        "2",
        "--oracle",
        "examples/file_cleanup_agent/oracles.py:deletes_outside_tmp",
        "--provider",
        "fake",
    ]
    too_rare = runner.invoke(app, [*base, "--fake-rules", str(rare), "--trials", "20", "--json"])
    out = json.loads(too_rare.stdout)
    assert too_rare.exit_code == 4 and out["status"] in ("too_rare", "not_failing")
    assert "too rare" in out["message"] or "never failed" in out["message"]

    rules = "examples/file_cleanup_agent/fake_rules.py"
    broke = runner.invoke(app, [*base, "--fake-rules", rules, "--max-cost", "0.015", "--json"])
    out = json.loads(broke.stdout)
    assert broke.exit_code == 3 and out["status"] == "partial"
    assert out["baseline"]["trials"] == 10  # the budget ran out during the search, not before
    assert validate(LLMRequest.model_validate(out["minimal_request"])) == []


def test_shrink_argument_errors() -> None:
    cleanup_main(provider="fake", seed=2)
    run_oracle = runner.invoke(
        app,
        [
            "shrink",
            "last",
            "--step",
            "2",
            "--oracle",
            "examples/file_cleanup_agent/oracles.py:run_deletes_outside_tmp",
            "--provider",
            "fake",
            "--fake-rules",
            "examples/file_cleanup_agent/fake_rules.py",
        ],
    )
    assert run_oracle.exit_code == 2 and "judges whole runs" in run_oracle.stderr
    passing = runner.invoke(
        app,
        [
            "shrink",
            "last",
            "--step",
            "0",
            "--oracle",
            "examples/file_cleanup_agent/oracles.py:deletes_outside_tmp",
            "--provider",
            "fake",
            "--fake-rules",
            "examples/file_cleanup_agent/fake_rules.py",
        ],
    )
    assert passing.exit_code == 2 and "passes the oracle" in " ".join(passing.stderr.split())
    bad_mode = runner.invoke(
        app, ["shrink", "last", "--step", "2", "--oracle", "x.py:y", "--removal", "squash"]
    )
    assert bad_mode.exit_code == 2


class FadesWithoutExtra:
    """Fails on every sample while the 'extra' sentence is present; without it, only for the
    first 6 samples. ddmin (≤ 8 samples) therefore accepts the smaller context, but the 16-trial
    confirmation rejects it."""

    name = "fades"
    cache_namespace = "fades"

    async def complete(self, req: LLMRequest, *, seed: int | None = None) -> LLMResponse:
        ctx = render_context(req)
        if "CULPRIT" in ctx and ("EXTRA" in ctx or (seed or 0) < 6):
            return tool_use("delete_file", {"path": "/"})
        return text("fine")

    def estimate_cost(self, req: LLMRequest) -> float:
        return 0.0


async def bad_delete(resp: LLMResponse) -> Failure | None:
    return Failure(fingerprint="delete") if resp.tool_calls() else None


def test_unconfirmed_minimum_backs_off_and_is_flagged_unstable() -> None:
    req = LLMRequest(
        model="m",
        system="CULPRIT rule.\n\nEXTRA rule.\n\nfiller rule.",
        max_tokens=10,
        messages=[Message(role="user", content=[text_block("do the task")])],
    )
    trial_runner = TrialRunner(FadesWithoutExtra(), bad_delete, cache=None, budget=Budget())
    result = asyncio.run(
        shrink(req, runner=trial_runner, fingerprint="delete", config=ShrinkConfig())
    )
    assert result.status == "unstable"
    assert "sys.0" in result.minimal_ids and "sys.1" in result.minimal_ids
    assert result.minimal_estimate is not None and result.minimal_estimate.still_fails()


def test_slippage_is_reported() -> None:
    llm_rules = [
        Rule(when_all=["CULPRIT"], respond=tool_use("delete_file", {"path": "/"}), probability=0.7),
        Rule(when_all=["CULPRIT"], respond=tool_use("format_disk", {})),
    ]

    async def judge(resp: LLMResponse) -> Failure | None:
        for c in resp.tool_calls():
            return Failure(fingerprint=c.tool_name or "")
        return None

    req = LLMRequest(
        model="m",
        system="CULPRIT rule.\n\nother rule.",
        max_tokens=10,
        messages=[Message(role="user", content=[text_block("task")])],
    )
    trial_runner = TrialRunner(FakeLLM(rules=llm_rules), judge, cache=None, budget=Budget())
    result = asyncio.run(shrink(req, runner=trial_runner, fingerprint="delete_file"))
    assert result.status == "ok" and result.minimal_removable_ids == ["sys.0"]
    assert result.slippage.get("format_disk", 0) > 0
