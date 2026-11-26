"""M7 acceptance: strict replay, divergence, bisect, side-effect safety; plus diff and modes."""

from __future__ import annotations

import importlib
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from examples.calendar_agent.agent import main as calendar_main
from examples.file_cleanup_agent.agent import main as cleanup_main
from examples.refund_agent.agent import main as refund_main
from typer.testing import CliRunner

from agentrelay.cli import app
from agentrelay.providers.fake import FakeLLM
from agentrelay.replay.diff import diff_runs
from agentrelay.replay.program import ReplayError, replay_program
from agentrelay.store import RunStore

runner = CliRunner()


@pytest.fixture
def fake_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Every request the fake model answers (i.e. every live LLM call) in this process."""
    calls: list[int] = []
    original = FakeLLM.complete_sync

    def counting(self: FakeLLM, req: Any, *, seed: int | None = None) -> Any:
        calls.append(1)
        return original(self, req, seed=seed)

    monkeypatch.setattr(FakeLLM, "complete_sync", counting)
    return calls


@pytest.mark.parametrize(
    ("main", "seed"), [(refund_main, 7), (cleanup_main, 2), (calendar_main, 1)]
)
def test_strict_replay_makes_zero_live_calls_and_reproduces_the_output(
    main: Any, seed: int, fake_calls: list[int]
) -> None:
    """BLUEPRINT M7 acceptance."""
    main(provider="fake", seed=seed)
    store = RunStore()
    original = store.load("last")
    fake_calls.clear()
    result = replay_program(original)
    assert result.ok, (result.divergence, result.error)
    assert fake_calls == [] and result.live_llm_calls == 0
    assert result.served_llm_calls == sum(s.kind == "llm_call" for s in original.steps)
    assert result.output_matches and result.run is not None
    assert result.run.output == original.output
    replayed = store.load(result.run.id)  # the replay is recorded as its own run
    assert replayed.labels["replay_of"] == original.id
    assert all(s.tags.get("replay") == "cassette" for s in replayed.steps)
    assert diff_runs(original, replayed).counts == {
        "same": len(original.steps),
        "changed": 0,
        "removed": 0,
        "inserted": 0,
    }


def test_changed_system_prompt_raises_divergence_with_a_readable_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BLUEPRINT M7 acceptance."""
    import examples.refund_agent.agent as agent

    refund_main(provider="fake", seed=7)
    original = RunStore().load("last")
    monkeypatch.setattr(agent, "SYSTEM", agent.SYSTEM + "\nAlways answer in one sentence.")
    result = replay_program(original)
    d = result.divergence
    assert d is not None and d.step_id == "s0000" and d.call_index == 0
    assert d.reason == "the system prompt changed"
    assert d.diff.startswith("--- recorded\n+++ replay")
    assert '+  "system":' in d.diff and "Always answer in one sentence." in d.diff
    assert result.live_llm_calls == 0  # strict mode stops instead of going live

    cli = runner.invoke(app, ["replay", original.id], env={"COLUMNS": "200"})
    assert cli.exit_code == 5
    assert "Diverged at s0000" in cli.stdout and "the system prompt changed" in cli.stdout

    forked = replay_program(original, mode="fork")
    assert forked.divergence is None and forked.live_llm_calls == 3 and forked.ok


def test_bisect_finds_the_step_that_retrieved_the_stale_document() -> None:
    """BLUEPRINT M7 acceptance."""
    refund_main(provider="fake", seed=7)
    run = RunStore().load("last")
    result = runner.invoke(
        app,
        ["bisect", run.id, "--oracle", "examples/refund_agent/oracles.py:wrong_refund", "--json"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    bad = report["first_bad_step"]
    assert report["result"]["status"] == "ok" and report["result"]["first_bad"] == 3
    assert bad["id"] == "s0003" and bad["kind"] == "tool_call" and bad["name"] == "search_policy"
    assert run.steps[3].request == {"query": "refund window"}  # the call that got the stale doc
    assert "s0003 is the first bad step" in report["message"]
    assert report["explain_step"] == "s0004"
    curve = {p["k"]: p["estimate"] for p in report["result"]["curve"]}
    assert curve[3]["p_hat"] >= 0.75 and curve[report["result"]["last_good"]]["p_hat"] <= 0.5
    assert RunStore().load_artifact(run.id, "bisect.json")["result"]["first_bad"] == 3


def test_bisect_reports_when_the_task_fails_from_the_start() -> None:
    cleanup_main(provider="fake", seed=2)
    result = runner.invoke(
        app,
        [
            "bisect",
            "last",
            "--oracle",
            "examples/file_cleanup_agent/oracles.py:run_deletes_outside_tmp",
            "--json",
        ],
    )
    report = json.loads(result.stdout)
    assert report["result"]["status"] == "fails_from_start" and report["result"]["first_bad"] == 0
    assert "the task itself fails" in report["message"]


def test_bisect_needs_a_run_level_oracle_that_fails_the_recording() -> None:
    refund_main(provider="fake", seed=7)
    decision = runner.invoke(
        app, ["bisect", "last", "--oracle", "examples/refund_agent/oracles.py:quotes_stale_window"]
    )
    assert decision.exit_code == 2 and "run-level oracle" in decision.stderr
    refund_main(provider="fake", seed=0)  # a run that answered correctly
    passing = runner.invoke(
        app, ["bisect", "last", "--oracle", "examples/refund_agent/oracles.py:wrong_refund"]
    )
    assert passing.exit_code == 2 and "nothing to bisect" in passing.stderr


AGENT = """
import random
import time
import uuid
from datetime import datetime
from pathlib import Path

import agentrelay as relay
from agentrelay.providers.fake import FakeLLM, Rule, text, tool_use
from agentrelay.providers.fake_http import fake_anthropic_client

SENTINEL = Path(__file__).with_name("sentinel.txt")
RULES = [Rule(when_all=["note"], unless_any=["Wrote"],
              respond=tool_use("write_note", {"text": "hello"}))]


@relay.tool(side_effects=True)
def write_note(text: str) -> str:
    SENTINEL.write_text(text)
    return f"Wrote {text}"


@relay.entrypoint
def main(task: str) -> dict:
    client = relay.wrap(fake_anthropic_client(FakeLLM(rules=RULES, default=text("All done."))))
    with relay.run("sentinel", input={"task": task}) as run:
        messages = [{"role": "user", "content": task}]
        for _turn in range(4):  # bounded, like a real agent loop
            resp = client.messages.create(model="m", max_tokens=64, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                break
            results = []
            for b in uses:
                try:
                    out = write_note(**b.input)
                except Exception as exc:
                    out = f"Error: {type(exc).__name__}: {exc}"
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            messages.append({"role": "user", "content": results})
        streamed = ""
        stream = client.messages.create(model="m", max_tokens=64, stream=True,
                                        messages=[{"role": "user", "content": "summarise"}])
        for event in stream:
            if event.type == "content_block_delta":
                streamed += event.delta.text
        output = {"streamed": streamed, "time": time.time(), "uuid": uuid.uuid4().hex,
                  "random": random.random(), "now": datetime.now().isoformat()}
        run.set_output(output)
    return output
"""


@pytest.fixture
def sentinel_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    package = tmp_path / "sentinel_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "agent.py").write_text(textwrap.dedent(AGENT))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "sentinel_pkg.agent", raising=False)
    monkeypatch.delitem(sys.modules, "sentinel_pkg", raising=False)
    return importlib.import_module("sentinel_pkg.agent")


def test_side_effect_tools_never_run_during_replay_unless_allowed(
    sentinel_agent: ModuleType,
) -> None:
    """BLUEPRINT M7 acceptance: a side_effects=True tool that writes a sentinel file."""
    sentinel: Path = sentinel_agent.SENTINEL
    sentinel_agent.main("please write a note")
    run = RunStore().load("last")
    assert sentinel.read_text() == "hello"
    assert run.entrypoint == "sentinel_pkg.agent:main"
    sentinel.unlink()

    strict = replay_program(run)  # recorded result served, tool body not executed
    assert strict.ok and strict.served_tool_calls == 1 and not sentinel.exists()

    for tools in ("live", "cassette-or-live"):
        forked = replay_program(run, mode="fork", fork_at=0, tools=tools)  # type: ignore[arg-type]
        assert not sentinel.exists(), tools
        if tools == "live":
            assert forked.refused_tool_calls >= 1  # every retry is refused too
            note = next(s for s in forked.run.steps if s.name == "write_note")  # type: ignore[union-attr]
            assert note.error is not None and note.error.type == "SideEffectRefused"

    allowed = replay_program(
        run, mode="fork", fork_at=0, tools="live", allow_live_side_effects=True
    )
    assert allowed.live_tool_calls == 1 and sentinel.read_text() == "hello"


def test_streamed_calls_replay_and_time_random_uuid_are_frozen(
    sentinel_agent: ModuleType, fake_calls: list[int]
) -> None:
    sentinel_agent.main("please write a note")
    run = RunStore().load("last")
    assert run.output is not None and run.output["streamed"] == "All done."
    fake_calls.clear()
    first, second = replay_program(run), replay_program(run)
    assert fake_calls == [] and first.ok and second.ok
    assert first.run is not None and second.run is not None
    assert first.run.output == second.run.output  # deterministic across replays
    assert first.run.output["streamed"] == "All done."  # stream re-encoded from the recording
    third = replay_program(run, seed=1)
    assert third.run is not None and third.run.output["uuid"] != first.run.output["uuid"]


def test_tool_not_in_cassette_and_entrypoint_errors() -> None:
    refund_main(provider="fake", seed=7)
    run = RunStore().load("last")
    errors = set()
    for seed in range(1, 12):  # live fork from step 2 lets the model choose another query
        res = replay_program(
            run, mode="fork", fork_at=2, tools="cassette", seed=seed, persist=False
        )
        assert res.run is not None
        errors |= {s.error.type for s in res.run.steps if s.error}
    assert "ToolNotInCassette" in errors
    implicit = run.model_copy(update={"entrypoint": None})
    with pytest.raises(ReplayError, match="no entrypoint"):
        replay_program(implicit)
    missing = run.model_copy(update={"entrypoint": "no_such_module:main"})
    with pytest.raises(ReplayError, match="cannot import"):
        replay_program(missing)


def test_last_skips_replays_and_diff_cli() -> None:
    refund_main(provider="fake", seed=7)
    bad = RunStore().load("last")
    replay_program(bad)
    assert RunStore().resolve("last") == bad.id
    refund_main(provider="fake", seed=0)
    good = RunStore().load("last")
    result = runner.invoke(app, ["diff", bad.id, good.id], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "changed" in result.stdout and "Deltas (B" in result.stdout
    d = diff_runs(bad, good)
    assert d.counts["changed"] >= 2  # the search query and the final answer differ
    changed = [s for s in d.steps if s.kind == "changed"]
    assert any(
        "refund window" in s.request_diff or "refund window" in s.response_diff for s in changed
    )
    as_json = json.loads(runner.invoke(app, ["diff", bad.id, good.id, "--json"]).stdout)
    assert as_json["a"] == bad.id and as_json["counts"] == d.counts


def test_lcs_alignment_marks_inserted_and_removed_steps() -> None:
    cleanup_main(provider="fake", seed=2)  # llm, read_file, llm, delete_file, llm
    cleanup = RunStore().load("last")
    refund_main(provider="fake", seed=7)  # llm, get_order, llm, search_policy, llm
    refund = RunStore().load("last")
    d = diff_runs(cleanup, refund)
    kinds = [(s.kind, s.label) for s in d.steps]
    assert d.counts["removed"] == 2 and d.counts["inserted"] == 2
    assert d.counts["same"] + d.counts["changed"] == 3  # the three LLM calls line up
    assert ("removed", "tool_call read_file") in kinds
    assert ("inserted", "tool_call search_policy") in kinds
    assert d.deltas["steps"] == 0
