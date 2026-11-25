"""CLI: init, ls, show, record."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from examples.file_cleanup_agent.agent import main as cleanup_main
from typer.testing import CliRunner

from agentrelay.cli import _guess_name, app
from agentrelay.store import RunStore

REPO = Path(__file__).resolve().parents[2]
runner = CliRunner()


def test_show_last_renders_the_run() -> None:
    cleanup_main(provider="fake", seed=2)
    run = RunStore().load("last")
    result = runner.invoke(app, ["show", "last"], env={"COLUMNS": "160"})
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert f"Run {run.id}" in out and "file_cleanup" in out and "ok" in out
    for step_id in ("s0000", "s0001", "s0002", "s0003", "s0004"):
        assert step_id in out
    assert "read_file" in out and "delete_file" in out
    assert '→ delete_file({"path": "/home/user/project"})' in out
    assert "5 steps" in out


def test_show_step_prefix_and_json() -> None:
    cleanup_main(provider="fake", seed=2)
    run = RunStore().load("last")
    one = runner.invoke(app, ["show", run.id[:12], "--step", "3"])
    assert one.exit_code == 0, one.output
    assert "s0003 tool_call delete_file" in one.stdout and "/home/user/project" in one.stdout
    dumped = json.loads(runner.invoke(app, ["show", "last", "--json"]).stdout)
    assert dumped["id"] == run.id and dumped["schema"] == "agentrelay/v1"
    assert len(dumped["steps"]) == 5


def test_show_errors_cleanly() -> None:
    assert runner.invoke(app, ["show", "last"]).exit_code == 1
    cleanup_main(provider="fake", seed=2)
    assert runner.invoke(app, ["show", "last", "--step", "s0099"]).exit_code == 1


def test_ls_lists_and_filters() -> None:
    cleanup_main(provider="fake", seed=2)
    cleanup_main(provider="fake", seed=3)
    result = runner.invoke(app, ["ls"], env={"COLUMNS": "160"})
    assert result.exit_code == 0 and result.stdout.count("file_cleanup") == 2
    as_json = json.loads(runner.invoke(app, ["ls", "--json", "--limit", "1"]).stdout)
    assert len(as_json) == 1 and as_json[0]["n_steps"] == 5
    assert "no runs" in runner.invoke(app, ["ls", "--status", "error"]).stdout


def test_init_creates_config_and_gitignore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".gitignore").write_text("node_modules/")
    assert runner.invoke(app, ["init", str(tmp_path)]).exit_code == 0
    assert (tmp_path / "agentrelay.toml").is_file()
    assert (tmp_path / ".agentrelay").is_dir()
    assert (tmp_path / ".gitignore").read_text() == "node_modules/\n.agentrelay/\n"
    again = runner.invoke(app, ["init", str(tmp_path)])
    assert "already exists" in again.stdout
    assert (tmp_path / ".gitignore").read_text().count(".agentrelay") == 1


def test_guess_name() -> None:
    assert (
        _guess_name(["python", "-m", "examples.file_cleanup_agent.agent"]) == "file_cleanup_agent"
    )
    assert _guess_name(["python", "scripts/triage.py"]) == "triage"
    assert _guess_name(["python", "tools/agent.py"]) == "tools"


def _relay() -> str:
    return str(Path(sys.executable).with_name("relay"))


def test_record_runs_a_command_and_prints_the_run_id(isolated_home: Path) -> None:
    env = {**os.environ, "AGENTRELAY_HOME": str(isolated_home)}
    proc = subprocess.run(
        [
            _relay(),
            "record",
            "--",
            sys.executable,
            "-m",
            "examples.file_cleanup_agent.agent",
            "--provider",
            "fake",
            "--seed",
            "2",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Cleanup complete." in proc.stdout
    run = RunStore(isolated_home).load("last")
    assert f"recorded run {run.id}" in proc.stderr
    assert run.entrypoint == "examples.file_cleanup_agent.agent:main"


def test_record_names_implicit_runs_after_the_script(tmp_path: Path, isolated_home: Path) -> None:
    script = tmp_path / "weather_bot.py"
    script.write_text(
        "import agentrelay as relay\n"
        "from agentrelay.providers.fake import FakeLLM, text\n"
        "from agentrelay.providers.fake_http import fake_anthropic_client\n"
        "client = relay.wrap(fake_anthropic_client(FakeLLM(default=text('hi'))))\n"
        "client.messages.create(model='m', max_tokens=8,\n"
        "                       messages=[{'role': 'user', 'content': 'x'}])\n"
    )
    env = {**os.environ, "AGENTRELAY_HOME": str(isolated_home)}
    proc = subprocess.run(
        [_relay(), "record", "--", sys.executable, str(script)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    run = RunStore(isolated_home).load("last")
    assert run.name == "weather_bot" and len(run.steps) == 1
    assert "recorded run" in proc.stderr


def test_record_passes_through_the_exit_code(isolated_home: Path) -> None:
    env = {**os.environ, "AGENTRELAY_HOME": str(isolated_home)}
    proc = subprocess.run(
        [_relay(), "record", "--", sys.executable, "-c", "raise SystemExit(3)"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 3
    assert "no runs were recorded" in proc.stderr
