"""M2 acceptance: the example agents record correct runs, offline, with secrets redacted."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from examples.calendar_agent.agent import main as calendar_main
from examples.file_cleanup_agent.agent import main as cleanup_main
from examples.file_cleanup_agent.tools import PROJECT
from examples.refund_agent.agent import main as refund_main

from agentrelay.providers.base import LLMRequest, render_context
from agentrelay.schema import Run
from agentrelay.store import RunStore

BUGGY_SEED = 2  # with this seed the cleanup agent follows the injection and deletes the project


def last_run() -> Run:
    return RunStore().load("last")


def test_file_cleanup_records_steps_kinds_dependencies_and_tokens() -> None:
    answer = cleanup_main(provider="fake", seed=BUGGY_SEED)
    run = last_run()

    assert answer == "Cleanup complete."
    assert (run.name, run.status) == ("file_cleanup", "ok")
    assert run.entrypoint == "examples.file_cleanup_agent.agent:main"
    assert run.input["seed"] == BUGGY_SEED and run.output == {"answer": answer}

    shape = [(s.id, s.kind, s.name) for s in run.steps]
    assert shape == [
        ("s0000", "llm_call", "fake-model"),
        ("s0001", "tool_call", "read_file"),
        ("s0002", "llm_call", "fake-model"),
        ("s0003", "tool_call", "delete_file"),
        ("s0004", "llm_call", "fake-model"),
    ]
    assert [s.index for s in run.steps] == list(range(5))
    assert run.steps[1].request == {"path": f"{PROJECT}/README.md"}
    assert run.steps[3].request == {"path": PROJECT}  # the bug: it deleted the project
    assert run.steps[3].tags["side_effects"] == "true"

    deps = {s.id: s.depends_on for s in run.steps}
    assert deps == {
        "s0000": [],
        "s0001": ["s0000"],  # tool call requested by the first LLM call
        "s0002": ["s0001"],  # sees the README
        "s0003": ["s0002"],
        "s0004": ["s0001", "s0003"],  # sees both tool results
    }

    for s in run.steps:
        if s.kind == "llm_call":
            req = LLMRequest.model_validate(s.request)
            assert s.tokens_in == max(1, len(render_context(req)) // 4)  # FakeLLM's count
            assert s.tokens_out is not None and s.tokens_out > 0
        else:
            assert s.tokens_in is None
    tokens_in = [s.tokens_in for s in run.steps if s.kind == "llm_call"]
    assert tokens_in == sorted(tokens_in)  # the context only grows


def test_examples_reproduce_their_bug_and_the_fix_removes_it() -> None:
    """Guards the FakeLLM rules: each bug must reproduce, and the documented fix must remove it."""

    def deleted_project(run: Run) -> bool:
        return any(s.request.get("path") == PROJECT for s in run.steps if s.name == "delete_file")

    def quoted_stale_policy(run: Run) -> bool:
        return "60 days" in (run.output or {}).get("answer", "")

    def booked_in_utc(run: Run) -> bool:
        return any(s.request["start"].endswith("Z") for s in run.steps if s.name == "create_event")

    def deleted_temp_file(run: Run) -> bool:
        return any(
            s.request.get("path", "").startswith("/tmp/")
            for s in run.steps
            if s.name == "delete_file"
        )

    for seed in range(30):  # the safe path really cleans up, with or without the fix
        cleanup_main(provider="fake", seed=seed, fixed=True)
        assert deleted_temp_file(last_run()), seed

    cases: list[tuple[Callable[..., str], Callable[[Run], bool], float]] = [
        (cleanup_main, deleted_project, 0.6),
        (refund_main, quoted_stale_policy, 0.1),  # needs the bad search (20%) and then 90%
        (calendar_main, booked_in_utc, 0.6),
    ]
    for main, is_bug, min_rate in cases:
        buggy = fixed = 0
        for seed in range(30):
            main(provider="fake", seed=seed)
            buggy += is_bug(last_run())
            main(provider="fake", seed=seed, fixed=True)
            fixed += is_bug(last_run())
        assert buggy / 30 >= min_rate, (main.__module__, buggy)
        assert fixed == 0, (main.__module__, fixed)


SECRETS = [
    "sk-ant-api03-Zx9aB7cD5eF3gH1iJ0kL2mN4oP6qR8sT",
    "sk-proj-4fGh7JkL9mNpQ2rStUvWxYz0123456789abc",
    "AKIAIOSFODNN7EXAMPLE",
    "ghp_16C7e42F292c6912E7710c838347Ae178B4a",
]


def test_secrets_never_reach_the_store(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import examples.file_cleanup_agent.tools as tools

    leaky_readme = tools.README + "\nDeploy keys: " + " ".join(SECRETS) + "\n"
    monkeypatch.setitem(tools.FILES, f"{PROJECT}/README.md", leaky_readme)
    task = f"Please clean up the temporary files in {PROJECT}. My keys: {' '.join(SECRETS)}"
    cleanup_main(task=task, provider="fake", seed=BUGGY_SEED)

    files = [p for p in isolated_home.rglob("*") if p.is_file()]
    assert any(p.name == "steps.jsonl" for p in files)
    stored = b"".join(p.read_bytes() for p in files)
    for secret in SECRETS:
        assert secret.encode() not in stored, secret
    for prefix in ("sk-ant-api03-", "sk-proj-", "AKIAIOSF", "ghp_16C7"):
        assert prefix.encode() not in stored, prefix
    for kind in ("anthropic_key", "openai_key", "aws_access_key", "github_token"):
        assert f"[REDACTED:{kind}:".encode() in stored
    # The run is still usable: redacted text is in both the task and the tool result.
    run = last_run()
    assert "[REDACTED:anthropic_key:" in run.input["task"]
    assert "[REDACTED:github_token:" in run.steps[1].response["result"]  # type: ignore[index]


def test_real_provider_needs_an_explicit_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from examples._common import cli

    monkeypatch.delenv("AGENTRELAY_MODEL", raising=False)
    with pytest.raises(SystemExit) as info:
        cli(cleanup_main, "task", ["--provider", "anthropic"])
    assert info.value.code == 2
    assert "needs a real model" in capsys.readouterr().err
