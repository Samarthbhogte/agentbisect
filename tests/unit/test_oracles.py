from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentbisect import Failure, oracle
from agentbisect.oracles.api import OracleError, acheck, check, load_oracle, registered
from agentbisect.oracles.builtin import (
    judge,
    loops,
    output_matches,
    output_not_matches,
    tool_called,
)
from agentbisect.providers.base import LLMResponse
from agentbisect.providers.fake import FakeLLM, text, tool_use
from agentbisect.schema import Run, Step, request_hash, step_id

DELETE_PROJECT = tool_use("delete_file", {"path": "/home/user/project"})
DELETE_TMP = tool_use("delete_file", {"path": "/tmp/x"})


def test_decorator_registers_and_checks() -> None:
    @oracle("unit-says-sorry")
    def says_sorry(resp: LLMResponse) -> Failure | None:
        return Failure(fingerprint="sorry") if "sorry" in resp.text().lower() else None

    assert registered()["unit-says-sorry"] is says_sorry
    assert check(says_sorry, text("Sorry!")) == Failure(fingerprint="sorry")
    assert check(says_sorry, text("Done.")) is None
    assert load_oracle("unit-says-sorry") is says_sorry


def test_oracle_exceptions_and_bad_returns_are_oracle_errors() -> None:
    @oracle("unit-broken")
    def broken(resp: LLMResponse) -> Failure | None:
        raise KeyError("path")

    @oracle("unit-wrong-type")
    def wrong_type(resp: LLMResponse) -> bool:
        return True

    with pytest.raises(OracleError, match="raised KeyError"):
        check(broken, text("x"))
    with pytest.raises(OracleError, match="returned bool"):
        check(wrong_type, text("x"))


def test_level_mismatch_is_an_oracle_error() -> None:
    with pytest.raises(OracleError, match="judges whole runs"):
        check(loops(), text("x"))
    with pytest.raises(OracleError, match="judges decisions"):
        check(tool_called("x"), make_run([]))


async def test_async_oracles_need_acheck() -> None:
    @oracle("unit-async")
    async def is_empty(resp: LLMResponse) -> Failure | None:
        return None if resp.text() else Failure(fingerprint="empty")

    assert await acheck(is_empty, text("")) == Failure(fingerprint="empty")
    with pytest.raises(OracleError, match="async"):
        check(is_empty, text(""))


def test_load_from_package_file_module_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    o = load_oracle("examples/file_cleanup_agent/oracles.py:deletes_outside_tmp")
    assert o.name == "deletes-outside-tmp" and o.level == "decision"
    assert load_oracle("examples.refund_agent.oracles:wrong_refund").level == "run"
    cfg = {"cleanup": "examples/file_cleanup_agent/oracles.py:deletes_outside_tmp"}
    assert load_oracle("cleanup", config_oracles=cfg) is o

    standalone = tmp_path / "my_oracles.py"
    standalone.write_text("def no_text(resp):\n    return None\n")
    plain = load_oracle(f"{standalone}:no_text")
    assert plain.level == "decision" and check(plain, text("x")) is None

    for bad in ("nope", "missing.py:x", f"{standalone}:absent"):
        with pytest.raises(OracleError):
            load_oracle(bad)


def test_example_decision_oracles() -> None:
    cleanup = load_oracle("examples/file_cleanup_agent/oracles.py:deletes_outside_tmp")
    assert check(cleanup, DELETE_PROJECT) == Failure(
        fingerprint="delete_file:outside_tmp", detail="/home/user/project"
    )
    assert check(cleanup, DELETE_TMP) is None
    refund = load_oracle("examples/refund_agent/oracles.py:quotes_stale_window")
    assert check(refund, text("You have a 60-day window.")) is not None
    assert check(refund, text("Refunds within 30 days.")) is None
    calendar = load_oracle("examples/calendar_agent/oracles.py:books_in_utc")
    assert check(
        calendar, tool_use("create_event", {"title": "x", "start": "2026-12-01T15:00:00Z"})
    )
    assert (
        check(
            calendar, tool_use("create_event", {"title": "x", "start": "2026-12-01T15:00:00-05:00"})
        )
        is None
    )


def test_tool_called_and_output_matchers() -> None:
    outside = tool_called("delete_file", where=lambda a: not a["path"].startswith("/tmp/"))
    assert check(outside, DELETE_PROJECT) is not None
    assert check(outside, DELETE_TMP) is None
    assert check(tool_called("delete_file"), DELETE_TMP).fingerprint == "tool_called:delete_file"  # type: ignore[union-attr]
    assert check(output_matches(r"60 days"), text("Within 60 DAYS.")) is not None
    assert check(output_not_matches(r"30 days"), text("Within 60 days.")) is not None
    assert check(output_not_matches(r"30 days"), text("Within 30 days.")) is None
    run_level = output_matches(r"60 days", level="run", fingerprint="stale")
    assert check(run_level, make_run([], output={"answer": "60 days!"})) == Failure(
        fingerprint="stale", detail="60 days"
    )


def make_run(
    tool_calls: list[tuple[str, dict[str, str]]], output: dict[str, str] | None = None
) -> Run:
    now = datetime.now(UTC)
    steps = [
        Step(
            id=step_id(i),
            run_id="r",
            index=i,
            kind="tool_call",
            name=name,
            request=args,
            response={"result": "ok"},
            raw_request=None,
            raw_response=None,
            started_at=now,
            ended_at=now,
            request_hash=request_hash(args),
        )
        for i, (name, args) in enumerate(tool_calls)
    ]
    return Run(
        id="r",
        name="t",
        entrypoint=None,
        input={},
        output=output,
        status="ok",
        steps=steps,
        started_at=now,
        ended_at=now,
        env={},
    )


def test_loops() -> None:
    calls = [("search", {"q": "a"})] * 4 + [("search", {"q": "b"})]
    assert check(loops(max_repeats=3), make_run(calls)) == Failure(
        fingerprint="loop:search", detail="4 identical calls"
    )
    assert check(loops(max_repeats=4), make_run(calls)) is None


async def test_judge_uses_a_provider_and_is_flagged_noisy() -> None:
    verdict = {"fail": True, "category": "Quoted Stale Policy!", "reason": "Says 60 days."}
    judge_llm = FakeLLM(
        script=[text(json.dumps(verdict)), text('{"fail": false}'), text("not json")]
    )
    o = judge(
        "Fail if the reply quotes a 60-day refund window.", model="judge-model", provider=judge_llm
    )
    assert o.noisy
    assert await acheck(o, text("60 days")) == Failure(
        fingerprint="judge:quoted_stale_policy", detail="Says 60 days."
    )
    assert await acheck(o, text("30 days")) is None
    with pytest.raises(OracleError, match="did not return JSON"):
        await acheck(o, text("??"))
