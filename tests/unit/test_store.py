from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agentbisect.ids import new_ulid
from agentbisect.providers.base import LLMRequest, Message, text, text_block
from agentbisect.schema import Run, Step, request_hash, step_id
from agentbisect.store import AmbiguousRunRef, BlobStore, RunNotFound, RunStore, resolve_home
from agentbisect.store.index import RunFilter

T0 = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)


def make_step(run_id: str, i: int, payload: str = "") -> Step:
    if i % 2 == 0:
        req = LLMRequest(
            model="fake-model",
            system="You clean temp files.",
            messages=[Message(role="user", content=[text_block(f"turn {i} {payload}")])],
            max_tokens=256,
        ).to_dict()
        kind, name, resp = "llm_call", "fake-model", text(f"reply {i}").to_dict()
    else:
        req = {"path": f"/tmp/f{i}.txt"}
        kind, name, resp = "tool_call", "read_file", {"result": f"contents {i} {payload}"}
    return Step(
        id=step_id(i),
        run_id=run_id,
        index=i,
        kind=kind,
        name=name,
        provider="fake",
        request=req,
        response=resp,
        raw_request=None,
        raw_response=None,
        depends_on=[step_id(i - 1)] if i else [],
        started_at=T0 + timedelta(seconds=i),
        ended_at=T0 + timedelta(seconds=i, milliseconds=5),
        tokens_in=10 + i,
        tokens_out=5,
        cost_usd=0.001 * i,
        request_hash=request_hash(req),
        tags={"i": str(i)},
    )


def make_run(n_steps: int, payload: str = "", **kw: object) -> Run:
    run_id = new_ulid()
    fields: dict[str, object] = dict(
        id=run_id,
        name="cleanup",
        entrypoint="examples.x:main",
        input={"task": "clean"},
        output={"answer": "done"},
        status="ok",
        started_at=T0,
        ended_at=T0 + timedelta(minutes=1),
        env={"python": "3.12"},
        labels={"failure": "deleted wrong file"},
    )
    fields.update(kw)
    return Run(steps=[make_step(run_id, i, payload) for i in range(n_steps)], **fields)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / ".agentbisect")


def test_round_trip_50_steps_with_save_run(store: RunStore) -> None:
    run = make_run(50)
    store.save_run(run)
    assert store.load(run.id) == run


def test_round_trip_50_steps_with_incremental_appends(store: RunStore) -> None:
    run = make_run(50)
    store.start_run(run.model_copy(update={"steps": [], "status": "unknown", "ended_at": None}))
    for step in run.steps:
        store.append_step(step)
    store.finish_run(run)
    assert store.load(run.id) == run


def test_large_strings_go_to_blobs_and_come_back(store: RunStore) -> None:
    big = "x" * (16 * 1024 + 1)
    run = make_run(4, payload=big, input={"doc": big})
    store.save_run(run)

    run_dir = store.run_dir(run.id)
    stored_text = (run_dir / "steps.jsonl").read_text() + (run_dir / "run.json").read_text()
    assert big not in stored_text
    assert '"$blob"' in stored_text
    blobs = list((store.home / "blobs").iterdir())
    assert any(p.read_text() == big for p in blobs)
    assert store.load(run.id) == run


def test_string_at_threshold_stays_inline(tmp_path: Path) -> None:
    blobs = BlobStore(tmp_path)
    exact = "y" * (16 * 1024)
    assert blobs.externalize({"s": exact}) == {"s": exact}


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
@given(
    st.recursive(
        st.none() | st.integers() | st.text(max_size=50) | st.just("z" * 20_000),
        lambda inner: (
            st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=5), inner, max_size=3)
        ),
        max_leaves=10,
    )
)
def test_blob_externalize_restore_is_identity(tmp_path: Path, value: object) -> None:
    blobs = BlobStore(tmp_path / "blobs")
    assert blobs.restore(blobs.externalize(value)) == value


def test_every_stored_file_carries_schema(store: RunStore) -> None:
    run = make_run(3)
    store.save_run(run)
    store.save_artifact(run.id, "shrink.json", {"minimal": ["msg.1"]})
    run_dir = store.run_dir(run.id)
    assert json.loads((run_dir / "run.json").read_text())["schema"] == "agentbisect/v1"
    assert json.loads((run_dir / "steps.jsonl").read_text().splitlines()[0]) == {
        "schema": "agentbisect/v1"
    }
    assert store.load_artifact(run.id, "shrink.json")["schema"] == "agentbisect/v1"


def test_torn_final_line_is_ignored(store: RunStore) -> None:
    run = make_run(5)
    store.save_run(run)
    with (store.run_dir(run.id) / "steps.jsonl").open("a") as fh:
        fh.write('{"id": "s0005", "run_id": "trunc')  # crash mid-write
    assert store.load(run.id).steps == run.steps


def test_resolve_last_prefix_and_errors(store: RunStore) -> None:
    a, b = make_run(1), make_run(1)
    store.save_run(a)
    store.save_run(b)
    assert store.resolve("last") == b.id
    assert store.resolve(b.id[-26:]) == b.id
    assert store.resolve(a.id.lower()) == a.id  # full id, case-insensitive prefix
    with pytest.raises(AmbiguousRunRef):
        store.resolve(a.id[:2])
    with pytest.raises(RunNotFound):
        store.resolve("ZZZZZZ")


def test_resolve_last_on_empty_store(store: RunStore) -> None:
    with pytest.raises(RunNotFound):
        store.resolve("last")


def test_index_list_and_filter(store: RunStore) -> None:
    ok = make_run(2)
    bad = make_run(3, status="error", name="refund")
    store.save_run(ok)
    store.save_run(bad)
    assert [r.id for r in store.list()] == [bad.id, ok.id]
    errors = store.list(RunFilter(status="error"))
    assert [r.id for r in errors] == [bad.id]
    assert errors[0].n_steps == 3
    assert errors[0].cost == pytest.approx(0.001 * (0 + 1 + 2))
    assert [r.id for r in store.list(RunFilter(name="cleanup"))] == [ok.id]


def test_artifacts_text_and_json(store: RunStore) -> None:
    run = make_run(1)
    store.save_run(run)
    store.save_artifact(run.id, "report.html", "<html></html>")
    assert store.load_artifact(run.id, "report.html") == "<html></html>"


def test_resolve_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTBISECT_HOME", raising=False)
    project = tmp_path / "proj"
    nested = project / "a" / "b"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("")
    assert resolve_home(nested) == (project / ".agentbisect").resolve()

    (project / "a" / ".agentbisect").mkdir()
    assert resolve_home(nested) == (project / "a" / ".agentbisect").resolve()

    monkeypatch.setenv("AGENTBISECT_HOME", str(tmp_path / "custom"))
    assert resolve_home(nested) == (tmp_path / "custom").resolve()
