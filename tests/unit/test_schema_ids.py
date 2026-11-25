from __future__ import annotations

from datetime import UTC, datetime

from hypothesis import given
from hypothesis import strategies as st

from agentrelay.ids import new_ulid
from agentrelay.schema import SCHEMA_VERSION, Run, canonical_json, request_hash, step_id

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(),
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(), inner, max_size=4),
    max_leaves=20,
)


def test_step_id_is_zero_padded() -> None:
    assert step_id(7) == "s0007"
    assert step_id(123) == "s0123"


@given(st.dictionaries(st.text(), json_values, max_size=6))
def test_request_hash_ignores_key_order(d: dict[str, object]) -> None:
    reordered = dict(reversed(list(d.items())))
    assert request_hash(d) == request_hash(reordered)
    assert canonical_json(d) == canonical_json(reordered)


def test_ulids_are_unique_sortable_and_well_formed() -> None:
    ids = [new_ulid() for _ in range(2000)]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)
    assert all(len(i) == 26 for i in ids)


def test_run_dumps_schema_under_alias() -> None:
    now = datetime.now(UTC)
    run = Run(
        id=new_ulid(),
        name="r",
        entrypoint=None,
        input={},
        output=None,
        status="ok",
        steps=[],
        started_at=now,
        ended_at=now,
        env={},
    )
    dumped = run.model_dump(by_alias=True)
    assert dumped["schema"] == SCHEMA_VERSION
    assert Run.model_validate(dumped) == run
