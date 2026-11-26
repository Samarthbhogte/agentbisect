"""ddmin properties (BLUEPRINT 6.8) with a deterministic oracle: fails iff all culprits present."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from agentbisect.algo.ddmin import ddmin
from agentbisect.schema import ContextUnit


def unit(
    uid: str, *, removable: bool = True, group: str | None = None, parent: str | None = None
) -> ContextUnit:
    return ContextUnit(
        id=uid,
        kind="chunk" if parent else "message",
        path="/",
        text=uid,
        tokens=10,
        group=group,
        parent=parent,
        removable=removable,
    )


@st.composite
def scenario(draw: Any) -> tuple[list[ContextUnit], set[str]]:
    n = draw(st.integers(1, 30))
    units: list[ContextUnit] = []
    group_of: dict[int, str] = {}
    i = 0
    while i < n:  # some consecutive units share a group
        if draw(st.integers(0, 4)) == 0 and i + 1 < n:
            group_of[i] = group_of[i + 1] = f"g{i}"
            i += 2
        else:
            i += 1
    for i in range(n):
        removable = draw(st.integers(0, 6)) > 0
        units.append(unit(f"u{i}", removable=removable or i in group_of, group=group_of.get(i)))
        if draw(st.integers(0, 3)) == 0:
            for c in range(draw(st.integers(1, 6))):
                units.append(unit(f"u{i}.c{c}", parent=f"u{i}"))
    candidates = [u.id for u in units if u.removable]
    culprits = set(
        draw(st.lists(st.sampled_from(candidates), max_size=3, unique=True)) if candidates else []
    )
    return units, culprits


def alive(keep: frozenset[str], units: list[ContextUnit]) -> set[str]:
    parents = {u.id: u.parent for u in units}
    return {i for i in keep if parents.get(i) is None or parents[i] in keep}


def expected_minimal(units: list[ContextUnit], culprits: set[str]) -> set[str]:
    by_id = {u.id: u for u in units}
    need = {u.id for u in units if not u.removable}
    for c in culprits:
        need.add(c)
        if by_id[c].parent:
            need.add(by_id[c].parent)  # type: ignore[arg-type]
    for uid in list(need):  # whole groups
        g = by_id[uid].group
        if g:
            need |= {u.id for u in units if u.group == g}
    return need


@settings(max_examples=400)
@given(scenario())
def test_ddmin_properties(case: tuple[list[ContextUnit], set[str]]) -> None:
    units, culprits = case
    calls: list[frozenset[str]] = []

    async def still_fails(keep: frozenset[str]) -> bool:
        calls.append(keep)
        return culprits <= alive(keep, units)

    result = asyncio.run(ddmin(units, still_fails))
    minimal = set(result.minimal_ids)
    ids = {u.id for u in units}
    by_id = {u.id: u for u in units}

    assert minimal <= ids  # result ⊆ input
    assert culprits <= alive(frozenset(minimal), units)  # still fails
    assert {u.id for u in units if not u.removable} <= minimal  # non-removable kept
    for g in {u.group for u in units if u.group}:  # groups never split
        members = {u.id for u in units if u.group == g}
        assert members <= minimal or not members & minimal
    assert minimal == expected_minimal(units, culprits)  # exactly the culprits
    for uid in minimal:  # 1-minimal
        u = by_id[uid]
        if not u.removable:
            continue
        group = {x.id for x in units if u.group and x.group == u.group} or {uid}
        without = frozenset(minimal - group)
        if (
            culprits
            and not group & culprits
            and not any(by_id[c].parent in group for c in culprits)
        ):
            continue  # removing a non-culprit container (e.g. a chunk's parent's group mate)
        assert not culprits <= alive(without, units), uid
    assert len(calls) == len(set(calls)) == result.candidates_tested  # memoized


def test_one_culprit_among_many_needs_few_questions() -> None:
    units = [unit(f"u{i}") for i in range(64)]

    async def still_fails(keep: frozenset[str]) -> bool:
        return "u37" in keep

    result = asyncio.run(ddmin(units, still_fails))
    assert result.minimal_ids == ["u37"]
    assert result.candidates_tested <= 2 * 6 + 2  # ~2 log2(n): far fewer than 64 single checks


def test_two_culprits_interaction() -> None:
    units = [unit(f"u{i}") for i in range(20)]

    async def still_fails(keep: frozenset[str]) -> bool:
        return {"u3", "u15"} <= keep

    assert asyncio.run(ddmin(units, still_fails)).minimal_ids == ["u15", "u3"]


def test_noisy_oracle_terminates_and_asks_each_question_once() -> None:
    units = [unit(f"u{i}") for i in range(25)]
    rng = random.Random(7)
    asked: list[frozenset[str]] = []

    async def coin(keep: frozenset[str]) -> bool:
        asked.append(keep)
        return rng.random() < 0.5

    result = asyncio.run(ddmin(units, coin))
    assert len(asked) == len(set(asked)) == result.candidates_tested
    assert set(result.minimal_ids) <= {u.id for u in units}


def test_on_accept_reports_each_new_failing_candidate() -> None:
    units = [unit(f"u{i}") for i in range(8)]
    seen: list[frozenset[str]] = []

    async def still_fails(keep: frozenset[str]) -> bool:
        return "u5" in keep

    result = asyncio.run(ddmin(units, still_fails, on_accept=seen.append))
    assert [sorted(s) for s in seen] == result.accepted
    assert seen and all("u5" in s for s in seen) and len(seen[-1]) == 1
