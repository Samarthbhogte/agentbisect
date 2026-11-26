"""Probabilistic ddmin (Zeller & Hildebrandt) over context units (BLUEPRINT 6.8).

Works on *items*: each item is a removable group of unit ids, never split. Non-removable units
are always kept. Two passes: first over top-level units (sections, messages, tool results, tool
specs), then over the chunks of the tool results that survived. `still_fails` is memoized by the
exact set of ids it is asked about, so a noisy oracle is asked each question once.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from agentbisect.schema import ContextUnit

StillFails = Callable[[frozenset[str]], Awaitable[bool]]
OnAccept = Callable[[frozenset[str]], None]


class DDMinStep(BaseModel):
    """One question asked of the oracle, for the report's step log."""

    level: Literal["units", "chunks"]
    n: int
    test: Literal["empty", "subset", "complement"]
    candidate_items: int
    candidate_tokens: int
    fails: bool
    cached: bool


class DDMinResult(BaseModel):
    minimal_ids: list[str]
    candidates_tested: int
    accepted: list[list[str]] = Field(default_factory=list)  # keep sets that failed, in order
    log: list[DDMinStep] = Field(default_factory=list)


Item = frozenset[str]


def _items(units: Sequence[ContextUnit]) -> list[Item]:
    """Removable units grouped by `group` (singletons otherwise), in first-seen order."""
    order: list[str] = []
    groups: dict[str, set[str]] = {}
    for u in units:
        if not u.removable:
            continue
        key = u.group or f"unit:{u.id}"
        if key not in groups:
            groups[key] = set()
            order.append(key)
        groups[key].add(u.id)
    return [frozenset(groups[k]) for k in order]


def _split(items: list[Item], n: int) -> list[list[Item]]:
    size, extra = divmod(len(items), n)
    out, start = [], 0
    for i in range(n):
        end = start + size + (1 if i < extra else 0)
        out.append(items[start:end])
        start = end
    return [c for c in out if c]


class _Search:
    def __init__(
        self, units: Sequence[ContextUnit], still_fails: StillFails, on_accept: OnAccept | None
    ) -> None:
        self.units = {u.id: u for u in units}
        self.children: dict[str, list[str]] = {}
        for u in units:
            if u.parent:
                self.children.setdefault(u.parent, []).append(u.id)
        self.still_fails = still_fails
        self.on_accept = on_accept
        self.memo: dict[frozenset[str], bool] = {}
        self.log: list[DDMinStep] = []
        self.accepted: list[list[str]] = []

    def tokens(self, ids: frozenset[str]) -> int:
        chunked = {p for p in self.children if p in ids}
        return sum(self.units[i].tokens for i in ids if i in self.units and i not in chunked)

    async def ask(
        self,
        keep: frozenset[str],
        *,
        level: Literal["units", "chunks"],
        n: int,
        test: Literal["empty", "subset", "complement"],
        items: int,
    ) -> bool:
        cached = keep in self.memo
        if not cached:
            self.memo[keep] = await self.still_fails(keep)
        fails = self.memo[keep]
        self.log.append(
            DDMinStep(
                level=level,
                n=n,
                test=test,
                candidate_items=items,
                candidate_tokens=self.tokens(keep),
                fails=fails,
                cached=cached,
            )
        )
        if fails and not cached:
            self.accepted.append(sorted(keep))
            if self.on_accept is not None:
                self.on_accept(keep)
        return fails

    async def run(
        self,
        items: list[Item],
        fixed: frozenset[str],
        expand: Callable[[frozenset[str]], frozenset[str]],
        level: Literal["units", "chunks"],
    ) -> list[Item]:
        """ddmin over `items`; each candidate is `fixed | expand(union of chosen items)`."""

        def keep(chosen: list[Item]) -> frozenset[str]:
            ids: frozenset[str] = frozenset().union(*chosen) if chosen else frozenset()
            return expand(fixed | ids)  # children of always-kept units count too

        current = list(items)
        if not current:
            return current
        if await self.ask(keep([]), level=level, n=0, test="empty", items=0):
            return []
        n = 2
        while len(current) >= 2:
            chunks = _split(current, min(n, len(current)))
            reduced = False
            for delta in chunks:
                if await self.ask(keep(delta), level=level, n=n, test="subset", items=len(delta)):
                    current, n, reduced = delta, 2, True
                    break
            if not reduced and len(chunks) > 2:
                for delta in chunks:
                    rest = [it for it in current if it not in delta]
                    if await self.ask(
                        keep(rest), level=level, n=n, test="complement", items=len(rest)
                    ):
                        current, n, reduced = rest, max(n - 1, 2), True
                        break
            if reduced:
                continue
            if n >= len(current):
                break
            n = min(2 * n, len(current))
        return current


async def ddmin(
    units: list[ContextUnit], still_fails: StillFails, *, on_accept: OnAccept | None = None
) -> DDMinResult:
    """Smallest set of unit ids (including non-removable ones) that still fails.

    Precondition: keeping every unit fails. Pass 1 works on top-level units, keeping all chunks of
    every tool result it keeps; pass 2 then shrinks the chunks of the surviving tool results.
    """
    search = _Search(units, still_fails, on_accept)
    fixed = frozenset(u.id for u in units if not u.removable)
    top = [u for u in units if u.parent is None]

    def with_children(ids: frozenset[str]) -> frozenset[str]:
        return ids | frozenset(c for p in ids for c in search.children.get(p, []))

    survivors = await search.run(_items(top), fixed, with_children, "units")
    kept_top: frozenset[str] = fixed.union(*survivors) if survivors else fixed
    chunk_units = [u for u in units if u.parent in kept_top]
    if chunk_units:
        chunk_items = await search.run(_items(chunk_units), kept_top, lambda ids: ids, "chunks")
        minimal: frozenset[str] = kept_top.union(*chunk_items) if chunk_items else kept_top
    else:
        minimal = kept_top
    return DDMinResult(
        minimal_ids=sorted(minimal),
        candidates_tested=len(search.memo),
        accepted=search.accepted,
        log=search.log,
    )
