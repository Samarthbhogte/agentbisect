"""`relay diff`: align two runs' steps with LCS over (kind, name) (BLUEPRINT 6.4)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from agentrelay.replay.cassette import request_diff
from agentrelay.schema import Run, Step, canonical_json

DiffKind = Literal["same", "changed", "removed", "inserted"]


class StepDiff(BaseModel):
    kind: DiffKind
    a: str | None = None
    b: str | None = None
    label: str
    request_diff: str = ""
    response_diff: str = ""


class RunDiff(BaseModel):
    a: str
    b: str
    steps: list[StepDiff]
    counts: dict[str, int]
    deltas: dict[str, float | None]


def _key(s: Step) -> tuple[str, str]:
    return s.kind, s.name


def lcs_align(a: list[Step], b: list[Step]) -> list[tuple[Step | None, Step | None]]:
    n, m = len(a), len(b)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            table[i][j] = (
                table[i + 1][j + 1] + 1
                if _key(a[i]) == _key(b[j])
                else max(table[i + 1][j], table[i][j + 1])
            )
    out: list[tuple[Step | None, Step | None]] = []
    i = j = 0
    while i < n and j < m:
        if _key(a[i]) == _key(b[j]):
            out.append((a[i], b[j]))
            i, j = i + 1, j + 1
        elif table[i + 1][j] >= table[i][j + 1]:
            out.append((a[i], None))
            i += 1
        else:
            out.append((None, b[j]))
            j += 1
    out += [(s, None) for s in a[i:]] + [(None, s) for s in b[j:]]
    return out


def _total(run: Run, attr: str) -> float | None:
    values = [getattr(s, attr) for s in run.steps if getattr(s, attr) is not None]
    return float(sum(values)) if values else None


def _duration(run: Run) -> float | None:
    return (run.ended_at - run.started_at).total_seconds() if run.ended_at else None


def _delta(x: float | None, y: float | None) -> float | None:
    return None if x is None or y is None else y - x


def diff_runs(a: Run, b: Run) -> RunDiff:
    steps: list[StepDiff] = []
    for sa, sb in lcs_align(a.steps, b.steps):
        if sa is not None and sb is not None:
            same_req = sa.request_hash == sb.request_hash
            same_resp = canonical_json(sa.response) == canonical_json(sb.response)
            same_err = canonical_json(sa.error) == canonical_json(sb.error)
            kind: DiffKind = "same" if same_req and same_resp and same_err else "changed"
            steps.append(
                StepDiff(
                    kind=kind,
                    a=sa.id,
                    b=sb.id,
                    label=f"{sa.kind} {sa.name}",
                    request_diff="" if same_req else request_diff(sa.request, sb.request),
                    response_diff=""
                    if same_resp
                    else request_diff(
                        {"response": sa.response, "error": canonical_json(sa.error)},
                        {"response": sb.response, "error": canonical_json(sb.error)},
                    ),
                )
            )
        elif sa is not None:
            steps.append(StepDiff(kind="removed", a=sa.id, label=f"{sa.kind} {sa.name}"))
        elif sb is not None:
            steps.append(StepDiff(kind="inserted", b=sb.id, label=f"{sb.kind} {sb.name}"))
    counts = {
        k: sum(s.kind == k for s in steps) for k in ("same", "changed", "removed", "inserted")
    }
    deltas = {
        "cost_usd": _delta(_total(a, "cost_usd"), _total(b, "cost_usd")),
        "tokens_in": _delta(_total(a, "tokens_in"), _total(b, "tokens_in")),
        "tokens_out": _delta(_total(a, "tokens_out"), _total(b, "tokens_out")),
        "duration_s": _delta(_duration(a), _duration(b)),
        "steps": float(len(b.steps) - len(a.steps)),
    }
    return RunDiff(a=a.id, b=b.id, steps=steps, counts=counts, deltas=deltas)
