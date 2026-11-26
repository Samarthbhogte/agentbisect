"""SQLite index of runs (`index.db`), used for fast listing and filtering."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    n_steps INTEGER NOT NULL,
    cost REAL,
    labels TEXT NOT NULL
)
"""


class RunSummary(BaseModel):
    id: str
    name: str
    status: str
    started_at: datetime
    n_steps: int
    cost: float | None
    labels: dict[str, str] = Field(default_factory=dict)


class RunFilter(BaseModel):
    status: str | None = None
    name: str | None = None
    limit: int | None = None


class RunIndex:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(_DDL)
        return conn

    def upsert(self, summary: RunSummary) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    summary.id,
                    summary.name,
                    summary.status,
                    summary.started_at.isoformat(),
                    summary.n_steps,
                    summary.cost,
                    json.dumps(summary.labels, sort_keys=True),
                ),
            )

    def query(self, flt: RunFilter | None = None) -> list[RunSummary]:
        flt = flt or RunFilter()
        sql = "SELECT id, name, status, started_at, n_steps, cost, labels FROM runs"
        clauses: list[str] = []
        params: list[object] = []
        if flt.status:
            clauses.append("status = ?")
            params.append(flt.status)
        if flt.name:
            clauses.append("name = ?")
            params.append(flt.name)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC"
        if flt.limit:
            sql += " LIMIT ?"
            params.append(flt.limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            RunSummary(
                id=r[0],
                name=r[1],
                status=r[2],
                started_at=datetime.fromisoformat(r[3]),
                n_steps=r[4],
                cost=r[5],
                labels=json.loads(r[6]),
            )
            for r in rows
        ]
