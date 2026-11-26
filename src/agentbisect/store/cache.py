"""Trial-result cache (`cache.db`): responses stored by (key, sample_idx).

key = sha256(namespace | canonical(request) | model | temperature). The namespace identifies the
provider (and, for FakeLLM, its rule set) so different backends never share cached answers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from agentbisect.providers.base import LLMRequest, LLMResponse
from agentbisect.schema import canonical_json, sha256_hex
from agentbisect.sdk.redact import Redactor


def trial_cache_key(req: LLMRequest, *, namespace: str = "") -> str:
    return sha256_hex(
        "|".join([namespace, canonical_json(req.to_dict()), req.model, str(req.temperature)])
    )


class TrialCache:
    """Responses are redacted before they are written, and `put` returns the stored (redacted)
    response so a first run and a cached re-run judge exactly the same text."""

    def __init__(self, path: Path, *, redactor: Redactor | None = None) -> None:
        self.path = path
        self.redactor = redactor or Redactor()
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS trials (key TEXT NOT NULL, sample_idx INTEGER NOT NULL,"
                " response_json TEXT NOT NULL, created_at TEXT NOT NULL,"
                " PRIMARY KEY (key, sample_idx))"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get(self, key: str, sample_idx: int) -> LLMResponse | None:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT response_json FROM trials WHERE key = ? AND sample_idx = ?",
                (key, sample_idx),
            ).fetchone()
        return LLMResponse.model_validate(json.loads(row[0])) if row else None

    def put(self, key: str, sample_idx: int, response: LLMResponse) -> LLMResponse:
        stored = LLMResponse.model_validate(self.redactor.redact(response.to_dict()))
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO trials VALUES (?, ?, ?, ?)",
                (key, sample_idx, json.dumps(stored.to_dict()), datetime.now(UTC).isoformat()),
            )
        return stored

    def __len__(self) -> int:
        with self._lock, closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
