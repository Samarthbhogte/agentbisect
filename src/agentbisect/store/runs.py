"""Run traces on disk: `runs/<id>/run.json`, `steps.jsonl`, `artifacts/`, plus `blobs/`.

Steps are appended one JSON line at a time and flushed, so a crashed run keeps every step
written before the crash. Any string longer than 16 KB is stored once under `blobs/<sha256>`
and replaced by ``{"$blob": "<sha256>"}``; reading restores it transparently.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentbisect.config import load_config
from agentbisect.schema import SCHEMA_VERSION, Run, Step, sha256_hex
from agentbisect.sdk.redact import Redactor
from agentbisect.store.index import RunFilter, RunIndex, RunSummary

BLOB_THRESHOLD_BYTES = 16 * 1024
BLOB_KEY = "$blob"
HOME_ENV = "AGENTBISECT_HOME"


class RunNotFound(KeyError):
    """No run matches the given id, prefix, or `last`."""


class AmbiguousRunRef(ValueError):
    """A run id prefix matches more than one run."""


def resolve_home(start: Path | None = None) -> Path:
    """Locate `.agentbisect/`.

    `AGENTBISECT_HOME` wins. Otherwise use the nearest parent of `start` (default: cwd) that
    contains `.agentbisect/` or `pyproject.toml`; failing that, `start/.agentbisect`.
    """
    env = os.environ.get(HOME_ENV)
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if (directory / ".agentbisect").is_dir() or (directory / "pyproject.toml").is_file():
            return directory / ".agentbisect"
    return here / ".agentbisect"


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


class BlobStore:
    """Content-addressed storage for large strings."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, value: str) -> str:
        digest = sha256_hex(value)
        path = self.root / digest
        if not path.exists():
            _atomic_write(path, value)
        return digest

    def get(self, digest: str) -> str:
        return (self.root / digest).read_text(encoding="utf-8")

    def externalize(self, obj: Any) -> Any:
        """Replace every string over the threshold with a blob reference."""
        if isinstance(obj, str):
            if len(obj.encode("utf-8")) > BLOB_THRESHOLD_BYTES:
                return {BLOB_KEY: self.put(obj)}
            return obj
        if isinstance(obj, dict):
            return {k: self.externalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.externalize(v) for v in obj]
        return obj

    def restore(self, obj: Any) -> Any:
        """Inverse of `externalize`."""
        if isinstance(obj, dict):
            if len(obj) == 1 and isinstance(obj.get(BLOB_KEY), str):
                return self.get(obj[BLOB_KEY])
            return {k: self.restore(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.restore(v) for v in obj]
        return obj


class RunStore:
    """Reads and writes runs. Every value is redacted before it is written.

    The default redactor uses the built-in secret patterns plus `agentbisect.toml` settings.
    """

    def __init__(self, home: Path | None = None, *, redactor: Redactor | None = None) -> None:
        self.home = home if home is not None else resolve_home()
        self.runs_dir = self.home / "runs"
        self.blobs = BlobStore(self.home / "blobs")
        self.index = RunIndex(self.home / "index.db")
        if redactor is None:
            cfg = load_config(self.home)
            redactor = Redactor(pii=cfg.redact_pii, extra_patterns=cfg.redact_patterns)
        self.redactor = redactor

    def _prepare(self, obj: Any) -> Any:
        """Redact, then move large strings to blobs. Everything written goes through here."""
        return self.blobs.externalize(self.redactor.redact(obj))

    # ---- paths -------------------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def _steps_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "steps.jsonl"

    # ---- writing -----------------------------------------------------------------------
    def start_run(self, run: Run) -> None:
        """Create the run directory, write `run.json`, and index the run."""
        steps_path = self._steps_path(run.id)
        steps_path.parent.mkdir(parents=True, exist_ok=True)
        if not steps_path.exists():
            steps_path.write_text(_header_line(), encoding="utf-8")
        self._write_meta(run)
        self.index.upsert(_summary(run))

    def append_step(self, step: Step) -> None:
        """Append one step and flush, so it survives a crash."""
        line = json.dumps(self._prepare(step.model_dump(mode="json")), ensure_ascii=False)
        with self._steps_path(step.run_id).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    def finish_run(self, run: Run) -> None:
        """Rewrite `run.json` with final status/output and refresh the index."""
        self._write_meta(run)
        self.index.upsert(_summary(run))

    def save_run(self, run: Run) -> None:
        """Write a complete run (metadata and all steps) in one go."""
        lines = [_header_line()]
        for step in run.steps:
            dumped = self._prepare(step.model_dump(mode="json"))
            lines.append(json.dumps(dumped, ensure_ascii=False) + "\n")
        _atomic_write(self._steps_path(run.id), "".join(lines))
        self.finish_run(run)

    def _write_meta(self, run: Run) -> None:
        meta = run.model_dump(mode="json", by_alias=True, exclude={"steps"})
        data = json.dumps(self._prepare(meta), ensure_ascii=False, indent=2)
        _atomic_write(self.run_dir(run.id) / "run.json", data)

    # ---- reading -----------------------------------------------------------------------
    def load(self, ref: str) -> Run:
        run_id = self.resolve(ref)
        meta_path = self.run_dir(run_id) / "run.json"
        meta = self.blobs.restore(json.loads(meta_path.read_text(encoding="utf-8")))
        steps = sorted(self._read_steps(run_id), key=lambda s: s.index)
        return Run.model_validate({**meta, "steps": steps})

    def _read_steps(self, run_id: str) -> Iterator[Step]:
        path = self._steps_path(run_id)
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    return  # torn final write from a crash; earlier steps are intact
                raise
            if "id" not in obj:
                continue  # header line
            yield Step.model_validate(self.blobs.restore(obj))

    def run_ids(self) -> list[str]:
        """All run ids on disk, oldest first (ULIDs sort by time)."""
        if not self.runs_dir.is_dir():
            return []
        return sorted(p.name for p in self.runs_dir.iterdir() if (p / "run.json").is_file())

    def resolve(self, ref: str) -> str:
        """Turn a full id, a unique prefix, or `last` into a run id."""
        ids = self.run_ids()
        if ref == "last":
            if not ids:
                raise RunNotFound("no runs recorded yet")
            return self._last_original(ids)
        if ref in ids:
            return ref
        matches = [i for i in ids if i.startswith(ref.upper())]
        if not matches:
            raise RunNotFound(f"no run matches {ref!r}")
        if len(matches) > 1:
            raise AmbiguousRunRef(f"{ref!r} matches {len(matches)} runs: {', '.join(matches[:5])}")
        return matches[0]

    def _last_original(self, ids: list[str]) -> str:
        """The newest run that is not a replay (replays are labeled `replay_of`)."""
        for run_id in reversed(ids):
            try:
                meta = json.loads((self.run_dir(run_id) / "run.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if "replay_of" not in (meta.get("labels") or {}):
                return run_id
        return ids[-1]

    def list(self, flt: RunFilter | None = None) -> list[RunSummary]:
        return self.index.query(flt)

    # ---- artifacts ---------------------------------------------------------------------
    def save_artifact(self, run_id: str, name: str, obj: BaseModel | dict[str, Any] | str) -> Path:
        """Save `obj` under `artifacts/<name>`. Strings are written as-is (e.g. HTML)."""
        path = self.run_dir(self.resolve(run_id)) / "artifacts" / name
        if isinstance(obj, str):
            _atomic_write(path, self.redactor.redact_text(obj))
            return path
        data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else dict(obj)
        payload = self.redactor.redact({"schema": SCHEMA_VERSION, **data})
        _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))
        return path

    def load_artifact(self, run_id: str, name: str) -> Any:
        path = self.run_dir(self.resolve(run_id)) / "artifacts" / name
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if path.suffix == ".json" else text


def _header_line() -> str:
    return json.dumps({"schema": SCHEMA_VERSION}) + "\n"


def _summary(run: Run) -> RunSummary:
    costs = [s.cost_usd for s in run.steps if s.cost_usd is not None]
    return RunSummary(
        id=run.id,
        name=run.name,
        status=run.status,
        started_at=run.started_at,
        n_steps=len(run.steps),
        cost=sum(costs) if costs else None,
        labels=run.labels,
    )
