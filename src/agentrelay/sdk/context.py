"""The active run, `relay.run()`, `@relay.entrypoint`, and the step recorder.

Recording must never break the user's agent: every recording path goes through `safe()`, which
logs a warning and carries on. `AGENTRELAY_DISABLED=1` turns all of it into a no-op.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import logging
import os
import platform
import sys
import threading
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

from agentrelay import __version__
from agentrelay._json import to_jsonable
from agentrelay.ids import new_ulid
from agentrelay.providers.base import LLMRequest, LLMResponse
from agentrelay.schema import ErrorInfo, Run, RunStatus, Step, request_hash, step_id
from agentrelay.sdk.deps import llm_dependencies, match_tool_use

log = logging.getLogger("agentrelay")
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

DISABLED_ENV = "AGENTRELAY_DISABLED"
RUN_NAME_ENV = "AGENTRELAY_RUN_NAME"


def disabled() -> bool:
    return os.environ.get(DISABLED_ENV, "").strip().lower() in ("1", "true", "yes")


def now() -> datetime:
    return datetime.now(UTC)


def safe(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
    """Run a recording action; on any error log a warning and return None."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # recording must never raise into user code
        log.warning(
            "agentrelay: recording failed (%s: %s); continuing without it", type(exc).__name__, exc
        )
        return None


def _env_info() -> dict[str, str]:
    info = {
        "python": platform.python_version(),
        "agentrelay": __version__,
        "platform": sys.platform,
    }
    sha = _git_sha()
    if sha:
        info["git_sha"] = sha
    return info


def _git_sha() -> str | None:
    """Read HEAD without spawning git (fast, no subprocess)."""
    for directory in (Path.cwd(), *Path.cwd().parents):
        head = directory / ".git" / "HEAD"
        if head.is_file():
            ref = head.read_text().strip()
            if ref.startswith("ref: "):
                ref_path = directory / ".git" / ref[5:]
                return ref_path.read_text().strip() if ref_path.is_file() else None
            return ref
    return None


class RunRecorder:
    """Builds one `Run` and appends its steps to the store as they happen."""

    def __init__(
        self,
        name: str,
        input: Any = None,
        *,
        entrypoint: str | None = None,
        labels: dict[str, str] | None = None,
        store: Any = None,
    ) -> None:
        self.enabled = not disabled()
        self.run = Run(
            id=new_ulid(),
            name=name,
            entrypoint=entrypoint,
            input=_as_dict(input, "input"),
            output=None,
            status="unknown",
            steps=[],
            started_at=now(),
            ended_at=None,
            env=_env_info(),
            labels=dict(labels or {}),
        )
        self._store = store
        self._lock = threading.Lock()
        self._claimed: set[str] = set()
        self._finished = False

    @property
    def id(self) -> str:
        return self.run.id

    @property
    def store(self) -> Any:
        if self._store is None:
            from agentrelay.store.runs import RunStore

            self._store = RunStore()
        return self._store

    # ---- lifecycle ------------------------------------------------------------------------
    def start(self) -> None:
        if self.enabled:
            safe(self.store.start_run, self.run)

    def set_output(self, value: Any) -> None:
        self.run.output = _as_dict(value, "return")

    def label(self, key: str, value: str) -> None:
        self.run.labels[key] = value

    def finish(self, status: RunStatus, error: BaseException | None = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.run.status = status
            self.run.ended_at = now()
            if error is not None and self.run.output is None:
                self.run.output = {"error": {"type": type(error).__name__, "message": str(error)}}
            self._finished = True
        safe(self.store.finish_run, self.run)

    # ---- steps ----------------------------------------------------------------------------
    def record_llm(
        self,
        *,
        provider: str,
        request: LLMRequest,
        response: LLMResponse | None,
        raw_request: dict[str, Any] | None,
        raw_response: dict[str, Any] | None,
        error: BaseException | None,
        started_at: datetime,
        ended_at: datetime,
        cost_usd: float | None = None,
    ) -> Step | None:
        if not self.enabled:
            return None
        req_dict = request.to_dict()
        with self._lock:
            index = len(self.run.steps)
            step = Step(
                id=step_id(index),
                run_id=self.run.id,
                index=index,
                kind="llm_call",
                name=request.model,
                provider=provider,
                request=req_dict,
                response=response.to_dict() if response else None,
                raw_request=raw_request,
                raw_response=raw_response,
                error=_error(error),
                depends_on=llm_dependencies(request, self.run.steps),
                started_at=started_at,
                ended_at=ended_at,
                tokens_in=response.tokens_in if response else None,
                tokens_out=response.tokens_out if response else None,
                cost_usd=cost_usd,
                request_hash=request_hash(req_dict),
            )
            self.run.steps.append(step)
        self.store.append_step(step)
        return step

    def record_tool(
        self,
        *,
        name: str,
        args: dict[str, Any],
        result: Any,
        error: BaseException | None,
        started_at: datetime,
        ended_at: datetime,
        side_effects: bool,
    ) -> Step | None:
        if not self.enabled:
            return None
        args_json = to_jsonable(args)
        with self._lock:
            index = len(self.run.steps)
            match = match_tool_use(name, args_json, self.run.steps, self._claimed)
            tags = {"side_effects": str(side_effects).lower()}
            depends_on: list[str] = []
            if match:
                self._claimed.add(match[0])
                tags["tool_use_id"] = match[0]
                depends_on = [match[1]]
            step = Step(
                id=step_id(index),
                run_id=self.run.id,
                index=index,
                kind="tool_call",
                name=name,
                request=args_json,
                response=None if error else {"result": to_jsonable(result)},
                raw_request=None,
                raw_response=None,
                error=_error(error),
                depends_on=depends_on,
                started_at=started_at,
                ended_at=ended_at,
                request_hash=request_hash(args_json),
                tags=tags,
            )
            self.run.steps.append(step)
        self.store.append_step(step)
        return step


def _error(exc: BaseException | None) -> ErrorInfo | None:
    return ErrorInfo(type=type(exc).__name__, message=str(exc)) if exc else None


def _as_dict(value: Any, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    data = to_jsonable(value)
    return data if isinstance(data, dict) else {key: data}


# ---- the active run -----------------------------------------------------------------------
_current: ContextVar[RunRecorder | None] = ContextVar("agentrelay_run", default=None)
_active: list[RunRecorder] = []  # process-wide, so calls made from worker threads still land
_active_lock = threading.Lock()
_implicit: RunRecorder | None = None
_entry: ContextVar[_EntryContext | None] = ContextVar("agentrelay_entry", default=None)


def _implicit_name() -> str:
    explicit = os.environ.get(RUN_NAME_ENV)
    if explicit:
        return explicit
    argv0 = sys.argv[0] if sys.argv and sys.argv[0] else ""
    return Path(argv0).stem or "implicit"


def current_recorder() -> RunRecorder | None:
    """The run new steps go to: the context's run, else any active run, else an implicit one."""
    rec = _current.get()
    if rec is not None:
        return rec
    with _active_lock:
        if _active:
            return _active[-1]
    return _get_implicit()


def _get_implicit() -> RunRecorder:
    global _implicit
    with _active_lock:
        if _implicit is None:
            _implicit = RunRecorder(_implicit_name())
            _implicit.start()
            atexit.register(_finish_implicit)
        return _implicit


def _finish_implicit() -> None:
    global _implicit
    rec, _implicit = _implicit, None
    if rec is not None:
        safe(rec.finish, "unknown")


def reset_state() -> None:
    """Finish the implicit run and forget all active runs (for tests)."""
    _finish_implicit()
    with _active_lock:
        _active.clear()


class run:
    """Context manager that records everything inside it as one run.

    Works with `with` and `async with`. The returned recorder has `set_output()`, `label()`
    and `id`; when the SDK is disabled it is still returned, but writes nothing.
    """

    def __init__(
        self, name: str, input: Any = None, *, labels: dict[str, str] | None = None
    ) -> None:
        self.name = name
        self.input = input
        self.labels = labels
        self.recorder: RunRecorder | None = None
        self._token: Any = None

    def __enter__(self) -> RunRecorder:
        entry = _entry.get()
        rec = RunRecorder(
            self.name, self.input, labels=self.labels, entrypoint=entry.path if entry else None
        )
        self.recorder = rec
        if rec.enabled:
            rec.start()
            self._token = _current.set(rec)
            with _active_lock:
                _active.append(rec)
            if entry:
                entry.recorders.append(rec)
        return rec

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        rec = self.recorder
        if rec is None or not rec.enabled:
            return
        if self._token is not None:
            _current.reset(self._token)
        with _active_lock:
            if rec in _active:
                _active.remove(rec)
        rec.finish("error" if exc is not None else "ok", exc)

    async def __aenter__(self) -> RunRecorder:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)


class _EntryContext:
    def __init__(self, path: str) -> None:
        self.path = path
        self.recorders: list[RunRecorder] = []


def entrypoint_path(fn: Callable[..., Any]) -> str:
    """`module:qualname`, resolving `__main__` to its real module name when run with `-m`."""
    module = fn.__module__
    if module == "__main__":
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        if spec is not None and spec.name:
            module = spec.name
    return f"{module}:{fn.__qualname__}"


def entrypoint(fn: F) -> F:
    """Mark the function program-level replay calls. Runs started inside it record its path
    as `run.entrypoint` and, if they set no output themselves, its return value as output."""
    path = entrypoint_path(fn)

    def _complete(ctx: _EntryContext, result: Any) -> None:
        for rec in ctx.recorders:
            if rec.run.output is None:
                rec.set_output(result)
                safe(rec.store.finish_run, rec.run)

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = _EntryContext(path)
            token = _entry.set(ctx)
            try:
                result = await fn(*args, **kwargs)
            finally:
                _entry.reset(token)
            safe(_complete, ctx, result)
            return result

        async_wrapper.__agentrelay_entrypoint__ = path  # type: ignore[attr-defined]
        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = _EntryContext(path)
        token = _entry.set(ctx)
        try:
            result = fn(*args, **kwargs)
        finally:
            _entry.reset(token)
        safe(_complete, ctx, result)
        return result

    wrapper.__agentrelay_entrypoint__ = path  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
