"""Program-level replay: re-run the recorded agent with the SDK in replay mode (BLUEPRINT 6.4).

The run's `entrypoint` is imported and called with `run.input`. Each intercepted LLM call is
matched with the recorded LLM call at the same position by `request_hash`:

* match, and before `fork_at`: the recorded response is served (through a copy of the user's
  client whose HTTP layer replays it, so the agent gets real SDK objects, streaming included);
* mismatch: `strict` raises `ReplayDivergence` with a readable diff; `fork` goes live from there;
* at or after `fork_at`: live.

Tools: `cassette` serves recorded results for matching name+args and raises `ToolNotInCassette`
otherwise; `live` runs them (recorded results are still served before `fork_at`);
`cassette-or-live` serves recorded results when they match and runs the rest live. Tools with
`side_effects=True` never run live unless `allow_live_side_effects=True`.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from agentbisect.providers.fake import SEED_OFFSET
from agentbisect.replay.cassette import (
    Cassette,
    ReplayDivergence,
    SideEffectRefused,
    ToolNotInCassette,
    divergence_reason,
    request_diff,
)
from agentbisect.replay.determinism import frozen
from agentbisect.schema import Run, Step, request_hash

ReplayMode = Literal["strict", "fork"]
ToolsMode = Literal["cassette", "live", "cassette-or-live"]


class ReplayError(Exception):
    """The run cannot be replayed (no entrypoint, import error, ...)."""


class RecordedToolError(Exception):
    """The recorded tool call raised; replay raises the same error again."""


class Divergence(BaseModel):
    step_id: str | None
    call_index: int
    reason: str
    diff: str


class ReplayResult(BaseModel):
    original_id: str
    run: Run | None
    mode: ReplayMode
    fork_at: int | None
    tools: ToolsMode
    seed: int
    served_llm_calls: int
    live_llm_calls: int
    served_tool_calls: int
    live_tool_calls: int
    refused_tool_calls: int
    divergence: Divergence | None = None
    error: str | None = None
    output_matches: bool | None = None

    @property
    def ok(self) -> bool:
        return self.divergence is None and self.error is None


class _MemoryStore:
    """Keeps replayed runs in memory only (used by bisect's many trials)."""

    def start_run(self, run: Run) -> None: ...
    def append_step(self, step: Step) -> None: ...
    def finish_run(self, run: Run) -> None: ...
    def save_run(self, run: Run) -> None: ...


class ReplayController:
    def __init__(
        self,
        run: Run,
        *,
        mode: ReplayMode,
        fork_at: int | None,
        tools: ToolsMode,
        allow_live_side_effects: bool,
        persist: bool,
    ) -> None:
        self.run = run
        self.cassette = Cassette(run)
        self.mode = mode
        self.fork_at = fork_at
        self.tools = tools
        self.allow_live_side_effects = allow_live_side_effects
        self.persist = persist
        self.live = False
        self.llm_index = 0
        self.served_llm = self.live_llm = 0
        self.served_tools = self.live_tools = self.refused_tools = 0
        self.divergence: ReplayDivergence | None = None
        self.recorders: list[Any] = []
        self._pending: list[dict[str, Any] | None] = []
        self._clients: dict[int, Any] = {}

    # ---- hooks called by the SDK ---------------------------------------------------------
    def on_run_start(self, rec: Any) -> None:
        rec.label("replay_of", self.run.id)
        rec.label(
            "replay_mode", self.mode + (f"@{self.fork_at}" if self.fork_at is not None else "")
        )
        if not self.persist:
            rec._store = _MemoryStore()
        self.recorders.append(rec)

    def route_llm(
        self,
        adapter: Any,
        fn: Callable[..., Any],
        kwargs: dict[str, Any],
        *,
        root: Any,
        path: tuple[str, ...],
    ) -> tuple[Callable[..., Any], dict[str, str]]:
        i = self.llm_index
        self.llm_index += 1
        step = self.cassette.llm_step(i)
        if self.live:
            return self._go_live(fn)
        if step is None:
            if self.mode == "strict":
                raise self._diverge(
                    None,
                    i,
                    "",
                    "the replay made more LLM calls than the "
                    f"{len(self.cassette.llm_steps)} that were recorded",
                )
            self.live = True
            return self._go_live(fn)
        if self.fork_at is not None and step.index >= self.fork_at:
            self.live = True
            return self._go_live(fn)
        replayed = adapter.request_from_native(kwargs).to_dict()
        if step.provider != adapter.provider or request_hash(replayed) != step.request_hash:
            if self.mode == "strict":
                raise self._diverge(
                    step,
                    i,
                    request_diff(step.request, replayed),
                    divergence_reason(step.request, replayed),
                )
            self.live = True
            return self._go_live(fn)
        target = self._cassette_client(root, adapter.provider)
        for name in path:
            target = getattr(target, name)

        def served(*args: Any, **kw: Any) -> Any:
            self._pending.append(step.raw_response)
            return target(*args, **kw)

        self.served_llm += 1
        return served, {"replay": "cassette", "replay_of": step.id}

    def route_tool(self, info: Any, args: dict[str, Any]) -> tuple[bool, Any, dict[str, str]]:
        from agentbisect._json import to_jsonable

        jargs = to_jsonable(args)
        step = None
        if self.tools in ("cassette", "cassette-or-live"):
            step = self.cassette.find_tool(info.name, jargs)
        elif self.fork_at is not None and not self.live:
            step = self.cassette.find_tool(info.name, jargs, before=self.fork_at)
        if step is not None:
            self.served_tools += 1
            if step.error is not None:
                raise RecordedToolError(f"{step.error.type}: {step.error.message}")
            result = (step.response or {}).get("result")
            return True, result, {"replay": "cassette", "replay_of": step.id}
        if self.tools == "cassette":
            raise ToolNotInCassette(f"no recorded call of {info.name} with arguments {jargs}")
        if info.side_effects and not self.allow_live_side_effects:
            self.refused_tools += 1
            raise SideEffectRefused(
                f"{info.name} has side effects and would run live during replay; "
                "pass --allow-live-side-effects to allow it"
            )
        self.live_tools += 1
        return False, None, {"replay": "live"}

    # ---- internals -----------------------------------------------------------------------
    def _go_live(self, fn: Callable[..., Any]) -> tuple[Callable[..., Any], dict[str, str]]:
        self.live_llm += 1
        return fn, {"replay": "live"}

    def _diverge(self, step: Step | None, index: int, diff: str, reason: str) -> ReplayDivergence:
        exc = ReplayDivergence(step, index, diff, reason)
        self.divergence = self.divergence or exc
        return exc

    def _serve(self) -> dict[str, Any] | None:
        return self._pending.pop(0) if self._pending else None

    def _cassette_client(self, root: Any, provider: str) -> Any:
        from agentbisect.providers.cassette_http import cassette_client

        key = id(root)
        if key not in self._clients:
            self._clients[key] = cassette_client(root, provider, self._serve)
        return self._clients[key]


def load_entrypoint(ref: str) -> Callable[..., Any]:
    from agentbisect._loader import load_attr

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        fn = load_attr(ref)
    except Exception as exc:
        raise ReplayError(f"cannot import entrypoint {ref!r}: {exc}") from exc
    if not callable(fn):
        raise ReplayError(f"entrypoint {ref!r} is not callable")
    return fn  # type: ignore[no-any-return]


def replay_program(
    run: Run,
    *,
    mode: ReplayMode = "strict",
    fork_at: int | None = None,
    tools: ToolsMode = "cassette",
    allow_live_side_effects: bool = False,
    seed: int = 0,
    persist: bool = True,
) -> ReplayResult:
    """Re-run `run`'s agent; see the module docstring for the rules."""
    from agentbisect.sdk.context import _replay

    if not run.entrypoint:
        raise ReplayError(
            f"run {run.id} has no entrypoint; decorate the agent's entry function "
            "with @relay.entrypoint so it can be replayed"
        )
    fn = load_entrypoint(run.entrypoint)
    controller = ReplayController(
        run,
        mode=mode,
        fork_at=fork_at,
        tools=tools,
        allow_live_side_effects=allow_live_side_effects,
        persist=persist,
    )
    token = _replay.set(controller)
    offset = SEED_OFFSET.set(seed)
    error: str | None = None
    try:
        module = sys.modules.get(getattr(fn, "__module__", "") or "")
        with frozen(run.id, run.started_at, seed, modules=[module] if module else []):
            result = fn(**run.input)
            if inspect.isawaitable(result):
                asyncio.run(_await(result))
    except ReplayDivergence:
        pass  # recorded on the controller
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        SEED_OFFSET.reset(offset)
        _replay.reset(token)
    recorded_llm = len(controller.cassette.llm_steps)
    if (
        controller.divergence is None
        and mode == "strict"
        and error is None
        and controller.llm_index < recorded_llm
    ):
        controller.divergence = ReplayDivergence(
            None,
            controller.llm_index,
            "",
            f"the replay made {controller.llm_index} of the {recorded_llm} recorded LLM calls",
        )
    new_run = controller.recorders[-1].run if controller.recorders else None
    d = controller.divergence
    return ReplayResult(
        original_id=run.id,
        run=new_run,
        mode=mode,
        fork_at=fork_at,
        tools=tools,
        seed=seed,
        served_llm_calls=controller.served_llm,
        live_llm_calls=controller.live_llm,
        served_tool_calls=controller.served_tools,
        live_tool_calls=controller.live_tools,
        refused_tool_calls=controller.refused_tools,
        divergence=Divergence(
            step_id=d.step_id, call_index=d.call_index, reason=d.reason, diff=d.diff
        )
        if d
        else None,
        error=error,
        output_matches=(new_run.output == run.output) if new_run is not None else None,
    )


async def _await(awaitable: Any) -> Any:
    return await awaitable
