"""`@oracle`, the registry, loading (`file.py:name`), and safe evaluation.

An exception inside an oracle is an *oracle error*, never a failure: evaluation raises
`OracleError` and the command aborts with a clear message.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from agentbisect.providers.base import LLMRequest, LLMResponse
from agentbisect.schema import Failure, Run

Level = Literal["decision", "run"]
_REGISTRY: dict[str, Oracle] = {}


class OracleError(Exception):
    """The oracle itself raised or returned something other than `Failure | None`."""


@dataclass(frozen=True)
class Oracle:
    """A judge. Decision oracles take an `LLMResponse` (and optionally the `LLMRequest`);
    run oracles take a `Run`. Either may be `async`. `noisy` marks LLM-as-judge oracles."""

    name: str
    level: Level
    fn: Callable[..., Any]
    noisy: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    @property
    def wants_request(self) -> bool:
        params = [
            p
            for p in inspect.signature(self.fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        return self.level == "decision" and len(params) >= 2


def oracle(
    name: str | None = None, *, level: Level = "decision", noisy: bool = False
) -> Callable[[Callable[..., Any]], Oracle]:
    """Register a function as an oracle: `@oracle("deletes-outside-tmp", level="decision")`."""
    if level not in ("decision", "run"):
        raise ValueError(f"level must be 'decision' or 'run', got {level!r}")

    def decorate(fn: Callable[..., Any]) -> Oracle:
        o = Oracle(name=name or fn.__name__.replace("_", "-"), level=level, fn=fn, noisy=noisy)
        _REGISTRY[o.name] = o
        return o

    return decorate


def registered() -> dict[str, Oracle]:
    return dict(_REGISTRY)


def load_oracle(ref: str, *, config_oracles: dict[str, str] | None = None) -> Oracle:
    """`path/to/oracles.py:attr`, `package.module:attr`, a name from `agentbisect.toml`
    `[oracles]`, or an already-registered oracle name. Plain functions become decision oracles."""
    from agentbisect._loader import load_attr

    if config_oracles and ref in config_oracles:
        ref = config_oracles[ref]
    elif ":" not in ref:
        if ref in _REGISTRY:
            return _REGISTRY[ref]
        raise OracleError(
            f"unknown oracle {ref!r}: use FILE.py:NAME or register it in "
            "agentbisect.toml under [oracles]"
        )
    try:
        obj = load_attr(ref)
    except Exception as exc:
        raise OracleError(f"could not load oracle {ref!r}: {exc}") from exc
    if isinstance(obj, Oracle):
        return obj
    if callable(obj):
        return Oracle(name=ref.rpartition(":")[2], level="decision", fn=obj)
    raise OracleError(f"{ref!r} is not an oracle or a function")


def _validate(o: Oracle, result: Any) -> Failure | None:
    if result is None or isinstance(result, Failure):
        return result
    raise OracleError(
        f"oracle {o.name!r} returned {type(result).__name__}; expected Failure or None"
    )


def _args(o: Oracle, subject: LLMResponse | Run, request: LLMRequest | None) -> tuple[Any, ...]:
    if o.level == "decision" and not isinstance(subject, LLMResponse):
        raise OracleError(
            f"oracle {o.name!r} judges decisions (LLMResponse), got a {type(subject).__name__}"
        )
    if o.level == "run" and not isinstance(subject, Run):
        raise OracleError(
            f"oracle {o.name!r} judges whole runs (Run), got a {type(subject).__name__}"
        )
    return (subject, request) if o.wants_request else (subject,)


async def acheck(
    o: Oracle, subject: LLMResponse | Run, request: LLMRequest | None = None
) -> Failure | None:
    try:
        result = o.fn(*_args(o, subject, request))
        if inspect.isawaitable(result):
            result = await result
    except OracleError:
        raise
    except Exception as exc:
        raise OracleError(f"oracle {o.name!r} raised {type(exc).__name__}: {exc}") from exc
    return _validate(o, result)


def check(
    o: Oracle, subject: LLMResponse | Run, request: LLMRequest | None = None
) -> Failure | None:
    """Synchronous `acheck` for non-async oracles."""
    try:
        result = o.fn(*_args(o, subject, request))
    except OracleError:
        raise
    except Exception as exc:
        raise OracleError(f"oracle {o.name!r} raised {type(exc).__name__}: {exc}") from exc
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise OracleError(f"oracle {o.name!r} is async; use acheck()")
    return _validate(o, result)
