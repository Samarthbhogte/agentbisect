"""Make a replayed program behave like the recorded one (BLUEPRINT 6.4).

While active: `time.time()` returns a clock that starts at the recorded run's start and advances
1 ms per call; `uuid.uuid4()` and the global `random` generator are seeded from the run id (and
the trial seed). `datetime.now()` / `utcnow()` follow the same clock **in the modules passed as
`modules`** (the agent's entrypoint module), where a `datetime` name bound to the real class is
swapped for a frozen subclass.

The `datetime.datetime` class itself is never replaced process-wide: libraries (for example the
LLM SDKs' pydantic models) build schemas from it while the agent runs, and a substituted class
breaks them.

Limitations: other modules' `datetime.now()`, code that bound `from time import time` before
replay started, other threads, subprocesses, `os.urandom`/`secrets`, and network calls that do not
go through a wrapped LLM client are not controlled.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import random
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import ModuleType
from typing import Any


@contextmanager
def frozen(
    run_id: str, start: _dt.datetime, seed: int = 0, modules: Sequence[ModuleType] = ()
) -> Iterator[None]:
    digest = hashlib.sha256(f"{run_id}|{seed}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    clock = [start.timestamp()]

    def fake_time() -> float:
        clock[0] += 0.001
        return clock[0]

    def fake_uuid4() -> uuid.UUID:
        return uuid.UUID(int=rng.getrandbits(128), version=4)

    real_datetime = _dt.datetime

    class FrozenDatetime(_dt.datetime):
        @classmethod
        def now(cls, tz: _dt.tzinfo | None = None) -> Any:
            value = real_datetime.fromtimestamp(fake_time(), tz=_dt.UTC)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

        @classmethod
        def utcnow(cls) -> Any:
            return real_datetime.fromtimestamp(fake_time(), tz=_dt.UTC).replace(tzinfo=None)

    swapped: list[tuple[ModuleType, str, Any]] = []
    for module in modules:
        for name, value in list(vars(module).items()):
            if value is real_datetime:
                swapped.append((module, name, value))
                setattr(module, name, FrozenDatetime)
    saved_time, saved_uuid4, saved_state = time.time, uuid.uuid4, random.getstate()
    random.seed(int.from_bytes(digest[8:16], "big"))
    time.time = fake_time
    uuid.uuid4 = fake_uuid4
    try:
        yield
    finally:
        time.time = saved_time
        uuid.uuid4 = saved_uuid4
        random.setstate(saved_state)
        for module, name, value in swapped:
            setattr(module, name, value)
