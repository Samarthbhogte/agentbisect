"""Decision-level replay: rebuild the request of one recorded LLM call and send it again."""

from __future__ import annotations

from pydantic import BaseModel

from agentbisect.providers.base import LLMProvider, LLMRequest, LLMResponse, ToolSpec
from agentbisect.schema import Run, Step


class RequestOverrides(BaseModel):
    """Replace parts of the recorded request: a different model, the current system prompt, or
    the current tool specs."""

    model: str | None = None
    temperature: float | None = None
    system: str | None = None
    tools: list[ToolSpec] | None = None

    def apply(self, req: LLMRequest) -> LLMRequest:
        update = {k: v for k, v in self.model_dump(exclude_none=True).items()}
        if self.tools is not None:
            update["tools"] = list(self.tools)
        return req.model_copy(update=update, deep=True) if update else req


class NotAnLLMStep(ValueError):
    pass


def request_of(step: Step) -> LLMRequest:
    if step.kind != "llm_call":
        raise NotAnLLMStep(f"step {step.id} is a {step.kind}, not an LLM call")
    return LLMRequest.model_validate(step.request)


def build_request(
    step: Step,
    *,
    keep_units: set[str] | frozenset[str] | None = None,
    overrides: RequestOverrides | None = None,
) -> LLMRequest:
    req = request_of(step)
    if keep_units is not None:
        from agentbisect.context.units import recompose

        req = recompose(req, frozenset(keep_units))
    return overrides.apply(req) if overrides else req


async def replay_decision(
    step: Step,
    *,
    keep_units: set[str] | None = None,
    provider: LLMProvider,
    seed: int,
    overrides: RequestOverrides | None = None,
) -> LLMResponse:
    return await provider.complete(
        build_request(step, keep_units=keep_units, overrides=overrides), seed=seed
    )


def find_step(run: Run, ref: str) -> Step:
    """`s0023`, `23`, or `s23`. (`auto` is resolved by the caller, which needs an oracle.)"""
    want = ref if ref.startswith("s") and ref[1:].isdigit() and len(ref) == 5 else None
    if want is None:
        digits = ref[1:] if ref.startswith("s") else ref
        if not digits.isdigit():
            raise KeyError(f"bad step reference {ref!r}: use s0023, 23 or auto")
        want = f"s{int(digits):04d}"
    for step in run.steps:
        if step.id == want:
            return step
    raise KeyError(f"run {run.id} has no step {ref}")
