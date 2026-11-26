"""Built-in oracles: tool_called, output_matches, output_not_matches, loops, judge."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from agentbisect.oracles.api import Level, Oracle
from agentbisect.providers.base import LLMProvider, LLMRequest, LLMResponse, Message, text_block
from agentbisect.schema import Failure, Run, canonical_json


def tool_called(
    name: str,
    where: Callable[[dict[str, Any]], bool] | None = None,
    *,
    fingerprint: str | None = None,
) -> Oracle:
    """Fails when the response calls tool `name` (and `where(args)` holds, if given)."""
    fp = fingerprint or f"tool_called:{name}"

    def fn(resp: LLMResponse) -> Failure | None:
        for call in resp.tool_calls():
            args = call.tool_input or {}
            if call.tool_name == name and (where is None or where(args)):
                return Failure(fingerprint=fp, detail=json.dumps(args, sort_keys=True))
        return None

    return Oracle(name=f"tool_called({name})", level="decision", fn=fn)


def _output_text(subject: LLMResponse | Run) -> str:
    if isinstance(subject, LLMResponse):
        return subject.text()
    out = subject.output or {}
    if set(out) <= {"answer", "return"} and out:
        value = next(iter(out.values()))
        return value if isinstance(value, str) else json.dumps(value)
    return json.dumps(out, sort_keys=True)


def output_matches(
    regex: str,
    *,
    level: Level = "decision",
    fingerprint: str | None = None,
    flags: int = re.IGNORECASE,
) -> Oracle:
    """Fails when the output text matches `regex`."""
    rx = re.compile(regex, flags)
    fp = fingerprint or f"output_matches:{regex}"

    def fn(subject: Any) -> Failure | None:
        m = rx.search(_output_text(subject))
        return Failure(fingerprint=fp, detail=m.group(0)) if m else None

    return Oracle(name=f"output_matches({regex})", level=level, fn=fn)


def output_not_matches(
    regex: str,
    *,
    level: Level = "decision",
    fingerprint: str | None = None,
    flags: int = re.IGNORECASE,
) -> Oracle:
    """Fails when the output text does not match `regex`."""
    rx = re.compile(regex, flags)
    fp = fingerprint or f"output_not_matches:{regex}"

    def fn(subject: Any) -> Failure | None:
        text = _output_text(subject)
        return None if rx.search(text) else Failure(fingerprint=fp, detail=text[:200])

    return Oracle(name=f"output_not_matches({regex})", level=level, fn=fn)


def loops(max_repeats: int = 3) -> Oracle:
    """Run oracle: fails when one tool is called with the same arguments more than
    `max_repeats` times."""

    def fn(run: Run) -> Failure | None:
        counts = Counter(
            (s.name, canonical_json(s.request)) for s in run.steps if s.kind == "tool_call"
        )
        for (name, _args), n in counts.most_common(1):
            if n > max_repeats:
                return Failure(fingerprint=f"loop:{name}", detail=f"{n} identical calls")
        return None

    return Oracle(name=f"loops({max_repeats})", level="run", fn=fn)


JUDGE_SYSTEM = """You are a strict reviewer of an AI assistant's response.
Decide whether the response violates the rubric. Reply with only a JSON object:
{"fail": true or false, "category": "<short_snake_case_label for the kind of violation>",
 "reason": "<one sentence>"}"""


def judge(
    rubric: str, model: str, *, provider: LLMProvider | None = None, max_tokens: int = 300
) -> Oracle:
    """LLM-as-judge. Fingerprint is `judge:<category>`. Marked `noisy`: reports flag it,
    because the judge is itself a nondeterministic model."""

    async def fn(resp: LLMResponse, req: LLMRequest | None = None) -> Failure | None:
        nonlocal provider
        if provider is None:
            from agentbisect.providers.anthropic import AnthropicProvider

            provider = AnthropicProvider()
        shown = {
            "text": resp.text(),
            "tool_calls": [{"name": c.tool_name, "input": c.tool_input} for c in resp.tool_calls()],
        }
        prompt = f"Rubric:\n{rubric}\n\nResponse to review:\n{json.dumps(shown, indent=2)}"
        judge_req = LLMRequest(
            model=model,
            system=JUDGE_SYSTEM,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[Message(role="user", content=[text_block(prompt)])],
        )
        verdict = _parse_verdict((await provider.complete(judge_req, seed=0)).text())
        if not verdict.get("fail"):
            return None
        category = re.sub(r"[^a-z0-9_]+", "_", str(verdict.get("category") or "violation").lower())
        return Failure(
            fingerprint=f"judge:{category.strip('_') or 'violation'}",
            detail=str(verdict.get("reason") or "")[:300],
        )

    return Oracle(name=f"judge({model})", level="decision", fn=fn, noisy=True)


def _parse_verdict(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"judge did not return JSON: {text[:200]!r}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict) or "fail" not in data:
        raise ValueError(f"judge JSON lacks 'fail': {text[:200]!r}")
    return data
