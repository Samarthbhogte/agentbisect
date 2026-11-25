"""Trace schema: runs, steps, failures, context units, and rate estimates.

Every stored file carries ``"schema": "agentrelay/v1"``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "agentrelay/v1"

StepKind = Literal["llm_call", "tool_call", "event"]
RunStatus = Literal["ok", "error", "unknown"]


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8 kept as-is."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_hash(request: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of a normalized request (or tool args)."""
    return sha256_hex(canonical_json(request))


def step_id(index: int) -> str:
    """Zero-padded step id, e.g. ``s0007``."""
    return f"s{index:04d}"


class ErrorInfo(BaseModel):
    type: str
    message: str


class Step(BaseModel):
    id: str
    run_id: str
    index: int
    kind: StepKind
    name: str
    provider: str | None = None
    request: dict[str, Any]
    response: dict[str, Any] | None
    raw_request: dict[str, Any] | None
    raw_response: dict[str, Any] | None
    error: ErrorInfo | None = None
    depends_on: list[str] = Field(default_factory=list)
    started_at: datetime
    ended_at: datetime
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    request_hash: str
    tags: dict[str, str] = Field(default_factory=dict)


class Run(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: str = Field(default=SCHEMA_VERSION, alias="schema")
    id: str
    name: str
    entrypoint: str | None
    input: dict[str, Any]
    output: dict[str, Any] | None
    status: RunStatus
    steps: list[Step]
    started_at: datetime
    ended_at: datetime | None
    env: dict[str, str]
    labels: dict[str, str] = Field(default_factory=dict)


class Failure(BaseModel):
    """What an oracle returns when a response or run fails."""

    fingerprint: str
    detail: str | None = None


class ContextUnit(BaseModel):
    """An atomic, removable piece of an LLM call's input (see BLUEPRINT 6.7)."""

    id: str
    kind: Literal["system_section", "message", "tool_result", "chunk", "tool_spec"]
    path: str
    text: str
    tokens: int
    group: str | None = None
    parent: str | None = None
    removable: bool = True


class RateEstimate(BaseModel):
    """Result of repeated trials (see BLUEPRINT 6.6)."""

    failures: int
    trials: int
    p_hat: float
    ci_low: float
    ci_high: float
    slippage: dict[str, int] = Field(default_factory=dict)
    decided: Literal["fails", "passes", "undecided"]
