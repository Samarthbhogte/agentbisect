"""Oracles for the refund agent."""

from __future__ import annotations

import re

from agentrelay import Failure, oracle
from agentrelay.providers.base import LLMResponse
from agentrelay.schema import Run

STALE_WINDOW = re.compile(r"\b60[- ]day|\b60 days", re.IGNORECASE)


@oracle("quotes-stale-window", level="decision")
def quotes_stale_window(resp: LLMResponse) -> Failure | None:
    match = STALE_WINDOW.search(resp.text())
    return Failure(fingerprint="refund_window:60_days", detail=match.group(0)) if match else None


@oracle("wrong-refund", level="run")
def wrong_refund(run: Run) -> Failure | None:
    answer = str((run.output or {}).get("answer", ""))
    match = STALE_WINDOW.search(answer)
    return Failure(fingerprint="refund_window:60_days", detail=match.group(0)) if match else None
