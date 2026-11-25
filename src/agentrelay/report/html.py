"""Self-contained HTML report for `relay explain --html` (inline CSS and SVG, no requests)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup, escape

from agentrelay import __version__
from agentrelay.algo.explain import ExplainResult
from agentrelay.context.units import recompose
from agentrelay.providers.base import LLMRequest
from agentrelay.schema import RateEstimate

_env = Environment(
    loader=PackageLoader("agentrelay.report", "templates"),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass
class Bar:
    label: str
    note: str
    est: RateEstimate
    kind: str  # baseline | minimal | ablation | culprit | counterfactual
    y: int = 0

    @property
    def x(self) -> float:
        return round(100 * self.est.p_hat, 2)

    @property
    def lo(self) -> float:
        return round(100 * self.est.ci_low, 2)

    @property
    def hi(self) -> float:
        return round(100 * self.est.ci_high, 2)


def _bars(r: ExplainResult) -> list[Bar]:
    bars = [Bar("Baseline", "original request", r.baseline, "baseline")]
    if r.minimal_estimate is not None:
        bars.append(Bar("Minimal", "minimal context", r.minimal_estimate, "minimal"))
    for a in r.attributions:
        bars.append(
            Bar(
                f"− {a.unit_id}",
                "removed from original",
                a.without_full,
                "culprit" if a.culprit else "ablation",
            )
        )
        if a.counterfactual and a.counterfactual.estimate:
            bars.append(
                Bar(
                    f"≈ {a.unit_id}",
                    f"neutral rewrite ({a.counterfactual.source})",
                    a.counterfactual.estimate,
                    "counterfactual",
                )
            )
    for i, b in enumerate(bars):
        b.y = 28 + i * 34
    return bars


def highlight(text: str, needles: list[str]) -> Markup:
    """Escape `text` and wrap every occurrence of each needle in <mark>."""
    spans: list[tuple[int, int]] = []
    for needle in sorted({n for n in needles if n.strip()}, key=len, reverse=True):
        start = text.find(needle)
        while start != -1:
            end = start + len(needle)
            if not any(s < end and start < e for s, e in spans):
                spans.append((start, end))
            start = text.find(needle, end)
    out, pos = [], 0
    for s, e in sorted(spans):
        out.append(str(escape(text[pos:s])))
        out.append(f"<mark>{escape(text[s:e])}</mark>")
        pos = e
    out.append(str(escape(text[pos:])))
    return Markup("".join(out))


def _transcript(r: ExplainResult, req: LLMRequest) -> dict[str, Any]:
    minimal = recompose(req, r.shrink.minimal_ids, r.shrink.removal)
    culprit_texts = [a.text for a in r.attributions if a.culprit]
    culprit_texts += [
        a.counterfactual.removed
        for a in r.attributions
        if a.culprit and a.counterfactual and a.counterfactual.removed
    ]
    tools = [
        {
            "name": t.name,
            "description": highlight(t.description, culprit_texts)
            if t.description
            else Markup("<em>(description removed)</em>"),
        }
        for t in minimal.tools
    ]
    messages: list[dict[str, Any]] = []
    for m in minimal.messages:
        blocks: list[dict[str, Any]] = []
        for b in m.content:
            if b.type == "text":
                blocks.append({"kind": "text", "body": highlight(b.text or "", culprit_texts)})
            elif b.type == "tool_use":
                blocks.append(
                    {
                        "kind": "tool_use",
                        "name": b.tool_name,
                        "body": highlight(str(b.tool_input), culprit_texts),
                    }
                )
            elif b.type == "tool_result":
                blocks.append(
                    {"kind": "tool_result", "body": highlight(b.text or "", culprit_texts)}
                )
            else:
                blocks.append({"kind": b.type, "body": Markup("<em>(passed through)</em>")})
        messages.append({"role": m.role, "blocks": blocks})
    system = highlight(minimal.system, culprit_texts) if minimal.system else None
    return {"system": system, "tools": tools, "messages": messages}


def render_report(r: ExplainResult, req: LLMRequest) -> str:
    bars = _bars(r)
    return _env.get_template("report.html.j2").render(
        r=r,
        s=r.shrink,
        bars=bars,
        chart_height=bars[-1].y + 40 if bars else 60,
        transcript=_transcript(r, req),
        version=__version__,
        bisect=None,
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        pct=lambda x: f"{100 * x:.0f}%",
    )
