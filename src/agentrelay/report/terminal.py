"""Terminal rendering of `relay explain` (BLUEPRINT 6.9 example layout)."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from agentrelay.algo.explain import ExplainResult, UnitAttribution
from agentrelay.schema import RateEstimate


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def rate(est: RateEstimate) -> str:
    return f"{pct(est.p_hat)}  [{pct(est.ci_low)}–{pct(est.ci_high)}]   (n={est.trials})"


def short(text: str, width: int = 64) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= width else one_line[: width - 1] + "…"


def key_text(a: UnitAttribution) -> str:
    """The part of the unit worth showing: what a neutral rewrite removed, else its start."""
    cf = a.counterfactual
    if cf is not None and cf.removed:
        return cf.removed
    return a.text


def render_explain(r: ExplainResult, console: Console, *, next_hint: str | None = None) -> None:
    s = r.shrink
    console.print(
        f"[bold]AgentRelay explain[/bold] — run {(r.run_id or '')[:10]}… step "
        f"{r.step_id} ({escape(r.model or '')})"
    )
    noisy = "  [yellow](LLM judge: noisier)[/yellow]" if r.oracle_noisy else ""
    console.print(
        f"Oracle: {escape(r.oracle or '')}   fingerprint: {escape(r.fingerprint)}" + noisy + "\n"
    )
    console.print(f"Baseline failure rate      {rate(r.baseline)}")
    if r.status in ("too_rare", "not_failing"):
        console.print(f"\n[yellow]{escape(r.message)}[/yellow]")
        return
    reduction = 1 - s.minimal_tokens / s.original_tokens if s.original_tokens else 0.0
    console.print(
        f"Minimal repro              {len(s.minimal_ids)} of {len(s.units)} units, "
        f"{s.minimal_tokens:,} of {s.original_tokens:,} tokens "
        f"(−{100 * reduction:.1f}%)"
    )
    if r.minimal_estimate is not None:
        console.print(f"Minimal failure rate       {rate(r.minimal_estimate)}")
    console.print()
    table = Table(box=None, header_style="bold", pad_edge=False)
    table.add_column("Rank", justify="right")
    table.add_column("Unit")
    table.add_column("Remove from original run")
    table.add_column("Attribution", justify="right")
    table.add_column("")
    p0 = pct(r.baseline.p_hat)
    for a in r.attributions:
        w = a.without_full
        change = f"{p0} → {pct(w.p_hat)} [{pct(w.ci_low)}–{pct(w.ci_high)}]"
        points = f"{100 * a.attribution:+.0f} pts"
        flag = "[bold red]CULPRIT[/bold red]" if a.culprit else ""
        table.add_row(str(a.rank), a.unit_id, change, points, flag)
        table.add_row("", f'[dim]"{escape(short(key_text(a)))}"[/dim]', "", "", "")
        cf = a.counterfactual
        if cf is not None:
            effect = f"{p0} → {pct(cf.estimate.p_hat)}" if cf.estimate else "not run"
            table.add_row(
                "",
                f"[dim]neutral rewrite ({cf.source}): {effect} — {escape(cf.note)}[/dim]",
                "",
                "",
                "",
            )
    start = len(r.attributions) + 1
    for i, label in enumerate(r.not_removable, start=start):
        table.add_row(str(i), escape(label), "not removable", "", "")
    console.print(table)
    for inter in r.interactions:
        console.print(f"\n[bold]Interaction ({inter.kind}):[/bold] {escape(inter.message)}")
    if r.status != "ok":
        console.print(f"\n[yellow]{escape(r.message)}[/yellow]")
    elapsed = f", {r.elapsed_s:.1f}s" if r.elapsed_s is not None else ""
    console.print(
        f"\nSpent: {r.trials} trials ({r.cached_trials} cached), {r.provider_calls} "
        f"provider calls, ${r.cost_usd:.2f}{elapsed}"
    )
    if next_hint:
        console.print(f"Next:  {escape(next_hint)}")
