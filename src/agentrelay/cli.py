"""`relay` command-line entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from agentrelay import __version__
from agentrelay.config import CONFIG_NAME, DEFAULT_TOML
from agentrelay.schema import Run, Step
from agentrelay.store.index import RunFilter
from agentrelay.store.runs import AmbiguousRunRef, RunNotFound, RunStore, resolve_home

app = typer.Typer(
    name="relay",
    help="AgentRelay: find which step and which context caused an AI agent failure.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err = Console(stderr=True)

JsonFlag = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"agentrelay {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """AgentRelay CLI."""


def _store() -> RunStore:
    return RunStore()


def _load(store: RunStore, ref: str) -> Run:
    try:
        return store.load(ref)
    except (RunNotFound, AmbiguousRunRef) as exc:
        err.print(f"[red]error:[/red] {escape(str(exc.args[0]))}")
        raise typer.Exit(1) from exc


def _print_json(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# ---- init ---------------------------------------------------------------------------------
@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Project directory.")] = Path("."),
) -> None:
    """Create agentrelay.toml and add .agentrelay/ to .gitignore."""
    root = path.resolve()
    config = root / CONFIG_NAME
    if config.exists():
        console.print(f"{CONFIG_NAME} already exists, left unchanged")
    else:
        config.write_text(DEFAULT_TOML, encoding="utf-8")
        console.print(f"created {CONFIG_NAME}")
    (root / ".agentrelay").mkdir(exist_ok=True)
    gitignore = root / ".gitignore"
    lines = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    if not any(line.strip().rstrip("/") in (".agentrelay", "/.agentrelay") for line in lines):
        prefix = "" if not lines or lines[-1] == "" else "\n"
        with gitignore.open("a", encoding="utf-8") as fh:
            fh.write(f"{prefix}.agentrelay/\n")
        console.print("added .agentrelay/ to .gitignore")


# ---- record -------------------------------------------------------------------------------
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def record(
    ctx: typer.Context,
    name: Annotated[
        str | None, typer.Option(help="Name for runs without an explicit name.")
    ] = None,
) -> None:
    """Run a command with capture enabled: relay record -- python my_agent.py"""
    cmd = list(ctx.args)
    if not cmd:
        err.print("usage: relay record -- <command...>")
        raise typer.Exit(2)
    home = resolve_home()
    store = RunStore(home)
    before = set(store.run_ids())
    env = dict(os.environ)
    env["AGENTRELAY_HOME"] = str(home)
    env["AGENTRELAY_RUN_NAME"] = name or _guess_name(cmd)
    env.pop("AGENTRELAY_DISABLED", None)
    code = subprocess.call(cmd, env=env)
    new = [i for i in store.run_ids() if i not in before]
    for run_id in new:
        err.print(f"recorded run {run_id}")
    if not new:
        err.print("no runs were recorded (is the client wrapped with relay.wrap?)")
    raise typer.Exit(code)


_GENERIC_NAMES = {"agent", "main", "__main__", "app", "run"}


def _guess_name(cmd: list[str]) -> str:
    """`python -m pkg.file_cleanup_agent.agent` -> `file_cleanup_agent`; `python x.py` -> `x`.

    A generic module or file name (agent, main, ...) falls back to its parent's name.
    """
    if len(cmd) >= 3 and cmd[1] == "-m":
        parts = cmd[2].split(".")
    else:
        script = Path(next((c for c in cmd[1:] if c.endswith(".py")), cmd[0]))
        parts = [script.parent.name, script.stem] if script.parent.name else [script.stem]
    while len(parts) > 1 and parts[-1] in _GENERIC_NAMES:
        parts.pop()
    return parts[-1] or "run"


# ---- ls -----------------------------------------------------------------------------------
@app.command("ls")
def list_runs(
    status: Annotated[str | None, typer.Option(help="ok, error or unknown.")] = None,
    name: Annotated[str | None, typer.Option(help="Only runs with this name.")] = None,
    limit: Annotated[int, typer.Option(help="Show at most this many runs.")] = 20,
    as_json: JsonFlag = False,
) -> None:
    """List recorded runs, newest first."""
    runs = _store().list(RunFilter(status=status, name=name, limit=limit))
    if as_json:
        _print_json([r.model_dump(mode="json") for r in runs])
        return
    if not runs:
        console.print("no runs recorded yet")
        return
    table = Table(box=None, header_style="bold")
    for col in ("id", "name", "status", "started", "steps", "cost", "labels"):
        table.add_column(col, justify="right" if col in ("steps", "cost") else "left")
    for r in runs:
        table.add_row(
            r.id,
            escape(r.name),
            _status(r.status),
            r.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            str(r.n_steps),
            f"${r.cost:.4f}" if r.cost is not None else "-",
            escape(", ".join(f"{k}={v}" for k, v in r.labels.items())),
        )
    console.print(table)


def _status(status: str) -> str:
    color = {"ok": "green", "error": "red"}.get(status, "yellow")
    return f"[{color}]{status}[/{color}]"


# ---- show ---------------------------------------------------------------------------------
@app.command()
def show(
    run_ref: Annotated[
        str, typer.Argument(metavar="RUN", help="Run id, unique prefix, or 'last'.")
    ],
    step: Annotated[str | None, typer.Option(help="Show one step in full: s0012 or 12.")] = None,
    as_json: JsonFlag = False,
) -> None:
    """Pretty-print a run or one of its steps."""
    run = _load(_store(), run_ref)
    if step is not None:
        found = _find_step(run, step)
        if as_json:
            _print_json(found.model_dump(mode="json"))
        else:
            _show_step(found)
        return
    if as_json:
        _print_json(run.model_dump(mode="json", by_alias=True))
        return
    _show_run(run)


def _find_step(run: Run, ref: str) -> Step:
    want = ref if ref.startswith("s") else f"s{int(ref):04d}"
    for s in run.steps:
        if s.id == want:
            return s
    err.print(f"[red]error:[/red] run {run.id} has no step {ref}")
    raise typer.Exit(1)


def _preview(step: Step, width: int = 70) -> str:
    if step.error:
        text = f"error {step.error.type}: {step.error.message}"
    elif step.kind == "llm_call" and step.response:
        parts = []
        for b in step.response.get("content") or []:
            if b.get("type") == "text":
                parts.append(b.get("text") or "")
            elif b.get("type") == "tool_use":
                parts.append(f"→ {b.get('tool_name')}({json.dumps(b.get('tool_input'))})")
        text = " ".join(parts)
    elif step.kind == "tool_call":
        args = ", ".join(f"{k}={json.dumps(v)}" for k, v in step.request.items())
        text = f"({args}) → {json.dumps((step.response or {}).get('result'))}"
    else:
        text = json.dumps(step.response)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _show_run(run: Run) -> None:
    duration = (run.ended_at - run.started_at).total_seconds() if run.ended_at else None
    console.print(f"[bold]Run {run.id}[/bold]  {escape(run.name)}  {_status(run.status)}")
    console.print(
        f"started {run.started_at:%Y-%m-%d %H:%M:%S}"
        + (f"   duration {duration:.2f}s" if duration is not None else "")
    )
    if run.entrypoint:
        console.print(f"entrypoint {escape(run.entrypoint)}")
    console.print(f"input  {escape(json.dumps(run.input, ensure_ascii=False))}")
    if run.output is not None:
        console.print(f"output {escape(json.dumps(run.output, ensure_ascii=False))}")
    if run.labels:
        console.print("labels " + escape(", ".join(f"{k}={v}" for k, v in run.labels.items())))
    table = Table(box=None, header_style="bold", pad_edge=False)
    for col in ("step", "kind", "name", "deps", "tokens", "summary"):
        table.add_column(col, overflow="fold" if col == "summary" else "ellipsis")
    for s in run.steps:
        tokens = f"{s.tokens_in}/{s.tokens_out}" if s.tokens_in is not None else ""
        table.add_row(
            s.id, s.kind, escape(s.name), ",".join(s.depends_on), tokens, escape(_preview(s))
        )
    console.print(table)
    tin = sum(s.tokens_in or 0 for s in run.steps)
    tout = sum(s.tokens_out or 0 for s in run.steps)
    console.print(f"{len(run.steps)} steps, {tin} input / {tout} output tokens")


def _show_step(step: Step) -> None:
    console.print(
        f"[bold]{step.id}[/bold] {step.kind} {escape(step.name)}"
        + (f" ({step.provider})" if step.provider else "")
    )
    console.print(f"depends_on: {', '.join(step.depends_on) or '-'}   tags: {step.tags or '-'}")
    if step.error:
        console.print(f"[red]error {escape(step.error.type)}: {escape(step.error.message)}[/red]")
    console.print("[bold]request[/bold]")
    console.print_json(json.dumps(step.request, ensure_ascii=False))
    console.print("[bold]response[/bold]")
    console.print_json(json.dumps(step.response, ensure_ascii=False))


# ---- doctor -------------------------------------------------------------------------------
@app.command()
def doctor() -> None:
    """Check config, keys, and store health."""
    home = resolve_home()
    console.print(f"agentrelay {__version__}")
    console.print(f"store      {home} ({'exists' if home.is_dir() else 'not created yet'})")
    config = home.parent / CONFIG_NAME
    console.print(f"config     {config} ({'found' if config.is_file() else 'not found'})")
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        console.print(f"{key:<18} {'set' if os.environ.get(key) else 'not set'}")
    if home.is_dir():
        console.print(f"runs       {len(RunStore(home).run_ids())}")


def run() -> None:  # pragma: no cover
    app(prog_name="relay")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
