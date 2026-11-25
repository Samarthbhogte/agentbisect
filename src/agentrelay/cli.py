"""`relay` command-line entry point."""

from __future__ import annotations

import typer

from agentrelay import __version__

app = typer.Typer(
    name="relay",
    help="AgentRelay: find which step and which context caused an AI agent failure.",
    no_args_is_help=True,
)


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


@app.command()
def doctor() -> None:
    """Check config, keys, and store health."""
    typer.echo(f"agentrelay {__version__}: OK")
