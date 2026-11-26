"""Regenerate docs/demo.svg and docs/demo.cast from real command output (no API key needed).

uv run python docs/make_demo.py
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 120
RELAY = str(Path(sys.executable).with_name("relay"))
COMMANDS = [
    (
        "python -m examples.file_cleanup_agent.agent --provider fake --seed 2",
        [
            sys.executable,
            "-m",
            "examples.file_cleanup_agent.agent",
            "--provider",
            "fake",
            "--seed",
            "2",
        ],
    ),
    ("relay show last", [RELAY, "show", "last"]),
    (
        "relay explain last --step auto \\\n    --oracle examples/file_cleanup_agent/oracles.py:"
        "deletes_outside_tmp \\\n    --provider fake --fake-rules "
        "examples/file_cleanup_agent/fake_rules.py",
        [
            RELAY,
            "explain",
            "last",
            "--step",
            "auto",
            "--oracle",
            "examples/file_cleanup_agent/oracles.py:deletes_outside_tmp",
            "--provider",
            "fake",
            "--fake-rules",
            "examples/file_cleanup_agent/fake_rules.py",
        ],
    ),
]


def run_all() -> list[tuple[str, str]]:
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        home = Path(tmp) / ".agentrelay"
        env = {
            **os.environ,
            "AGENTRELAY_HOME": str(home),
            "COLUMNS": str(WIDTH),
            "FORCE_COLOR": "1",
            "TERM": "xterm-256color",
        }
        out = []
        for shown, argv in COMMANDS:
            proc = subprocess.run(
                argv, cwd=ROOT, env=env, capture_output=True, text=True, check=False
            )
            text = proc.stdout + proc.stderr
            for prefix in (str(home.resolve()), str(home)):  # /tmp is /private/tmp on macOS
                text = text.replace(prefix, ".agentrelay")
            out.append((shown, text))
        return out


def write_svg(results: list[tuple[str, str]]) -> None:
    console = Console(
        record=True,
        width=WIDTH,
        force_terminal=True,
        color_system="truecolor",
        file=io.StringIO(),
    )
    shown, text = results[-1]
    console.print(Text("$ " + shown, style="bold green"))
    console.print(Text.from_ansi(text.rstrip()))
    console.save_svg(str(ROOT / "docs" / "demo.svg"), title="relay explain")


def write_cast(results: list[tuple[str, str]]) -> None:
    events: list[list[object]] = []
    t = 0.5
    for shown, text in results:
        for ch in "$ " + shown:
            events.append([round(t, 3), "o", ch.replace("\n", "\r\n")])
            t += 0.03
        events.append([round(t, 3), "o", "\r\n"])
        t += 0.6
        for line in text.splitlines():
            events.append([round(t, 3), "o", line + "\r\n"])
            t += 0.04
        t += 1.5
    header = {
        "version": 2,
        "width": WIDTH,
        "height": 40,
        "title": "AgentRelay: why did the agent delete the project?",
        "env": {"TERM": "xterm-256color", "SHELL": "/bin/sh"},
    }
    lines = [json.dumps(header)] + [json.dumps(e) for e in events]
    (ROOT / "docs" / "demo.cast").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    results = run_all()
    write_svg(results)
    write_cast(results)
    print("wrote docs/demo.svg and docs/demo.cast")
