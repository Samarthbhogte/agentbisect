"""`agentrelay.toml` loading.

The file lives next to `.agentrelay/` (the project root). Every key is optional::

    redact_pii = false               # also redact emails and phone numbers

    [redact_patterns]                # extra secrets to redact: name = "regex"
    internal_token = "itk_[A-Za-z0-9]{20}"
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

CONFIG_NAME = "agentrelay.toml"


class Config(BaseModel):
    redact_pii: bool = False
    redact_patterns: dict[str, str] = Field(default_factory=dict)
    # model id -> [input $/Mtok, output $/Mtok]; overrides providers/pricing.py
    prices: dict[str, tuple[float, float]] = Field(default_factory=dict)
    # registered oracle name -> "path/to/oracles.py:attr"
    oracles: dict[str, str] = Field(default_factory=dict)
    # The agent's *current* system prompt and tool specs, swapped into regression scenarios so a
    # fix to the prompt is what the test checks: "module:ATTR" or "file.py:ATTR".
    system_prompt: str | None = None
    tools: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def config_path(home: Path) -> Path:
    return home.parent / CONFIG_NAME


def load_config(home: Path) -> Config:
    path = config_path(home)
    if not path.is_file():
        return Config()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    names = ("redact_pii", "redact_patterns", "prices", "oracles", "system_prompt", "tools")
    known = {k: data.pop(k) for k in names if k in data}
    return Config(**known, extra=data)


DEFAULT_TOML = """\
# AgentRelay configuration. Every key is optional.

# The agent's current system prompt and tool specs ("module:ATTR" or "file.py:ATTR"). Regression
# tests from `relay emit-test` swap these into the recorded request, so they test your fix.
# system_prompt = "myagent.prompts:SYSTEM"
# tools = "myagent.prompts:TOOLS"

# Also redact email addresses and phone numbers before anything is written.
redact_pii = false

# Extra secret patterns to redact, as name = "regex".
[redact_patterns]
# internal_token = "itk_[A-Za-z0-9]{20}"

# Model prices in USD per million tokens: [input, output]. Used for budgets and cost reports.
# Models without a price use a deliberately high default so budgets are never under-counted.
[prices]
# "your-model-id" = [3.0, 15.0]

# Oracles you can refer to by name: --oracle no-outside-deletes
[oracles]
# no-outside-deletes = "examples/file_cleanup_agent/oracles.py:deletes_outside_tmp"
"""
