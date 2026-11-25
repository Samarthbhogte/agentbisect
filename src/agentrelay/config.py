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
    extra: dict[str, Any] = Field(default_factory=dict)


def config_path(home: Path) -> Path:
    return home.parent / CONFIG_NAME


def load_config(home: Path) -> Config:
    path = config_path(home)
    if not path.is_file():
        return Config()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {k: data.pop(k) for k in ("redact_pii", "redact_patterns") if k in data}
    return Config(**known, extra=data)


DEFAULT_TOML = """\
# AgentRelay configuration. Every key is optional.

# Also redact email addresses and phone numbers before anything is written.
redact_pii = false

# Extra secret patterns to redact, as name = "regex".
[redact_patterns]
# internal_token = "itk_[A-Za-z0-9]{20}"
"""
