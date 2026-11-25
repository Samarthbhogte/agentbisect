"""Secret redaction. Runs on every value before it is written to disk.

Redacted values become ``[REDACTED:<type>:<short hash>]``. The hash is of the secret itself, so two
occurrences of the same secret stay equal after redaction.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Order matters: more specific patterns first (an Anthropic key also looks like an OpenAI key).
_NOT_WORD = r"(?<![A-Za-z0-9_])"
SECRET_PATTERNS: list[tuple[str, str]] = [
    ("anthropic_key", _NOT_WORD + r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    ("openai_key", _NOT_WORD + r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{16,}"),
    ("aws_access_key", _NOT_WORD + r"(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])"),
    ("github_token", _NOT_WORD + r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    ("slack_token", _NOT_WORD + r"xox[abposr]-[A-Za-z0-9\-]{10,}"),
    ("bearer_token", r"(?i)(?<=bearer )[A-Za-z0-9._~+/=\-]{16,}"),
]
PII_PATTERNS: list[tuple[str, str]] = [
    ("email", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    ("phone", r"(?<![\w-])(?:\+\d{1,3}[\s.\-]?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?![\w-])"),
]


def _marker(kind: str, value: str) -> str:
    return f"[REDACTED:{kind}:{hashlib.sha256(value.encode()).hexdigest()[:8]}]"


def _replacer(kind: str) -> Callable[[re.Match[str]], str]:
    return lambda m: _marker(kind, m.group(0))


@dataclass
class Redactor:
    pii: bool = False
    extra_patterns: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        specs = list(SECRET_PATTERNS)
        if self.pii:
            specs += PII_PATTERNS
        specs += list(self.extra_patterns.items())
        self._compiled = [(name, re.compile(rx)) for name, rx in specs]

    def redact_text(self, text: str) -> str:
        for name, rx in self._compiled:
            text = rx.sub(_replacer(name), text)
        return text

    def redact(self, obj: Any) -> Any:
        """Redact every string inside a JSON-like structure."""
        if isinstance(obj, str):
            return self.redact_text(obj)
        if isinstance(obj, dict):
            return {k: self.redact(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.redact(v) for v in obj]
        return obj

    def find(self, text: str) -> list[str]:
        """Names of patterns that match `text` (used to warn about emitted scenarios)."""
        return [name for name, rx in self._compiled if rx.search(text)]
