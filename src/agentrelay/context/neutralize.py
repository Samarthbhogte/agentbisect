"""A heuristic "neutral rewrite": drop text that tries to instruct the model, keep the facts.

Used for counterfactual substitution when no LLM rewrite is available (e.g. with FakeLLM) or
wanted. It removes HTML/XML comments and sentences with common prompt-injection cues. It is a
heuristic, so reports label its results "heuristic".
"""

from __future__ import annotations

import re

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_CUES = re.compile(
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) instructions|disregard (all |any )?"
    r"(previous|prior|above)|\bAI (assistant|agent|model)s?\b|\byou must now\b|"
    r"\bnew instructions\b|\bsystem prompt\b",
    re.I,
)
_SENTENCE = re.compile(r"[^.!?\n]*[.!?]?")


def neutralize(text: str) -> str:
    """`text` without comments and without sentences that contain injection cues."""
    out = _COMMENT.sub("", text)
    kept_lines = []
    for line in out.splitlines(keepends=True):
        if _CUES.search(line):
            parts = [s for s in _SENTENCE.findall(line) if s.strip() and not _CUES.search(s)]
            line = "".join(parts).rstrip() + ("\n" if line.endswith("\n") else "")
        kept_lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "".join(kept_lines))


def removed_text(original: str, neutral: str) -> str:
    """What the rewrite dropped, for display (best effort)."""
    comments = _COMMENT.findall(original)
    if comments:
        return " ".join(c.strip() for c in comments)
    return " ".join(s.strip() for s in _SENTENCE.findall(original) if s.strip() and _CUES.search(s))
