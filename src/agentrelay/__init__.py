"""AgentRelay: git bisect + delta debugging for AI agents.

import agentrelay as relay

client = relay.wrap(anthropic.Anthropic())

@relay.tool
def read_file(path: str) -> str: ...

@relay.entrypoint
def main(task: str) -> str:
    with relay.run("cleanup", input={"task": task}):
        ...
"""

from __future__ import annotations

__version__ = "0.1.0"

from agentrelay.schema import Failure
from agentrelay.sdk.context import entrypoint, run
from agentrelay.sdk.tools import tool
from agentrelay.sdk.wrap import wrap

__all__ = ["Failure", "__version__", "entrypoint", "run", "tool", "wrap"]
