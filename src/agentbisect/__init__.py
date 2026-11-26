"""AgentBisect: git bisect + delta debugging for AI agents.

import agentbisect as relay

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

from agentbisect.oracles.api import oracle
from agentbisect.schema import Failure
from agentbisect.sdk.context import entrypoint, run
from agentbisect.sdk.tools import tool
from agentbisect.sdk.wrap import wrap

__all__ = ["Failure", "__version__", "entrypoint", "oracle", "run", "tool", "wrap"]
