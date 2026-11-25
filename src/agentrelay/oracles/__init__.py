"""Oracles decide whether a response or a run failed, and which failure it was."""

from agentrelay.oracles.api import Oracle, OracleError, acheck, check, load_oracle, oracle

__all__ = ["Oracle", "OracleError", "acheck", "check", "load_oracle", "oracle"]
