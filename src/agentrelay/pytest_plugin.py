"""Pytest plugin (auto-loaded): registers the `agentrelay` marker used by emitted tests."""

from __future__ import annotations

from typing import Any


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers", "agentrelay: regression scenario emitted by AgentRelay (relay emit-test)"
    )
