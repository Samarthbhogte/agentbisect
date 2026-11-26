"""Pytest plugin (auto-loaded): registers the `agentbisect` marker used by emitted tests."""

from __future__ import annotations

from typing import Any


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers", "agentbisect: regression scenario emitted by AgentBisect (relay emit-test)"
    )
