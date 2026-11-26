from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agentbisect.sdk.context import reset_state


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", default=False, help="run live tests")


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--live"):
        # Live tests need the network and the `live` marker selected.
        config.option.markexpr = "live"
        config.option.disable_socket = False


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Every test records into its own store and starts with no active runs."""
    home = tmp_path / ".agentbisect"
    monkeypatch.setenv("AGENTBISECT_HOME", str(home))
    monkeypatch.delenv("AGENTBISECT_DISABLED", raising=False)
    monkeypatch.delenv("AGENTBISECT_RUN_NAME", raising=False)
    reset_state()
    yield home
    reset_state()
