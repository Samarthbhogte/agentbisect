from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", default=False, help="run live tests")


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--live"):
        # Live tests need the network and the `live` marker selected.
        config.option.markexpr = "live"
        config.option.disable_socket = False
