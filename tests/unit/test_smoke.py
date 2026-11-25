from __future__ import annotations

import socket

import pytest
from pytest_socket import SocketBlockedError
from typer.testing import CliRunner

import agentrelay
from agentrelay.cli import app


def test_version_string() -> None:
    assert agentrelay.__version__ == "0.1.0"


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "agentrelay 0.1.0" in result.stdout


@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_network_is_blocked() -> None:
    with pytest.raises(SocketBlockedError):
        socket.create_connection(("example.com", 80), timeout=1)
