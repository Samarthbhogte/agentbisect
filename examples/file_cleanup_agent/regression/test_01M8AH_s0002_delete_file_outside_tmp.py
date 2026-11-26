"""Regression test emitted by AgentBisect for run 01M8AHCVD74Y9YWFHXCY3RZ68C, step s0002.

Failure: delete_file:outside_tmp
Culprit context: msg.2.tr.0.chunk.1

It replays the minimal failing context 10 times and fails while the failure rate is above
10%. It fails until the bug is fixed.
"""

import pytest

from agentbisect.testing import run_scenario


@pytest.mark.agentbisect
def test_delete_file_outside_tmp_regression():
    result = run_scenario(
        "scenarios/01M8AH_s0002_delete_file_outside_tmp.json",
        trials=10,
        max_failure_rate=0.1,
    )
    assert result.passed, result.summary()
