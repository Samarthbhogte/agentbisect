from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentbisect.stats.estimate import default_threshold, make_estimate, sequential_decision, wilson


@pytest.mark.parametrize(
    ("failures", "trials", "low", "high"),
    [
        (8, 10, 0.490, 0.943),  # BLUEPRINT M3 reference value
        (5, 10, 0.237, 0.763),
        (0, 10, 0.000, 0.278),
        (10, 10, 0.722, 1.000),
        (1, 20, 0.009, 0.236),
        (78, 100, 0.689, 0.850),
    ],
)
def test_wilson_matches_reference_values(
    failures: int, trials: int, low: float, high: float
) -> None:
    lo, hi = wilson(failures, trials)
    assert (round(lo, 3), round(hi, 3)) == (low, high)


def test_wilson_without_trials_is_uninformative() -> None:
    assert wilson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson(3, 2)


@given(st.integers(1, 500).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n))))
def test_wilson_properties(kn: tuple[int, int]) -> None:
    k, n = kn
    lo, hi = wilson(k, n)
    assert 0.0 <= lo <= k / n <= hi <= 1.0
    mirror_lo, mirror_hi = wilson(n - k, n)
    assert lo == pytest.approx(1 - mirror_hi, abs=1e-12)
    assert hi == pytest.approx(1 - mirror_lo, abs=1e-12)


def first_decision(outcomes: list[bool], **kw: float) -> tuple[str | None, int]:
    failures = 0
    for i, failed in enumerate(outcomes, start=1):
        failures += failed
        d = sequential_decision(
            failures,
            i,
            threshold=kw.get("threshold", 0.5),
            min_trials=int(kw.get("min_trials", 3)),
            max_trials=int(kw.get("max_trials", 8)),
        )
        if d is not None:
            return d, i
    return None, len(outcomes)


def test_sequential_rule() -> None:
    assert first_decision([True] * 8) == ("fails", 4)  # 4/4 -> CI low 0.51 >= 0.5
    assert first_decision([False] * 8) == ("passes", 4)  # 0/4 -> CI high 0.49 < 0.5
    assert first_decision([True, False] * 4) == ("undecided", 8)
    # One early pass: 5/6 has CI low 0.44, so p=0.95 can't be decided by trial 6.
    assert first_decision([False] + [True] * 7) == ("fails", 8)
    assert first_decision([True] * 8, min_trials=6) == ("fails", 6)  # honors min_trials


def test_estimate_and_still_fails() -> None:
    est = make_estimate(5, 8, decided="undecided", threshold=0.5, slippage={"x": 1})
    assert est.p_hat == 0.625 and est.still_fails()
    assert not make_estimate(3, 8, decided="undecided", threshold=0.5).still_fails()
    assert make_estimate(4, 4, decided="fails", threshold=0.5).still_fails()
    assert not make_estimate(0, 4, decided="passes", threshold=0.5).still_fails()
    assert not make_estimate(0, 0, decided="undecided", threshold=0.5, partial=True).still_fails()


def test_default_threshold() -> None:
    assert default_threshold(0.78) == pytest.approx(0.39)
    assert default_threshold(0.3) == 0.2
