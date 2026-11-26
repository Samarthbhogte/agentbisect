from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agentbisect.oracles.api import acheck, oracle
from agentbisect.providers.base import LLMRequest, LLMResponse, Message, ProviderError, text_block
from agentbisect.providers.fake import FakeLLM, Rule, text, tool_use
from agentbisect.schema import Failure
from agentbisect.stats.runner import MAX_RETRY_WAIT, Budget, TrialRunner
from agentbisect.store.cache import TrialCache

DELETE = tool_use("delete_file", {"path": "/home/user/project"})
LIST = tool_use("list_dir", {"path": "/"})


def req(task: str = "please delete the project") -> LLMRequest:
    return LLMRequest(
        model="fake-model",
        system=None,
        max_tokens=64,
        messages=[Message(role="user", content=[text_block(task)])],
    )


@oracle("unit-bad-tool", level="decision")
def bad_tool(resp: LLMResponse) -> Failure | None:
    for call in resp.tool_calls():
        if call.tool_name == "delete_file":
            return Failure(fingerprint="delete_file:outside_tmp")
        if call.tool_name == "list_dir":
            return Failure(fingerprint="list_dir:root")
    return None


async def judge(resp: LLMResponse) -> Failure | None:
    return await acheck(bad_tool, resp)


def runner(
    llm: FakeLLM, tmp_path: Path, *, max_cost: float = 2.0, concurrency: int = 4, **kw: object
) -> TrialRunner:
    return TrialRunner(
        llm,
        judge,
        cache=TrialCache(tmp_path / "cache.db"),
        budget=Budget(max_cost=max_cost),
        concurrency=concurrency,
        cache_namespace=llm.cache_namespace,
        **kw,
    )  # type: ignore[arg-type]


async def test_p095_decides_fails_quickly(tmp_path: Path) -> None:
    """BLUEPRINT M3: sequential test decides p≈0.95 at threshold 0.5 within 6 trials.

    With a 95% Wilson interval the earliest decision is at trial 4 (4/4). One pass among the
    first trials makes 6 too few (5/6 -> CI low 0.44), so at p=0.95 the share decided by
    trial 6 is about 0.95**4 ≈ 81%; all are decided `fails` by trial 8.
    """
    llm = FakeLLM(rules=[Rule(when_all=["delete"], respond=DELETE, probability=0.95)])
    r = runner(llm, tmp_path)
    by_six = eventually = 0
    trials_used = []
    n = 300
    for i in range(n):
        est = await r.failure_rate(
            req(f"please delete the project #{i}"),
            fingerprint="delete_file:outside_tmp",
            min_trials=3,
            max_trials=8,
            threshold=0.5,
        )
        trials_used.append(est.trials)
        by_six += est.decided == "fails" and est.trials <= 6
        eventually += est.decided == "fails"
    # Expected: P(decided by 6) = 0.95**4 = 0.81; P(decided `fails` by 8) = first four all fail,
    # or exactly one pass among the first four and none after = 0.8145 + 0.1397 = 0.954.
    assert by_six / n >= 0.75, by_six / n
    assert eventually / n >= 0.90, eventually / n
    assert sorted(trials_used)[n // 2] == 4  # median


async def test_repeat_is_served_from_cache_with_zero_provider_calls(tmp_path: Path) -> None:
    llm = FakeLLM(rules=[Rule(when_all=["delete"], respond=DELETE, probability=0.7)])
    first = runner(llm, tmp_path)
    a = await first.failure_rate(
        req(), fingerprint="delete_file:outside_tmp", min_trials=10, max_trials=10, early_stop=False
    )
    calls_after_first = llm.calls
    second = runner(llm, tmp_path)
    b = await second.failure_rate(
        req(), fingerprint="delete_file:outside_tmp", min_trials=10, max_trials=10, early_stop=False
    )
    assert calls_after_first == 10 and llm.calls == 10  # nothing new reached the provider
    assert second.stats.provider_calls == 0 and second.stats.cache_hits == 10
    assert second.stats.cost_usd == 0 and a == b


async def test_other_fingerprints_count_as_slippage(tmp_path: Path) -> None:
    llm = FakeLLM(
        rules=[
            Rule(when_all=["delete"], respond=DELETE, probability=0.5),
            Rule(when_all=["delete"], respond=LIST),
        ]
    )
    est = await runner(llm, tmp_path).failure_rate(
        req(), fingerprint="delete_file:outside_tmp", min_trials=40, max_trials=40, early_stop=False
    )
    assert est.trials == 40
    assert 0 < est.failures < 40
    assert est.slippage == {"list_dir:root": 40 - est.failures}
    anything = await runner(llm, tmp_path).failure_rate(
        req(), fingerprint="*", min_trials=40, max_trials=40, early_stop=False
    )
    assert anything.failures == 40 and anything.slippage == {}


class ExactCostFake(FakeLLM):
    """Charges exactly its estimate, so the number of affordable calls is predictable."""

    def cost(self, resp: LLMResponse, req: LLMRequest | None = None) -> float:
        assert req is not None
        return self.estimate_cost(req)


async def test_budget_stops_early_with_partial_result(tmp_path: Path) -> None:
    llm = ExactCostFake(rules=[Rule(when_all=["delete"], respond=DELETE)])
    per_call = llm.estimate_cost(req())
    r = runner(llm, tmp_path, max_cost=per_call * 2.5, concurrency=1)
    est = await r.failure_rate(
        req(), fingerprint="delete_file:outside_tmp", min_trials=10, max_trials=10, early_stop=False
    )
    assert est.partial and est.decided == "undecided"
    assert est.trials == 2 and llm.calls == 2
    assert est.note and "budget" in est.note and "partial" in est.note
    assert r.budget.spent <= r.budget.max_cost < r.budget.spent + per_call


async def test_budget_counts_actual_cost_not_the_estimate(tmp_path: Path) -> None:
    llm = FakeLLM(rules=[Rule(when_all=["delete"], respond=DELETE)])  # actual < estimate
    per_call = llm.estimate_cost(req())
    r = runner(llm, tmp_path, max_cost=per_call * 2.5, concurrency=1)
    est = await r.failure_rate(
        req(), fingerprint="*", min_trials=40, max_trials=40, early_stop=False
    )
    assert est.partial and 2 < est.trials < 40
    assert r.budget.spent <= r.budget.max_cost < r.budget.spent + per_call
    assert r.stats.cost_usd == pytest.approx(r.budget.spent)


async def test_budget_is_not_overspent_by_concurrent_trials(tmp_path: Path) -> None:
    llm = ExactCostFake(rules=[Rule(when_all=["delete"], respond=DELETE)])
    per_call = llm.estimate_cost(req())
    r = runner(llm, tmp_path, max_cost=per_call * 3.5, concurrency=8)
    est = await r.failure_rate(req(), fingerprint="*", min_trials=8, max_trials=8, early_stop=False)
    assert est.partial and llm.calls == 3 and est.trials == 3


class Flaky:
    """Fails with a retryable error `fail_times` times, then answers."""

    name = "flaky"

    def __init__(
        self, fail_times: int, *, retryable: bool = True, retry_after: float | None = None
    ) -> None:
        self.fail_times = fail_times
        self.retryable = retryable
        self.retry_after = retry_after
        self.attempts = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, req: LLMRequest, *, seed: int | None = None) -> LLMResponse:
        self.attempts += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.001)
        self.in_flight -= 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ProviderError(
                "overloaded",
                retryable=self.retryable,
                status_code=529,
                retry_after=self.retry_after,
            )
        return text("ok")

    def estimate_cost(self, req: LLMRequest) -> float:
        return 0.0


async def no_sleep(_: float) -> None:
    return None


async def test_retries_with_backoff_on_retryable_errors() -> None:
    delays: list[float] = []

    async def record_sleep(d: float) -> None:
        delays.append(d)

    flaky = Flaky(fail_times=3)
    r = TrialRunner(flaky, judge, cache=None, budget=Budget(), sleep=record_sleep)
    assert (await r.sample(req(), 0)).text() == "ok"
    assert r.stats.retries == 3 and flaky.attempts == 4
    assert [0.5 <= delays[0] <= 1.0, 1.0 <= delays[1] <= 2.0, 2.0 <= delays[2] <= 4.0] == [
        True,
        True,
        True,
    ]


async def test_waits_as_long_as_the_provider_asks_unless_it_is_a_daily_quota() -> None:
    delays: list[float] = []

    async def record_sleep(d: float) -> None:
        delays.append(d)

    r = TrialRunner(
        Flaky(fail_times=1, retry_after=30.0),
        judge,
        cache=None,
        budget=Budget(),
        sleep=record_sleep,
    )
    assert (await r.sample(req(), 0)).text() == "ok"
    assert 30.0 <= delays[0] <= 31.0  # per-minute quota: wait what the provider said

    daily = Flaky(fail_times=1, retry_after=MAX_RETRY_WAIT * 100)
    r2 = TrialRunner(daily, judge, cache=None, budget=Budget(), sleep=record_sleep)
    with pytest.raises(ProviderError):
        await r2.sample(req(), 0)  # daily quota: fail at once, don't sleep for hours
    assert daily.attempts == 1 and len(delays) == 1


async def test_gives_up_after_max_retries_and_never_retries_other_errors() -> None:
    r = TrialRunner(Flaky(fail_times=10), judge, cache=None, budget=Budget(), sleep=no_sleep)
    with pytest.raises(ProviderError):
        await r.sample(req(), 0)
    assert r.stats.retries == 5
    plain = Flaky(fail_times=1, retryable=False)
    r2 = TrialRunner(plain, judge, cache=None, budget=Budget(), sleep=no_sleep)
    with pytest.raises(ProviderError):
        await r2.sample(req(), 0)
    assert plain.attempts == 1


async def test_concurrency_limit_is_respected() -> None:
    flaky = Flaky(fail_times=0)
    r = TrialRunner(flaky, judge, cache=None, budget=Budget(), concurrency=3)
    await r.failure_rate(req(), fingerprint="*", min_trials=12, max_trials=12, early_stop=False)
    assert flaky.attempts == 12 and flaky.max_in_flight == 3


def test_cache_keys_separate_rule_sets_models_and_temperatures(tmp_path: Path) -> None:
    from agentbisect.store.cache import trial_cache_key

    a = FakeLLM(rules=[Rule(when_all=["x"], respond=DELETE)])
    b = FakeLLM(rules=[Rule(when_all=["x"], respond=DELETE, probability=0.5)])
    assert a.cache_namespace != b.cache_namespace
    base = req()
    keys = {
        trial_cache_key(base, namespace=a.cache_namespace),
        trial_cache_key(base, namespace=b.cache_namespace),
        trial_cache_key(base.model_copy(update={"model": "other"}), namespace=a.cache_namespace),
        trial_cache_key(base.model_copy(update={"temperature": 0.0}), namespace=a.cache_namespace),
    }
    assert len(keys) == 4


def test_cache_redacts_responses_before_writing(tmp_path: Path) -> None:
    cache = TrialCache(tmp_path / "cache.db")
    secret = "sk-ant-api03-Zx9aB7cD5eF3gH1iJ0kL2mN4oP6qR8sT"
    stored = cache.put("k", 0, text(f"your key is {secret}"))
    assert secret not in stored.text() and "[REDACTED:anthropic_key:" in stored.text()
    assert cache.get("k", 0) == stored and cache.get("k", 1) is None
    assert secret.encode() not in (tmp_path / "cache.db").read_bytes()
