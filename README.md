# AgentBisect

**git bisect + delta debugging for AI agents.**
AgentBisect takes a failed agent run, finds *which step* and *which piece of context* caused the
failure, shrinks it to the smallest reproducible case, and turns it into a regression test that
fails until the bug is fixed.

![relay explain finds the prompt injection hidden in a README](docs/demo.svg)

*An agent asked to clean up temp files deleted the whole project. `relay explain` points at the
one chunk of a tool result that caused it: a hidden HTML comment telling the model to delete the
project directory. ([Terminal recording](docs/demo.cast), playable with `asciinema play`.)*

---

## Why

When an agent misbehaves you usually open a tracing UI, scroll through dozens of steps, and guess.
Replay tools let you re-run and fork a trace, but a human still has to find the cause. AgentBisect
answers the questions directly, and treats every answer statistically, because models are
nondeterministic:

| Question | Command | How |
|---|---|---|
| *Where* did the run go wrong? | `relay bisect` | Binary search over steps: replay the first *k* from the recording, continue live |
| *What* in the context caused it? | `relay shrink` | Probabilistic delta debugging (ddmin) over context units |
| *How sure* are we, and which unit matters most? | `relay explain` | Ablation and counterfactual rewrites, with confidence intervals |
| How do we stop it coming back? | `relay emit-test` | A pytest regression test built from the minimal failing case |

It is local-first (no telemetry, no hosted service), works with the Anthropic and OpenAI SDKs, and
runs in CI.

## Quickstart (no API key, under 5 minutes)

Every example agent can run against **FakeLLM**, an offline model that reproduces each bug, so you
can try the whole loop without a key. AgentBisect needs Python 3.11+ and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Samarthbhogte/agentbisect.git agentbisect
cd agentbisect
uv sync --all-extras

# 1. Record a run where the cleanup agent deletes the whole project
uv run python -m examples.file_cleanup_agent.agent --provider fake --seed 2
uv run relay show last

# 2. Find the cause (and write an HTML report next to the run)
uv run relay explain last --step auto \
  --oracle examples/file_cleanup_agent/oracles.py:deletes_outside_tmp \
  --provider fake --fake-rules examples/file_cleanup_agent/fake_rules.py --html

# 3. Turn it into a regression test that checks the fixed prompt
uv run relay emit-test last --step s0002 --out my_regressions \
  --system-prompt examples.file_cleanup_agent.agent:SYSTEM_FIXED
AGENTBISECT_PROVIDER=fake uv run pytest my_regressions

# 4. Find the first bad step of a multi-step failure (the refund agent quotes a stale policy)
uv run python -m examples.refund_agent.agent --provider fake --seed 7
uv run relay bisect last --oracle examples/refund_agent/oracles.py:wrong_refund
```

`explain` ranks the injected chunk first (removing it takes the failure rate from about 90% to 0%),
and `bisect` reports that the run is recoverable until step `s0003`, the `search_policy` call that
retrieved the outdated document. Point any command at a real model with `--provider anthropic` and
`ANTHROPIC_API_KEY` set.

## Use it on your agent

```python
import anthropic
import agentbisect as relay

client = relay.wrap(anthropic.Anthropic())      # or openai.OpenAI(); sync, async and streaming


@relay.tool                                     # records arguments, result, errors and timing
def read_file(path: str) -> str: ...


@relay.tool(side_effects=True)                  # never runs live during replay unless allowed
def delete_file(path: str) -> str: ...


@relay.entrypoint                               # what `relay replay` and `relay bisect` call
def main(task: str) -> str:
    with relay.run("cleanup", input={"task": task}) as run:
        answer = ...                            # your agent loop, using `client`
        run.set_output(answer)
        return answer
```

Then write an **oracle**: a function that says whether a response (or a whole run) failed, and
*which* failure it was.

```python
from agentbisect import Failure, oracle


@oracle("deletes-outside-tmp", level="decision")
def deletes_outside_tmp(resp) -> Failure | None:
    for call in resp.tool_calls():
        if call.tool_name == "delete_file" and not call.tool_input["path"].startswith("/tmp/"):
            return Failure(fingerprint="delete_file:outside_tmp", detail=call.tool_input["path"])
    return None
```

Run your agent as usual (or `relay record -- python my_agent.py`), then `relay explain last --step
auto --oracle oracles.py:deletes_outside_tmp`. See [writing oracles](docs/oracles.md).

## Commands

| Command | What it does |
|---|---|
| `relay init` | Create `agentbisect.toml` and add `.agentbisect/` to `.gitignore` |
| `relay record -- <cmd>` | Run a command with capture enabled and print the run id |
| `relay ls`, `relay show <run>` | List runs; show a run or one step |
| `relay try <run> --step S` | Re-run one decision N times and report its failure rate |
| `relay shrink <run> --step S --oracle F:N` | Smallest context that still fails |
| `relay explain <run> --step S --oracle F:N` | Rank the culprits, with intervals; `--html` report |
| `relay emit-test <run> --step S` | Write a pytest regression test and its scenario |
| `relay ci` | Run all regression scenarios; exit 1 on any regression |
| `relay replay <run>` | Re-run the agent from the recording (`--mode strict\|fork`, `--fork-at`) |
| `relay diff <runA> <runB>` | Align two runs step by step and show what changed |
| `relay bisect <run> --oracle F:N` | First step after which the run keeps failing |
| `relay doctor` | Check the store, config and API keys |

Runs are referred to by id, a unique prefix, or `last`. Analysis commands take `--json`, and
`--provider anthropic|openai|fake`, `--max-cost`, `--concurrency`. Every trial is cached, so
repeating a command costs nothing.

## How it works

```
  your agent ──► Capture SDK ──► .agentbisect/  (runs as JSONL, redacted; blobs; trial cache)
                                      │
            ┌─────────────────────────┼──────────────────────────┐
     decision-level replay      context units                program-level replay
     (rebuild one request)   (sections, messages,           (re-run the agent; serve
            │                 tool results, chunks,          recorded responses, go
            │                 tool descriptions)             live from step k)
            └──────────► statistical engine ◄────────────────────┘
                (repeated trials, Wilson intervals, sequential stopping, cache, budget)
                     │              │               │
                  shrink         explain          bisect   ──► reports, pytest tests, CI
```

- **Two replay levels.** Decision-level replay rebuilds the exact request of one LLM call, edits
  its context and sends it again; it needs no user code and powers `try`, `shrink`, `explain` and
  the generated tests. Program-level replay re-runs your entrypoint with the SDK serving recorded
  responses (through a copy of your own client, so your code gets real SDK objects) and powers
  `replay` and `bisect`.
- **Every check is a rate.** "Does this still fail?" is answered with repeated trials, a 95%
  Wilson interval and a sequential stopping rule. A trial only counts if the oracle reports the
  *same* failure fingerprint, so shrinking can't drift into a different bug.
- **Shrinking keeps requests valid.** Tool calls and their results stay paired, roles alternate,
  and removed tool results become a placeholder, so every candidate is a request the provider
  accepts.

More in [concepts](docs/concepts.md).

## Results (RelayBench)

[RelayBench](bench/) has 30 seeded synthetic cases (15 to 77 context units, 8 needing two
culprits together, failure rates 0.40 to 0.91). It runs offline in about 15 seconds.

| Target (v1) | Result |
|---|---|
| Top-1 culprit accuracy ≥ 85% | **100%** |
| Median shrink ratio ≥ 10× | **22.6×** |
| ≥ 40% fewer model calls than leave-one-out (cases > 30 units) | **59% fewer** |

| | AgentBisect | Leave-one-out | Random ablation (same budget) |
|---|---|---|---|
| False culprit rate | **0%** | 60% | 41% |
| Model calls per diagnosis (mean) | 168 | 356 | 168 |

Honest reading: top-1 accuracy doesn't separate the methods on this benchmark (planted triggers
make culprits crisp, and random ablation also reaches 100% at the same budget). What AgentBisect
adds is that it doesn't wrongly blame innocent context, it costs about half of leave-one-out, and
it hands you a minimal reproduction and a test. All 8 two-culprit cases were reported as the right
"and" interaction. These are synthetic, offline numbers, so treat them as an upper bound; a live
benchmark against a real model is not done yet. Full tables in [bench/RESULTS.md](bench/RESULTS.md).

## Limitations

- **Replay determinism is best effort.** `time.time`, `uuid4` and `random` are frozen during
  replay, and `datetime.now()` only in your entrypoint's module. Threads, subprocesses and network
  calls that don't go through a wrapped client are not controlled.
- **Statistics cost calls.** A low failure rate (under 20%) is refused as too rare to diagnose at
  the default budget; raise `--trials` or lower the temperature. Unknown model prices are charged
  at a deliberately high default so budgets err on the safe side; set real prices in
  `agentbisect.toml`.
- **Necessity is not blame.** Context the model needs (for example the user's profile) can be as
  necessary as the culprit; explain reports that as an interaction, and the counterfactual
  rewrite helps tell instructions from facts.
- Only the SDK methods named above are intercepted (`messages.create`,
  `chat.completions.create`, `responses.create`). Images and other non-text blocks are kept as is.

Full list: [docs/limitations.md](docs/limitations.md).

## Documentation

- [Concepts](docs/concepts.md): runs, steps, context units, oracles, fingerprints, rates
- [Writing oracles](docs/oracles.md): decision and run oracles, built-ins, LLM judges
- [CI guide](docs/ci.md): emitted tests, `relay ci`, the reusable GitHub workflow
- [Limitations](docs/limitations.md): determinism caveats and known gaps
- [Example agents](examples/README.md) and [RelayBench](bench/RESULTS.md)

## Status

Version 0.1, alpha. The distribution name is `agentbisect` (the import name is `agentbisect` and
the command is `relay`); it is not on PyPI yet, so install from source as above. Licensed under
[Apache-2.0](LICENSE).
