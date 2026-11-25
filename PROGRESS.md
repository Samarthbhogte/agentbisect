# AgentRelay — Progress

Milestones from BLUEPRINT.md section 8. A milestone is ticked only when every acceptance
criterion passes.

## M0 — Scaffolding
- [x] `uv` project, `src/` layout, extras `[anthropic]`, `[openai]`, `[dev]`, console script `relay`
- [x] ruff, mypy, pytest configured; pytest-socket blocks network by default
- [x] `.github/workflows/ci.yml` (lint, type check, tests on 3.11 and 3.12)
- [x] PROGRESS.md
- [x] Accept: `uv run pytest` passes; `ruff check` and `mypy` clean; `relay --version` prints version

## M1 — Schema, storage, FakeLLM
- [x] `schema.py`, `store/` (runs, blobs, index), `providers/base.py`, `providers/fake.py`
- [x] Accept: 50-step Run round trip (both `save_run` and incremental `append_step`)
- [x] Accept: strings > 16 KB go to blobs and are restored (plus a hypothesis identity test)
- [x] Accept: FakeLLM scripted + rules; determinism; p=0.8 fires 399/500 = 79.8%; `when_all`/`unless_any`

## M2 — Capture SDK
- [x] `relay.wrap` (Anthropic, OpenAI; sync, async, streaming), `@relay.tool`, `relay.run`, `@relay.entrypoint`
- [x] Dependency linking, redaction, `relay init`, `relay record`, `relay ls`, `relay show`
- [x] Three example agents run with `--provider fake` (each bug reproduces; `--fixed` removes it, 0/40)
- [x] Accept: file_cleanup run recorded with correct order, kinds, `depends_on`, tokens
- [x] Accept: simulated disk error does not raise into user code
- [x] Accept: secret patterns never appear in stored files
- [x] Accept: `relay show last` renders the run
- Measured recording overhead (no disk I/O): median 0.18 ms, p95 0.23 ms per LLM step with a
  21-message, ~17 KB request. Target: < 2 ms.

## M3 — Decision replay, oracles, statistical engine
- [x] `replay/decision.py`, `oracles/`, `stats/`, cache, budget, `relay try`
- [x] Accept: Wilson 8/10 → [0.490, 0.943] (plus five more reference values and property tests)
- [x] Accept: sequential test decides p≈0.95 at threshold 0.5 within 6 trials — **as often as the
  math allows**: 4/4 is the earliest possible decision, and one early pass pushes it past 6
  (5/6 has CI low 0.44). Expected 0.95⁴ = 81% decided by trial 6; the test checks ≥ 75% over 300
  requests, a median of 4 trials, and ≥ 90% decided `fails` by trial 8 (expected 95.4%).
- [x] Accept: repeated `relay try` makes zero provider calls (FakeLLM call counter and runner stats)
- [x] Accept: `--max-cost 0.01` stops early with a partial result (exit code 3)
- [x] Accept: other fingerprints counted as slippage
- Coverage: stats 97–100%, replay 92%, oracles 94–95%, cache 94%.

## M4 — Context decomposition and shrink
- [ ] `context/units.py`, `context/constraints.py`, `algo/ddmin.py`, `relay shrink`, artifact
- [ ] Accept: `recompose(req, all_ids) == req` (hypothesis)
- [ ] Accept: every recomposed request passes `validate()`
- [ ] Accept: all ddmin properties from 6.8
- [ ] Accept: file_cleanup minimal set contains injected chunk, ≤ 3 removable units

## M5 — Explain and reports
- [ ] `algo/explain.py`, interactions, terminal + HTML report, `--json`
- [ ] Accept: ground-truth culprit ranks first on all three examples
- [ ] Accept: two-culprit case reports the right interaction
- [ ] Accept: `report.html` makes no network requests

## M6 — Test emitter and CI
- [ ] `emit/`, `run_scenario`, current-prompt override, `relay ci`, GitHub Action
- [ ] Accept: emitted test fails on buggy prompt, passes after fix
- [ ] Accept: `relay ci` exit codes and Markdown summary

## M7 — Program replay, diff, bisect
- [ ] `replay/program.py`, `cassette.py`, `determinism.py`, `relay replay`, `relay diff`, `relay bisect`
- [ ] Accept: strict replay makes zero live calls, same output
- [ ] Accept: changed system prompt → `ReplayDivergence` with readable diff
- [ ] Accept: bisect finds stale-document step on refund_agent
- [ ] Accept: side-effect tool never runs in replay unless allowed

## M8 — RelayBench
- [ ] `bench/generate.py`, `bench/run_bench.py`, baselines, `bench/RESULTS.md`
- [ ] Accept: offline in < 3 min; targets met or gap reported honestly

## M9 — Docs and launch polish
- [ ] README, quickstart, docs/, PyPI prep (no publish without go-ahead)

## Decisions
- M0: the repo root is this folder (it already held BLUEPRINT.md), not a nested
  `agentrelay/` directory.
- M0: dev Python pinned to 3.12 via `.python-version`; package supports `>=3.11`.
- M0: mypy runs over all of `src/agentrelay`, with `strict = true` only for `algo`, `stats`,
  `context`, and `schema` as the blueprint requires. `warn_unused_configs` is off so the strict
  override for not-yet-written modules doesn't produce noise.
- M0: `--live` is a custom pytest option in `tests/conftest.py`; it selects only `live` tests and
  re-enables sockets.
- M0: ruff excludes `*.md`. ruff 0.17 formats Python code blocks inside Markdown, which would
  rewrite the spec's code samples in BLUEPRINT.md.
- M1: FakeLLM seeds its RNG from `sha256(canonical(request) + seed)`, not Python's `hash()`, which
  is salted per process and would break reproducibility across runs and in CI.
- M1: FakeLLM rule matching is case-insensitive substring matching over `render_context(req)`:
  system prompt, every tool's name and description, and every message's text, tool names and tool
  inputs. The first rule that matches and wins its coin flip responds. A seed of `None` means 0.
- M1: FakeLLM takes either `script` or `rules`, not both. A scripted FakeLLM raises
  `FakeScriptExhausted` when it runs out. It counts calls in `.calls` (for M3's zero-call check).
- M1: FakeLLM prices default to $1 / $5 per million input / output tokens; tokens ≈ len(text)/4.
  `estimate_cost` charges `max_tokens` for output, so it is an upper bound.
- M1: `steps.jsonl` starts with a header line `{"schema": "agentrelay/v1"}`; JSON artifacts get a
  `schema` key; `run.json` has it via the `Run` alias. Text artifacts (HTML) carry no marker.
- M1: a torn final line in `steps.jsonl` (crash mid-write) is skipped on load; a bad line anywhere
  else raises.
- M1: run ids are resolved by scanning `runs/` on disk, so `last` and prefixes work even if
  `index.db` is stale. The index is updated at `start_run` and `finish_run`, not per step, to keep
  step overhead low; its `n_steps` for a crashed run is therefore stale.
- M1: blob threshold is > 16,384 UTF-8 bytes. A user value that is itself exactly
  `{"$blob": "<str>"}` would be misread as a blob reference; accepted as a known edge case.
- M1: added `ids.py` (ULID generator, no extra dependency) and the pydantic mypy plugin.
- M1: `Failure`, `ContextUnit`, and `RateEstimate` are defined in `schema.py` now, as the repo
  layout lists them there; their behavior arrives in M3/M4.

- M2: anthropic 2.x and openai 3.x are built on `httpx2`, not `httpx`, and reject `httpx` clients.
  `providers/fake_http.py` serves FakeLLM behind an `httpx2.MockTransport` that speaks each API's
  real wire format (JSON and server-sent events), so tests and `--provider fake` run the genuine
  SDK code paths, streaming included, with no network. A FakeLLM exception becomes an HTTP 500.
- M2: async methods in these SDKs are not `async def`, so the wrapper calls the real method and
  checks whether the result is awaitable instead of inspecting the function.
- M2: added `"other"` to `ContentBlock.type` for blocks the spec's list doesn't cover (thinking,
  documents, server tools, reasoning items). They are kept verbatim in `raw` and will be treated
  as non-removable, like images.
- M2: redaction runs inside `RunStore` (every write goes through `_prepare`: redact, then blob),
  so nothing can reach disk unredacted, whoever writes it. `request_hash` is computed on the
  unredacted request, so M7's cassette matching compares like with like; it is a one-way hash.
- M2: OpenAI normalization joins system/developer messages into `system` (their position is
  lost), merges consecutive same-role messages, and uses `max_tokens = 4096` when the request sets
  no limit, recording the original parameter name in `extra["max_tokens_param"]`.
- M2: a tool call depends on the LLM call whose `tool_use` block requested it (matched by name and
  arguments, most recent first). The spec only defines LLM → tool links; this is the reverse edge.
  Its `tool_use_id` is stored in `tags["tool_use_id"]`, and `tags["side_effects"]` records the flag.
- M2: the active run is found through a context variable, then a process-wide stack of active
  runs (so calls from worker threads still land in the run), then an implicit run named from
  `AGENTRELAY_RUN_NAME` or the script name, finished at exit with status `unknown`.
- M2: `relay.run()` returns its recorder (`set_output()`, `label()`, `id`). `@relay.entrypoint`
  records `module:qualname` (resolving `__main__` for `python -m`) and, if the run set no output,
  stores the return value as `{"return": value}`.
- M2: `relay record` reports new runs by comparing run ids on disk before and after the command,
  and exits with the command's exit code. It names runs after the module or script, skipping
  generic names like `agent` or `main`.
- M2: in examples, `run.input` holds all of `main()`'s arguments, so M7 can call
  `entrypoint(**run.input)`. Each example has `--fixed` for its documented fix.
- M2: FakeLLM rules match substrings anywhere in the context, case-insensitively, so rule marker
  strings must not appear in prompts or tool specs. Two such collisions in the examples were found
  and fixed while building M2; `test_examples_reproduce_their_bug_and_the_fix_removes_it` now
  checks every example's bug and fix rates.
- M3: `RateEstimate` gained `threshold`, `partial` and `note`, and a `still_fails()` method:
  `fails`, or `undecided` with `p_hat >= threshold` (the spec's "decide by p_hat at max_trials").
- M3: Wilson bounds are exactly 0 at k = 0 and exactly 1 at k = n (floating point otherwise gave
  0.99999…, breaking `p_hat <= ci_high`).
- M3: the cache key adds a provider namespace to the spec's `sha256(request + model + temperature)`:
  `anthropic`, `openai`, or a hash of the FakeLLM rule set. Without it, a fake run and a real run
  of the same request (or two FakeLLM rule sets) would share cached answers.
- M3: the cache redacts responses before writing and returns the stored copy, so a first run and a
  cached re-run judge identical text.
- M3: the budget reserves each trial's estimated cost before calling (so concurrent trials can't
  overspend) and then settles to the actual cost. Estimates charge the full `max_tokens` of output,
  so they are upper bounds. The run stops when the *next* trial's estimate would not fit.
- M3: trials run in batches of `--concurrency` but are judged in index order, so early stopping
  gives the same answer as running them one at a time (a few extra cached samples may be paid for).
- M3: retries: an explicit `retryable` flag on the error wins; otherwise HTTP 429 and 5xx are
  retried, up to 5 times with exponential backoff (0.5 s base) plus jitter. SDK errors are wrapped in
  `ProviderError`.
- M3: `providers/pricing.py` only knows FakeLLM's price. Real prices go in `agentrelay.toml`
  `[prices]`; an unknown model is charged a deliberately high $15 / $75 per Mtok, so a budget can
  only be exceeded on the safe side. (I did not want to hardcode prices I couldn't verify.)
- M3: `relay try` runs exactly `--trials` trials (no early stop) and reports the rate and Wilson
  interval. The target fingerprint is `--fingerprint`, else the one the recorded response fails
  with, else any failure (`*`). Without `--oracle` it shows the response distribution only.
  Exit codes: 1 run not found, 2 bad arguments or oracle error, 3 partial result (budget).
- M3: decision oracles may take `(resp)` or `(resp, req)` and may be `async`. Plain functions
  loaded with `FILE:NAME` become decision oracles. Files inside a package are imported by dotted
  name, so relative imports in `fake_rules.py` / `oracles.py` work.
- M3: `OpenAIProvider` replays through Chat Completions (normalized requests from the Responses API
  are replayed as chat requests).
- M3: found and fixed a third FakeLLM marker collision (file_cleanup's "Deleted " marker matched
  "what you deleted" in the system prompt, so the safe path never deleted the temp file). The
  example test now also checks the safe path.

## Known gaps
- No real model prices are built in (see Decisions); set them in `agentrelay.toml`. Recorded
  `cost_usd` on captured steps is still `None`.
- `judge()` calls are not cached or counted against the budget yet.
- Live tests against real APIs (`tests/live/`) are not written yet; everything so far is verified
  offline through the real SDKs with a mocked transport.
- Not intercepted: Anthropic `messages.stream()` helper, `messages.batches`, beta namespaces, and
  OpenAI `responses.stream()` / `chat.completions.parse()`. Only the methods the spec names are.
- A stream abandoned without being exhausted, closed, or exited is never recorded.
