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
- [x] `context/units.py`, `context/constraints.py`, `algo/ddmin.py`, `algo/shrink.py`, `relay shrink`, artifact
- [x] Accept: `recompose(req, all_ids) == req` (hypothesis, 200 generated requests, both removal modes)
- [x] Accept: every recomposed request passes `validate()` (hypothesis, 300 random keep-sets)
- [x] Accept: all ddmin properties from 6.8 (hypothesis, 400 scenarios with groups, chunks and
  non-removable units): result ⊆ input, still fails, 1-minimal, groups never split,
  non-removable kept, result equals the culprit set exactly, each question asked once
- [x] Accept: file_cleanup minimal set = the tool result + the injected chunk (2 removable units),
  225 of 590 tokens, confirmed at 12/16. Refund isolates the stale policy chunk; calendar keeps
  the `create_event` description.
- Coverage: context 98%, algo/stats/replay 92–100% (98% combined).

## M5 — Explain and reports
- [x] `algo/explain.py`, interactions, terminal + HTML report, `--json`, `relay explain`
- [x] Accept: ground-truth culprit ranks first on all three examples, as the only culprit:
  file_cleanup injected chunk (+90 pts), refund stale-policy chunk (+95 pts), calendar
  `create_event` description (+90 pts)
- [x] Accept: two-culprit case reports an "and" interaction between exactly `sys.0` and
  `msg.2.tr.0`; a redundant-cause case reports an "or" interaction for the right pair
- [x] Accept: `report.html` makes no network requests (no scripts at all, no external URLs;
  hostile content is escaped). Checked visually in a browser in dark, light and phone widths.
- Coverage: algo 94–100%, context 98–100%, report 92–99% (97% combined).

## M6 — Test emitter and CI
- [x] `emit/`, `agentrelay.testing.run_scenario`, current-prompt override, `relay emit-test`,
  `relay ci`, reusable workflow `.github/workflows/agentrelay-action.yml`, pytest marker plugin
- [x] Accept: the emitted file_cleanup test fails against the buggy prompt (7/10 > 10%) and passes
  after applying the documented fix, run through a real `pytest` process
- [x] Accept: `relay ci` exits 1 on regression and 0 otherwise, writes `agentrelay-report.json`
  and appends a Markdown table to `$GITHUB_STEP_SUMMARY`
- The repository's own CI runs the reusable workflow on a committed example regression test
  (`examples/file_cleanup_agent/regression/`), so the action is exercised on every push.

## M7 — Program replay, diff, bisect
- [x] `replay/program.py`, `cassette.py`, `determinism.py`, `diff.py`, `algo/bisect.py`,
  `relay replay`, `relay diff`, `relay bisect`
- [x] Accept: strict replay of all three examples makes zero live calls (FakeLLM call counter)
  and reproduces the same output; the replay's steps are identical to the recording
- [x] Accept: changing the system prompt raises `ReplayDivergence` at `s0000` ("the system prompt
  changed") with a unified diff; `relay replay` exits 5
- [x] Accept: on refund_agent, bisect reports `s0003`, the `search_policy("refund window")` call
  that retrieved the stale document (f = 25% up to s0002, 100% from s0003); suggests explain s0004
- [x] Accept: a `side_effects=True` tool that writes a sentinel file never runs in strict,
  fork/live or fork/cassette-or-live replay, and runs only with `allow_live_side_effects=True`
- Coverage: replay 92–100%, cassette transport 86%, bisect plumbing 85% (92% combined).

## M8 — RelayBench
- [x] `bench/generate.py` (30 seeded cases, 15–77 units, 8 two-culprit), `bench/run_bench.py`,
  baselines (leave-one-out, budget-matched random ablation), `bench/RESULTS.md`, `results.json`
- [x] Accept: offline in 15 s (CI job enforces a 3-minute limit); all three targets met:
  top-1 100% (≥ 85%), median shrink 22.6× (≥ 10×), 59% fewer model calls than leave-one-out on
  the 26 cases with > 30 units (≥ 40%). RESULTS.md also states what the targets don't show:
  random ablation matches top-1 at equal cost (false culprits 41% vs 0%), 2 low-rate cases ranked
  the culprit first without flagging it (doubled trials fix both), and leave-one-out is cheaper on
  small contexts.

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
- Examples: no built-in model id. `--model` or
  `AGENTRELAY_MODEL` sets it, and `--provider anthropic` without one is a clear usage error.
  Tests use `fake-model`. (Changing the model name changes FakeLLM's seeded draws, so the
  documented failing seeds are now 2 for file_cleanup, 7 for refund, and 1 for calendar.)
- M4: never removable: the first user message and the last user message with text (the spec's
  example marks `msg.0 (task)` this way). The conversation therefore always starts with the user.
- M4: in `placeholder` mode (default) assistant messages that call tools are not removable: their
  tool results stay as placeholders, so the calls must stay too. In `delete` mode the assistant
  message and the tool results answering it form one group (`pair.msg.<i>`) and go together.
  A user message unit covers its text; its tool results are separate units.
- M4: `recompose` treats a partially kept group as removed, and a chunk as removed when its tool
  result is; merging same-role messages puts tool results first. If every system section is
  removed, `system` becomes `None`; removed sections/chunks leave the rest joined and trimmed.
- M4: chunks pack paragraphs (then lines) greedily up to ~800 characters. The file_cleanup README
  and the refund policy documents were lengthened so they cross the ~400-token chunking threshold
  and each refund document is its own chunk; the documented failing seeds did not change.
- M4: `ddmin()` returns a `DDMinResult` (minimal ids, accepted candidates, step log); baseline,
  confirmation and back-off live in `algo/shrink.py`, whose `ShrinkResult` is the artifact.
  ddmin first asks whether removing everything removable still fails, and in pass 1 the chunks of
  every kept unit (including non-removable ones) are kept. Hypothesis found the latter bug.
- M4: `still_fails` uses `RateEstimate.still_fails()` (`fails`, or `undecided` at max trials with
  `p_hat >= threshold`), following 6.6's "decide by p_hat at max_trials" rather than 6.8's
  stricter `decided == "fails"`.
- M4: confirmation always runs the full 16 trials. Back-off walks the accepted candidates from
  smallest to largest (then the full request) and keeps the first that confirms.
- M4: `relay shrink` statuses and exit codes: ok/unstable 0, partial (budget) 3, too_rare or
  not_failing 4. A partial result carries the smallest failing candidate found so far.
  `--trials` sets the baseline trial count. The recorded response must fail the oracle, unless
  `--fingerprint` names the failure to look for.
- M5: FakeLLM rules gained `min_turn`/`max_turn` (turn = assistant messages so far). The example
  rules for earlier agent stages (read the README, look up the order, read the profile) re-fired
  whenever their trigger text was removed, which made that supporting context look exactly as
  necessary as the planted culprit. Gating them by turn models agent stages and leaves each
  example with one true culprit. Seeds and bug rates did not change. General caveat, stated in
  reports: necessity is not blame; with a real model, supporting context can be as necessary as
  the culprit, and explain reports that as an "and" interaction.
- M5: explain works at the finest level: a tool result whose chunks are in the minimal set is
  explained through its chunks. Removing a unit also removes its group mates and children.
- M5: attribution interval = Newcombe's hybrid score interval for p0 − p(Full without u), built
  from the two Wilson intervals. Culprit = the Wilson intervals do not overlap
  (`ci_high(without) < ci_low(p0)`). Every measurement runs a fixed number of trials (baseline 20,
  everything else 16) with no early stopping, so all intervals are comparable.
- M5: interactions come in two kinds. "and": two or more culprits, each necessary (the planted
  two-culprit case). "or": no single culprit, and a pair within M (|M| ≤ 6) whose joint removal
  fixes the failure; this is the spec's pair test. Ranking: culprits first, then attribution.
- M5: counterfactual substitution uses `--substitute UNIT=FILE` first, then an LLM rewrite with a
  real provider ("generated", cached through the trial cache), else a heuristic rewrite that
  drops comments and sentences with injection cues ("heuristic", used with FakeLLM). If the
  rewrite is identical, no trials run and the report says the failure comes from the facts.
  Counterfactuals are reported but do not change the ranking.
- M5: `relay explain` reuses `shrink-<step>.json` when its fingerprint and removal mode match,
  otherwise it shrinks first. Artifacts: `explain-<step>.json`, and with `--html`,
  `report.html` (one per run: the latest explain wins). Exit codes match shrink (no_culprit 0).
- M5: the HTML report has no JavaScript; the chart is server-rendered inline SVG. Text is escaped
  by Jinja autoescape, and culprit text is highlighted after escaping.
- M6: a scenario stores the minimal failing request, the oracle reference, fingerprint, culprits,
  trials, allowed failure rate, baseline, the provider the step was recorded with, and the
  FakeLLM rules file used for the diagnosis. Shrink artifacts now record `oracle_ref`,
  `recorded_provider`, `provider` and `fake_rules` so this is possible.
- M6: current-prompt overrides, in priority order: `run_scenario`/`relay ci` arguments, the
  references stored at emit time (`--system-prompt`, `--tools`), then `agentrelay.toml`
  (`system_prompt`, `tools`). The system override replaces the whole system prompt of the
  minimal request (that is where fixes usually go); the tools override replaces the tool specs.
  `AGENTRELAY_MODEL` overrides the model. References resolve against the project root found
  from the scenario file (nearest `agentrelay.toml`, else `pyproject.toml`).
- M6: provider: `AGENTRELAY_PROVIDER`, else the provider the step was recorded with. Without its
  API key a test is skipped (`pytest.skip`, or "skipped" in `relay ci`), unless
  `AGENTRELAY_REQUIRE_LIVE=1`, which makes it an error. With `fake`, `AGENTRELAY_FAKE_RULES` or
  the stored rules file is used.
- M6: a test passes when failures/trials ≤ `max_failure_rate` (defaults 10 trials, 10%). It runs
  all trials (no early stopping) and uses the project's trial cache, so unchanged scenarios cost
  nothing on re-runs. Per-scenario budget: `AGENTRELAY_MAX_COST` (default $1); `relay ci
  --max-cost` is a total spread across scenarios.
- M6: `relay emit-test` uses `explain-<step>.json` (its culprits) or `shrink-<step>.json`; with
  `--oracle` and no artifact it shrinks first. File names: `<run id[:6]>_<step>_<fingerprint
  slug>`. It refuses to overwrite without `--force`, and warns when the scenario still matches a
  redaction pattern.
- M6: `relay ci` exit codes: 1 if any scenario regressed or errored, else 0 (skips don't fail).
- M6: a `pytest11` entry point registers the `agentrelay` marker in projects that install
  agentrelay, so emitted tests run without "unknown marker" warnings.
- M7: served LLM calls go through a copy of the user's client (`client.with_options(http_client=
  ...)`) whose in-memory HTTP layer returns the recorded provider-native response, re-encoded as
  server-sent events for streaming calls. The agent receives genuine SDK objects; live calls use
  the original client. Proxies now carry the root client and attribute path for this.
- M7: a third tools mode, `cassette-or-live` (recorded result when name and arguments match,
  read-only tools live otherwise, side-effect tools refused). Strict `cassette` cannot continue a
  diverged run (a live model that chooses a new search query would raise `ToolNotInCassette`),
  so `relay bisect` defaults to `cassette-or-live`; `relay replay` keeps the spec's `cassette`.
- M7: `fork_at=k` counts all steps (LLM and tool): recorded LLM calls with index < k are served,
  later ones go live; with `tools=live`, recorded tool results before k are still served.
  Strict replay that makes fewer LLM calls than were recorded is also a divergence.
- M7: FakeLLM gained `SEED_OFFSET` (a context variable added to every seed). Program replay sets
  it to the trial seed, so live continuations draw fresh samples per trial the way a real model at
  temperature > 0 would; recordings (offset 0) are unchanged.
- M7: determinism: `time.time`, `uuid.uuid4` and `random` are frozen process-wide during replay;
  `datetime.now()` is frozen only in the entrypoint's module (its `datetime` name is swapped).
  Replacing `datetime.datetime` globally broke pydantic schema building inside the LLM SDKs, which
  happens lazily during a replay, so it is not done. Limitations are in the module docstring.
- M7: replays are recorded as new runs labeled `replay_of` / `replay_mode`; `last` now means the
  newest run that is not a replay. Bisect trials run with an in-memory store, so they don't fill
  `.agentrelay/`.
- M7: bisect measures each fork point sequentially (min 3, max `--trials` 8 replays, threshold
  0.5, trial seeds 1..n), binary-searches the first k that still fails, and reports the curve,
  whether it is monotone, and the first LLM call at or after k to explain. Statuses: ok,
  fails_from_start (f(0) already fails: the task itself fails), not_reproduced (exit 4).
- M7: `relay diff` aligns steps by LCS over (kind, name); aligned steps are "changed" when the
  request hash, response, or error differ, with unified diffs of each.
- M8: cases plant unique trigger codes ("COBALT-HARBOR-42") in system sections, tool
  descriptions and tool results (chunks); a rule fires `purge_archive` when all triggers are
  present (p 0.4–0.91), and a decoy rule on ordinary distractor text fires a different failure
  (`send_report`, p 0.3) so slippage is exercised. Ground truth = the finest units containing a
  trigger, checked by a test.
- M8: every method is charged in model calls that miss a per-method trial cache. Leave-one-out
  uses the same 20-trial baseline and 3–8-trial sequential test as AgentRelay's search; random
  ablation gets AgentRelay's exact call budget per case (4 trials per random subset).
- M8: top-1 = rank-1 unit is a culprit; top-3 = all culprits in the top 3; false culprit rate =
  wrongly flagged / all flagged; missed = culprits not flagged.
- M8: `run_bench.py` rewrites `RESULTS.md` only on a full run (`--cases N` is a quick look) and
  writes a dynamic "Findings" section, re-checking unflagged cases with doubled trials.

## Known gaps
- No real model prices are built in (see Decisions); set them in `agentrelay.toml`. Recorded
  `cost_usd` on captured steps is still `None`.
- `judge()` calls are not cached or counted against the budget yet.
- Live tests against real APIs (`tests/live/`) are not written yet; everything so far is verified
  offline through the real SDKs with a mocked transport.
- The opt-in live bench (the three examples against a real model, 10 repetitions each) is not
  built: it needs an API key and budget approval.
- Bisect has no cost budget (it runs whole agent replays, whose calls go through the user's own
  client), and its curve is not yet drawn in `report.html` (terminal and JSON only).
- Not intercepted: Anthropic `messages.stream()` helper, `messages.batches`, beta namespaces, and
  OpenAI `responses.stream()` / `chat.completions.parse()`. Only the methods the spec names are.
- A stream abandoned without being exhausted, closed, or exited is never recorded.
