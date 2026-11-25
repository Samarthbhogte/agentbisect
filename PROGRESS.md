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
- [ ] `relay.wrap` (Anthropic, OpenAI; sync, async, streaming), `@relay.tool`, `relay.run`, `@relay.entrypoint`
- [ ] Dependency linking, redaction, `relay init`, `relay record`, `relay ls`, `relay show`
- [ ] Three example agents run with `--provider fake`
- [ ] Accept: file_cleanup run recorded with correct order, kinds, `depends_on`, tokens
- [ ] Accept: simulated disk error does not raise into user code
- [ ] Accept: secret patterns never appear in stored files
- [ ] Accept: `relay show last` renders the run

## M3 — Decision replay, oracles, statistical engine
- [ ] `replay/decision.py`, `oracles/`, `stats/`, cache, budget, `relay try`
- [ ] Accept: Wilson 8/10 → [0.490, 0.943]
- [ ] Accept: sequential test decides p≈0.95 at threshold 0.5 within 6 trials
- [ ] Accept: repeated `relay try` makes zero provider calls
- [ ] Accept: `--max-cost 0.01` stops early with partial result
- [ ] Accept: other fingerprints counted as slippage

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

## Known gaps
- `providers/pricing.py` and the real Anthropic/OpenAI providers are not written yet. M1 only
  needs FakeLLM; they come with the SDK wrappers (M2) and the budget logic (M3).
