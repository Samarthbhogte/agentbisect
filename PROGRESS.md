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
- [ ] `schema.py`, `store/` (runs, blobs, index), `providers/base.py`, `providers/fake.py`
- [ ] Accept: 50-step Run round trip
- [ ] Accept: strings > 16 KB go to blobs and are restored
- [ ] Accept: FakeLLM scripted + rules; determinism; p=0.8 fires 70–90% over 500 seeds; `when_all`/`unless_any`

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

## Known gaps
- None yet.
