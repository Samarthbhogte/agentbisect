# Concepts

## Runs and steps

A **run** is one end-to-end execution of an agent on one task. It is recorded as a sequence of
**steps**: LLM calls (`llm_call`), tool calls (`tool_call`), or custom events. Each step stores the
normalized request and response, timing, token counts, and `depends_on`: the earlier steps whose
outputs appear in its input. An LLM call depends on the tool calls whose results it sees; a tool
call depends on the LLM call that requested it.

Runs live in `.agentrelay/` (the nearest parent directory with `.agentrelay/` or
`pyproject.toml`, or `$AGENTRELAY_HOME`):

```
.agentrelay/
├── runs/<run id>/run.json        # the run without its steps
├── runs/<run id>/steps.jsonl     # one step per line, flushed as it happens (crash-safe)
├── runs/<run id>/artifacts/      # shrink-*.json, explain-*.json, bisect.json, report.html
├── blobs/<sha256>                # strings over 16 KB, stored once
├── index.db                      # run list for `relay ls`
└── cache.db                      # trial results: re-running a question costs nothing
```

Everything is **redacted before it is written**: API keys (Anthropic, OpenAI, AWS, GitHub, Slack,
bearer tokens), optionally emails and phone numbers (`redact_pii = true`), and your own patterns
from `agentrelay.toml`. A redacted value becomes `[REDACTED:<type>:<hash>]`, so equal secrets stay
equal. HTTP headers are never recorded.

Run ids are ULIDs (sortable by time). Commands accept a full id, a unique prefix, or `last` (the
newest run that is not a replay).

## Decisions and the normalized format

A **decision** is an LLM call whose output is judged. Internally every request and response uses a
provider-neutral format (`LLMRequest`, `LLMResponse`: system prompt, messages of text, tool-use and
tool-result blocks, tool specs). Adapters translate to and from the Anthropic Messages API and
OpenAI's Chat Completions and Responses APIs; blocks AgentRelay doesn't model (images, thinking,
documents) are kept as is.

## Oracles and fingerprints

An **oracle** is your function that decides whether a response (`level="decision"`) or a whole run
(`level="run"`) failed. It returns a `Failure` with a **fingerprint**, a short string naming *which*
failure, such as `delete_file:outside_tmp`. A trial only counts as failing when the fingerprint
matches the one being investigated; other failures are reported as **slippage**. This stops a
search from "shrinking" into a different, simpler bug. See [writing oracles](oracles.md).

## Failure rates, not booleans

Models are nondeterministic, so "does this still fail?" is a rate. AgentRelay runs repeated
**trials** (trial *i* always uses sample index *i*, so results are cached and reproducible),
computes a 95% **Wilson** interval, and stops early once the interval is clearly above or below the
threshold. The default threshold is `max(0.2, p0 / 2)`: a candidate still fails if it fails at
least half as often as the original. A baseline under 20% is refused as too rare to diagnose at
this budget.

Budgets: every command has `--max-cost` (default $2). Each trial's cost is estimated before it
runs (charging the full `max_tokens` of output) and settled to the actual cost after. Set model
prices in `agentrelay.toml` under `[prices]`.

## Context units

`shrink` and `explain` split one request into **context units**:

| Unit id | What it is | Removing it |
|---|---|---|
| `sys.2` | a system-prompt section (markdown heading, XML-style tag, or paragraph) | drops the section |
| `tool.delete_file` | a tool's description | empties the description; the tool stays callable |
| `msg.4` | a message (for user messages, its text) | drops it; neighbors with the same role merge |
| `msg.4.tr.0` | a tool result | replaces its content with `[omitted by agentrelay]` (or, with `--removal delete`, removes the call and the result) |
| `msg.4.tr.0.chunk.3` | a ~200-token chunk of a long tool result | drops that chunk |

The first user message and the last user message with text (the task) are never removed. Every
recomposed request is checked: tool calls and results stay paired, roles alternate, tool results
come first. `recompose(request, all units) == request`.

## Shrink: probabilistic ddmin

`relay shrink` runs delta debugging (Zeller & Hildebrandt's ddmin) over removable groups of units,
first at the level of sections, messages, tool results and tool specs, then inside the surviving
tool results at chunk level. Each question is asked once (memoized). The result is confirmed with
16 trials; if it no longer fails, AgentRelay backs off to the last candidate that does and marks
the result unstable.

## Explain: attribution

For each unit `u` of the minimal set, `relay explain` measures:

1. `p(minimal without u)` and, more importantly, `p(original without u)`: would removing it fix the
   real failure?
2. for tool results and chunks, a **counterfactual**: the same text rewritten to keep its facts but
   drop any instructions (from `--substitute UNIT=FILE`, an LLM rewrite, or a heuristic offline).

`attribution(u) = p0 − p(original without u)`, with a Newcombe interval. A unit is a **culprit**
when its interval and the baseline's do not overlap. Several culprits form an "and"
**interaction** (the failure needs them together); if no single unit is a culprit, pairs are tested
for an "or" interaction (either one is enough). The counterfactual tells apart text that
*instructs* the model (rewrite fixes it) from text with *wrong facts* (nothing to rewrite).

## Replay and bisect

**Decision-level replay** rebuilds one recorded request and sends it again, optionally with a
different model, temperature, system prompt or tool specs. **Program-level replay** imports your
`@relay.entrypoint` and calls it with the recorded input while the SDK serves recorded responses:

- `strict`: every LLM call must match the recording (by request hash) or replay stops with
  `ReplayDivergence` and a diff ("the system prompt changed").
- `fork` / `--fork-at k`: serve the first *k* steps from the recording, then continue live.
- tools: `cassette` (recorded results only), `live`, or `cassette-or-live`. Tools marked
  `side_effects=True` never run live unless you pass `--allow-live-side-effects`.

`relay bisect` measures `f(k)`, the failure rate when continuing live from step *k*, and binary
searches for the first *k* that keeps failing: "until s0002 the run is recoverable; from s0003 it
fails 100%: `search_policy` returned the outdated policy". It needs a run-level oracle.
