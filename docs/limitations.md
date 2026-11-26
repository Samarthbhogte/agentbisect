# Limitations

AgentBisect reports uncertainty instead of hiding it, and so should this page.

## Replay determinism

During program-level replay AgentBisect freezes `time.time()` (a clock starting at the recorded run's
start), `uuid.uuid4()` and the global `random` generator, seeded from the run id and trial seed.
`datetime.now()` / `utcnow()` are frozen **only in your entrypoint's module**: replacing the
`datetime` class process-wide breaks libraries (including the LLM SDKs) that build schemas from it
while the agent runs. Not controlled:

- code that bound `from time import time` before replay started, and `datetime.now()` in other
  modules,
- other threads and subprocesses, `os.urandom` / `secrets`,
- network calls that do not go through a client wrapped with `relay.wrap` (your agent's own HTTP
  calls, databases, files).

Tool results are served from the recording when the call matches, which covers most of this, but a
tool that reads changing state and runs live will see today's state.

Recorded values are redacted, so a replay serves redacted text; if a request contained a secret,
the next request can differ from the recording and strict replay will report a divergence.

## Statistics and cost

- Every check is a rate from repeated trials. Low failure rates need many trials: a baseline below
  20% is refused at the default budget, and an attribution can come out "no culprit" when the
  intervals overlap. Raise `--trials` (RelayBench shows doubled trials resolving both such cases).
- The sequential test can decide in 4 trials at best (4/4 failures); one early pass pushes the
  decision later.
- Built-in model prices are deliberately absent; unknown models are charged a high default
  ($15/$75 per million tokens) so a budget is only ever exceeded on the safe side. Set real prices
  in `agentbisect.toml`.
- `relay bisect` replays whole agent runs and has no cost budget yet. The LLM-as-judge oracle's
  own calls are not cached or budgeted.

## Interpretation

- **Necessity is not blame.** A unit whose removal fixes the failure may be the cause or may be
  context the model needs (the task, the user's profile). Several such units are reported as an
  "and" interaction. The counterfactual rewrite helps separate text that *instructs* the model
  from text that carries facts.
- Shrinking finds *a* 1-minimal failing context, not necessarily the only one. If two unrelated
  causes can each trigger the failure, the minimal set contains one of them; `explain` reports the
  pair as an "or" interaction when both are in its input.
- RelayBench is synthetic: its culprits are exact trigger strings, so accuracy there is an upper
  bound on what to expect from a live model. The live benchmark is not done yet.

## Coverage

- Intercepted: Anthropic `messages.create`, OpenAI `chat.completions.create` and
  `responses.create`, sync and async, streaming or not. Not intercepted: helper APIs such as
  `messages.stream()`, `responses.stream()`, `chat.completions.parse()`, batches and beta
  namespaces.
- Images and other non-text blocks pass through untouched and are never removed.
- OpenAI system/developer messages are joined into one system prompt (their position in the
  conversation is not kept); `OpenAIProvider` replays through Chat Completions.
- Single-agent decisions only: no attribution across cooperating agents.
- Not built yet: the OpenTelemetry/Langfuse/LangSmith importers, a web UI, and automatic fix
  suggestions.
