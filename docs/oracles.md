# Writing oracles

An oracle decides whether something failed and names the failure. It returns `None` (passed) or a
`Failure(fingerprint=..., detail=...)`.

## Decision oracles

Used by `try`, `shrink`, `explain` and generated tests. They receive one `LLMResponse` (and,
optionally, the `LLMRequest` it answered):

```python
from agentbisect import Failure, oracle
from agentbisect.providers.base import LLMRequest, LLMResponse


@oracle("deletes-outside-tmp", level="decision")
def deletes_outside_tmp(resp: LLMResponse) -> Failure | None:
    for call in resp.tool_calls():                      # tool_use blocks
        path = (call.tool_input or {}).get("path", "")
        if call.tool_name == "delete_file" and not path.startswith("/tmp/"):
            return Failure(fingerprint="delete_file:outside_tmp", detail=path)
    return None


@oracle("answers-in-english")
def answers_in_english(resp: LLMResponse, req: LLMRequest) -> Failure | None:
    ...                                                 # two parameters: also gets the request
```

`resp.text()` joins the text blocks; `resp.tool_calls()` lists the tool calls with `tool_name` and
`tool_input`.

## Run oracles

Used by `relay bisect`. They receive the whole recorded `Run`:

```python
from agentbisect.schema import Run


@oracle("wrong-refund", level="run")
def wrong_refund(run: Run) -> Failure | None:
    answer = str((run.output or {}).get("answer", ""))
    if "60 days" in answer:
        return Failure(fingerprint="refund_window:60_days", detail=answer[:80])
    return None
```

`run.steps` has every LLM and tool call (`step.kind`, `step.name`, `step.request`,
`step.response`).

## Choosing fingerprints

A fingerprint should name the *kind* of failure, not its details: `delete_file:outside_tmp`, not
the path. AgentBisect only counts a trial as failing when the fingerprint matches the one being
investigated; anything else is slippage and is reported separately. Too specific and harmless
variation looks like a different bug; too generic and shrinking can drift to a different bug.

## Built-in oracles

```python
from agentbisect.oracles.builtin import judge, loops, output_matches, output_not_matches, tool_called

tool_called("delete_file", where=lambda args: not args["path"].startswith("/tmp/"))
output_matches(r"60[- ]day")                       # fails when the text matches
output_not_matches(r"\b30 days\b")                 # fails when it does not
loops(max_repeats=3)                               # run oracle: same tool, same args, > 3 times
judge("Fail if the reply promises a refund the policy doesn't allow.", model="your-model-id")
```

`judge` asks an LLM to grade the response and returns `judge:<category>` fingerprints. Reports
flag judge-based results as noisier, since the judge is itself a model.

## Loading oracles

On the command line, refer to an oracle as `path/to/oracles.py:name` or `package.module:name`.
Files inside a package are imported by their dotted name, so relative imports work. You can also
register names in `agentbisect.toml`:

```toml
[oracles]
no-outside-deletes = "myagent/oracles.py:deletes_outside_tmp"
```

and use `--oracle no-outside-deletes`.

## Errors

An exception inside an oracle is an **oracle error**, never a failure: the command stops with
exit code 2 and the error. Returning anything other than `Failure` or `None` is also an error.
Oracles can be `async`.
