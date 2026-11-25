# Example agents

Three small agents built on the Anthropic SDK and wrapped with AgentRelay. Each has a planted bug,
and each runs fully offline with `--provider fake`, which answers through FakeLLM rules
(`fake_rules.py`) instead of a real model. Run them from the repository root.

| Agent | Bug | Ground-truth culprit |
|---|---|---|
| [`file_cleanup_agent`](file_cleanup_agent/README.md) | Deletes the whole project | A prompt injection hidden in README.md |
| [`refund_agent`](refund_agent/README.md) | Quotes a 60-day refund window instead of 30 | An outdated policy document in search results |
| [`calendar_agent`](calendar_agent/README.md) | Books meetings in UTC | The `create_event` tool description |

Every example takes `--seed N` (which FakeLLM sample to draw) and `--fixed` (apply the documented
fix). To run against a real model, use `--provider anthropic` with `ANTHROPIC_API_KEY` set and
pick the model with `--model <model id>` or the `AGENTRELAY_MODEL` environment variable. With
`--provider fake` the model name doesn't matter.
