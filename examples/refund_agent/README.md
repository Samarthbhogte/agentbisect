# refund_agent

The agent answers "Am I still eligible for a refund?" for an order placed 45 days ago. It looks up
the order, searches the policy knowledge base, and answers.

`search_policy` returns five documents. For some queries ("refund window") one of them is an
outdated policy from 2024 that allowed refunds within 60 days. The current window is 30 days.

**Bug:** the agent tells the customer they qualify under a 60-day window. With FakeLLM this happens
in roughly 1 run in 4: the model first has to choose the query that retrieves the stale document
(20%), then quote it (90%). **Culprit:** the stale policy document in the search results.
**Fix:** a system-prompt rule to trust the document with the latest effective date (`--fixed`).

```bash
relay record -- python -m examples.refund_agent.agent --provider fake --seed 7
relay show last
```

Seed 7 is one of the failing runs. This example is the bisect target in milestone M7: the run is
recoverable until the stale document is retrieved, and fails almost every time after.
