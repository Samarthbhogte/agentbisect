# AgentRelay

**git bisect + delta debugging for AI agents.**

AgentRelay takes a failed agent run, finds which step and which piece of context caused the
failure, shrinks it to the smallest reproducible case, and emits a regression test.

Work in progress. See [BLUEPRINT.md](BLUEPRINT.md) and [PROGRESS.md](PROGRESS.md).
