"""`relay emit-test`: write `scenarios/<name>.json` and `test_<name>.py` (BLUEPRINT 6.11)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, PackageLoader

from agentrelay import __version__
from agentrelay.algo.shrink import ShrinkResult
from agentrelay.sdk.redact import Redactor
from agentrelay.testing import Scenario

_env = Environment(
    loader=PackageLoader("agentrelay.emit", "templates"),
    keep_trailing_newline=True,
    autoescape=False,
)


class EmitError(Exception):
    pass


@dataclass
class Emitted:
    scenario: Path
    test: Path
    name: str
    warnings: list[str]


def slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return out or "failure"


def emit_test(
    shrink: ShrinkResult,
    *,
    out_dir: Path,
    culprits: list[str] | None = None,
    trials: int = 10,
    max_failure_rate: float = 0.1,
    system_prompt: str | None = None,
    tools: str | None = None,
    redactor: Redactor | None = None,
    overwrite: bool = False,
) -> Emitted:
    if shrink.minimal_request is None or shrink.status not in ("ok", "unstable"):
        raise EmitError(
            f"the shrink result for {shrink.step_id} has status {shrink.status!r}; "
            "a confirmed minimal failing context is needed to emit a test"
        )
    if not shrink.oracle_ref:
        raise EmitError("the shrink artifact does not record its oracle; re-run `relay shrink`")
    fp = slug(shrink.fingerprint)
    name = f"{(shrink.run_id or 'run')[:6]}_{shrink.step_id}_{fp}"
    scenario = Scenario(
        name=name,
        run_id=shrink.run_id or "",
        step_id=shrink.step_id or "",
        oracle=shrink.oracle_ref,
        fingerprint=shrink.fingerprint,
        request=shrink.minimal_request,
        culprits=culprits or shrink.minimal_removable_ids,
        trials=trials,
        max_failure_rate=max_failure_rate,
        baseline=shrink.baseline,
        minimal_estimate=shrink.minimal_estimate,
        recorded_provider=shrink.recorded_provider,
        fake_rules=shrink.fake_rules,
        system_prompt=system_prompt,
        tools=tools,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        agentrelay=__version__,
    )
    scenario_file = out_dir / "scenarios" / f"{name}.json"
    test_file = out_dir / f"test_{name}.py"
    if not overwrite and (scenario_file.exists() or test_file.exists()):
        raise EmitError(f"{test_file} already exists; pass --force to overwrite")
    payload = json.dumps(
        scenario.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False
    )
    found = (redactor or Redactor()).find(payload)
    warnings = [
        f"the scenario still matches the {kind} redaction pattern; check it before committing"
        for kind in found
    ]
    scenario_file.parent.mkdir(parents=True, exist_ok=True)
    scenario_file.write_text(payload + "\n", encoding="utf-8")
    test_file.write_text(
        _env.get_template("test_scenario.py.j2").render(
            name=name,
            func=f"test_{fp}_regression",
            scenario=f"scenarios/{name}.json",
            trials=trials,
            max_failure_rate=max_failure_rate,
            fingerprint=shrink.fingerprint,
            run_id=shrink.run_id,
            step_id=shrink.step_id,
            culprits=scenario.culprits,
        ),
        encoding="utf-8",
    )
    return Emitted(scenario=scenario_file, test=test_file, name=name, warnings=warnings)
