from __future__ import annotations

import pytest

from agentrelay.sdk.redact import Redactor

SECRETS = {
    "anthropic_key": "sk-ant-api03-Zx9aB7cD5eF3gH1iJ0kL2mN4oP6qR8sT",
    "openai_key": "sk-proj-4fGh7JkL9mNpQ2rStUvWxYz0123456789abc",
    "aws_access_key": "AKIAIOSFODNN7EXAMPLE",
    "github_token": "ghp_16C7e42F292c6912E7710c838347Ae178B4a",
    "slack_token": "xoxb-1234567890-abcdefghijkl",
}


@pytest.mark.parametrize(("kind", "secret"), SECRETS.items())
def test_each_secret_type_is_redacted(kind: str, secret: str) -> None:
    out = Redactor().redact_text(f"config: {secret} end")
    assert secret not in out
    assert f"[REDACTED:{kind}:" in out
    assert out.startswith("config: ") and out.endswith(" end")


def test_bearer_token_keeps_scheme() -> None:
    out = Redactor().redact_text("Authorization: Bearer abcdefghijklmnop.qrstuvwx")
    assert out.startswith("Authorization: Bearer [REDACTED:bearer_token:")


def test_same_secret_redacts_to_same_marker() -> None:
    r = Redactor()
    secret = SECRETS["openai_key"]
    a, b = r.redact_text(secret), r.redact_text(f"again {secret}")
    assert b == f"again {a}"
    assert r.redact_text(SECRETS["aws_access_key"]) != a


def test_no_false_positives_on_ordinary_text() -> None:
    text = (
        "task-management-system-v2 uses disk-space; risk-assessment-framework-docs; "
        "2026-12-01T15:00:00Z; order A1001; ghp_short"
    )
    assert Redactor().redact_text(text) == text


def test_pii_is_opt_in() -> None:
    text = "mail dana@example.com or call (617) 555-0123"
    assert Redactor().redact_text(text) == text
    out = Redactor(pii=True).redact_text(text)
    assert "dana@example.com" not in out and "555-0123" not in out
    assert "[REDACTED:email:" in out and "[REDACTED:phone:" in out
    assert (
        Redactor(pii=True).redact_text("2026-12-01T15:00:00-05:00") == "2026-12-01T15:00:00-05:00"
    )


def test_user_patterns_and_nested_structures() -> None:
    r = Redactor(extra_patterns={"internal": r"itk_[a-z0-9]{8}"})
    data = {"a": ["itk_abcd1234", {"b": SECRETS["github_token"]}], "n": 3, "ok": None}
    out = r.redact(data)
    assert out["n"] == 3 and out["ok"] is None
    assert out["a"][0].startswith("[REDACTED:internal:")
    assert out["a"][1]["b"].startswith("[REDACTED:github_token:")
    assert r.find("x itk_abcd1234") == ["internal"]
