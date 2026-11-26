"""Build a provider from CLI options: `--provider anthropic|openai|fake --fake-rules FILE`."""

from __future__ import annotations

from typing import Any

from agentbisect.providers.base import LLMProvider

PROVIDERS = ("anthropic", "openai", "fake")


class ProviderSetupError(ValueError):
    """The requested provider can't be built (unknown name, missing rules or SDK)."""


def make_provider(
    name: str,
    *,
    fake_rules: str | None = None,
    prices: dict[str, tuple[float, float]] | None = None,
) -> LLMProvider:
    if name == "fake":
        if not fake_rules:
            raise ProviderSetupError("--provider fake needs --fake-rules FILE (a file with RULES)")
        from agentbisect.providers.fake import FakeLLM

        return FakeLLM.from_rules_file(fake_rules)
    if name == "anthropic":
        _require("anthropic")
        from agentbisect.providers.anthropic import AnthropicProvider

        return AnthropicProvider(prices=prices)
    if name == "openai":
        _require("openai")
        from agentbisect.providers.openai import OpenAIProvider

        return OpenAIProvider(prices=prices)
    raise ProviderSetupError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")


def _require(package: str) -> Any:
    try:
        return __import__(package)
    except ImportError as exc:
        raise ProviderSetupError(
            f"the {package} SDK is not installed: pip install 'agentbisect[{package}]'"
        ) from exc


def cache_namespace(provider: LLMProvider) -> str:
    return str(getattr(provider, "cache_namespace", provider.name))
