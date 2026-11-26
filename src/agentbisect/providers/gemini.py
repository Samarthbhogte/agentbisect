"""Google Gemini, reached through its OpenAI-compatible Chat Completions endpoint.

Requests and responses use the OpenAI chat format, so this reuses `OpenAIProvider` with a
different base URL and key. Two differences matter: the endpoint rejects `seed` (HTTP 400), and
trial results are cached under their own namespace so they never mix with OpenAI's.
"""

from __future__ import annotations

import os
from typing import Any

from agentbisect.providers.openai import OpenAIProvider

PROVIDER = "gemini"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY_ENV = "GEMINI_API_KEY"
_HOST = "generativelanguage.googleapis.com"


def is_gemini_base_url(base_url: Any) -> bool:
    """Whether an OpenAI client's base URL points at Gemini's compatible endpoint."""
    return _HOST in str(base_url or "")


class GeminiProvider(OpenAIProvider):
    """Sends normalized requests to Gemini (async OpenAI client, Gemini base URL)."""

    name = PROVIDER
    supports_seed = False

    @property
    def client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                base_url=BASE_URL, api_key=os.environ.get(API_KEY_ENV), max_retries=0
            )
        return self._client
