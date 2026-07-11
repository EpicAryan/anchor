from __future__ import annotations

import os

from anchor.providers import LLMProvider, ProviderError, post_json


class GeminiProvider(LLMProvider):
    name = "gemini"
    is_cloud = True
    MODEL = "gemini-2.0-flash"

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set. Put it in ~/.anchor/env "
                "(chmod 600) or export it.")

    def generate(self, prompt: str) -> str:
        # Key goes in a header, NOT the URL: URLs end up in shell history,
        # proxy logs, and tracebacks.
        data = post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.MODEL}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            payload={"contents": [{"parts": [{"text": prompt}]}]})
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise ProviderError(
                "unexpected Gemini response shape (possibly a safety block)"
            ) from exc
