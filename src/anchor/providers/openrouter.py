from __future__ import annotations

import os

from anchor.providers import LLMProvider, ProviderError, post_json


class OpenRouterProvider(LLMProvider):
    name = "openrouter"
    is_cloud = True
    # A :free-tier model so no credit is ever consumed by default.
    # Override with ANCHOR_OPENROUTER_MODEL to use any OpenRouter model id.
    DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "OPENROUTER_API_KEY is not set. Put it in ~/.anchor/env "
                "(chmod 600) or export it.")
        self.model = os.environ.get("ANCHOR_OPENROUTER_MODEL",
                                    self.DEFAULT_MODEL)

    def generate(self, prompt: str) -> str:
        data = post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload={"model": self.model,
                     "messages": [{"role": "user", "content": prompt}]})
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise ProviderError("unexpected OpenRouter response shape") from exc
