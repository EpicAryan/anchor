from __future__ import annotations

import os

from anchor.providers import LLMProvider, ProviderError, post_json


class GroqProvider(LLMProvider):
    name = "groq"
    is_cloud = True
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "GROQ_API_KEY is not set. Put it in ~/.anchor/env "
                "(chmod 600) or export it.")

    def generate(self, prompt: str) -> str:
        data = post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload={"model": self.MODEL,
                     "messages": [{"role": "user", "content": prompt}]})
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise ProviderError("unexpected Groq response shape") from exc
