from __future__ import annotations

import os

from anchor.providers import LLMProvider, ProviderError, post_json


class OllamaProvider(LLMProvider):
    name = "ollama"
    is_cloud = False

    def __init__(self):
        self.base_url = os.environ.get("ANCHOR_OLLAMA_URL",
                                       "http://localhost:11434")
        self.model = os.environ.get("ANCHOR_OLLAMA_MODEL", "qwen2.5:7b")

    def generate(self, prompt: str) -> str:
        try:
            data = post_json(
                f"{self.base_url}/api/generate", headers={},
                payload={"model": self.model, "prompt": prompt,
                         "stream": False},
                timeout=300)
        except ProviderError as exc:
            raise ProviderError(
                f"Ollama unavailable ({exc}). Install/start Ollama, or use a "
                f"cloud provider with --cloud.") from exc
        return data.get("response", "").strip()
