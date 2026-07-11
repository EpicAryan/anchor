from __future__ import annotations

import abc
import time

import requests


class ProviderError(Exception):
    pass


class LLMProvider(abc.ABC):
    name: str
    is_cloud: bool

    @abc.abstractmethod
    def generate(self, prompt: str) -> str: ...


def post_json(url: str, headers: dict, payload: dict,
              timeout: int = 60, retries: int = 2) -> dict:
    """POST JSON with 429 backoff. Error messages include the response body
    (truncated) but never request headers — that is where the API key lives."""
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=timeout)
        except requests.RequestException as exc:
            raise ProviderError(f"request to {url} failed: "
                                f"{type(exc).__name__}") from exc
        if resp.status_code == 429 and attempt < retries:
            time.sleep(2 ** (attempt + 1))
            continue
        if resp.status_code == 429:
            raise ProviderError(f"rate limited by {url} after {retries} retries")
        if resp.status_code != 200:
            raise ProviderError(
                f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")
        return resp.json()
    raise ProviderError("unreachable")


def get_provider(name: str) -> LLMProvider:
    from anchor.providers.gemini import GeminiProvider
    from anchor.providers.groq import GroqProvider
    from anchor.providers.ollama import OllamaProvider
    from anchor.providers.openrouter import OpenRouterProvider

    registry = {"ollama": OllamaProvider, "gemini": GeminiProvider,
                "groq": GroqProvider, "openrouter": OpenRouterProvider}
    if name not in registry:
        raise ProviderError(
            f"unknown provider {name!r}; expected one of {sorted(registry)}")
    return registry[name]()
