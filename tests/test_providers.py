import pytest

import anchor.providers as providers_mod
from anchor.providers import LLMProvider, ProviderError, get_provider, post_json


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


def test_get_provider_gemini_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        get_provider("gemini")


def test_get_provider_unknown():
    with pytest.raises(ProviderError, match="unknown provider"):
        get_provider("openai")


def test_flags(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert get_provider("ollama").is_cloud is False
    assert get_provider("gemini").is_cloud is True
    assert get_provider("groq").is_cloud is True


def test_post_json_retries_on_429_then_succeeds(monkeypatch):
    responses = [FakeResponse(429, {}), FakeResponse(200, {"ok": True})]
    monkeypatch.setattr(providers_mod.requests, "post",
                        lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: None)
    assert post_json("http://x", {}, {}) == {"ok": True}


def test_post_json_raises_after_exhausted_retries(monkeypatch):
    monkeypatch.setattr(providers_mod.requests, "post",
                        lambda *a, **k: FakeResponse(429, {}))
    monkeypatch.setattr(providers_mod.time, "sleep", lambda s: None)
    with pytest.raises(ProviderError, match="rate limited"):
        post_json("http://x", {}, {})


def test_post_json_error_does_not_leak_headers(monkeypatch):
    monkeypatch.setattr(providers_mod.requests, "post",
                        lambda *a, **k: FakeResponse(500, "boom"))
    with pytest.raises(ProviderError) as exc_info:
        post_json("http://x", {"x-goog-api-key": "SUPERSECRET"}, {})
    assert "SUPERSECRET" not in str(exc_info.value)


def test_gemini_generate_parses_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"], captured["headers"] = url, headers
        return FakeResponse(200, {"candidates": [
            {"content": {"parts": [{"text": "the answer"}]}}]})

    monkeypatch.setattr(providers_mod.requests, "post", fake_post)
    assert get_provider("gemini").generate("q?") == "the answer"
    assert "key=" not in captured["url"]          # key travels in header, not URL
    assert captured["headers"]["x-goog-api-key"] == "k"


def test_groq_generate_parses_response(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(
        providers_mod.requests, "post",
        lambda url, headers=None, json=None, timeout=None: FakeResponse(
            200, {"choices": [{"message": {"content": "groq says"}}]}))
    assert get_provider("groq").generate("q?") == "groq says"


def test_ollama_connection_refused_becomes_provider_error(monkeypatch):
    import requests as real_requests

    def refuse(*a, **k):
        raise real_requests.ConnectionError("refused")

    monkeypatch.setattr(providers_mod.requests, "post", refuse)
    with pytest.raises(ProviderError, match="Ollama"):
        get_provider("ollama").generate("q?")
