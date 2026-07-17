import pytest

from anchor.config import Config
from anchor.providers import LLMProvider, ProviderError
from anchor.query import (Answer, answer_question, find_matches,
                          infer_source_type)


class FakeEmbedder:
    def embed_query(self, text):
        return [1.0, 0.0]


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.last_source_type = "UNSET"

    def query(self, embedding, top_k=5, source_type=None):
        self.last_source_type = source_type
        return self.hits


class CapturingProvider(LLMProvider):
    name = "fake"

    def __init__(self, is_cloud, reply="synthesized answer"):
        self.is_cloud = is_cloud
        self.reply = reply
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply


class FailingProvider(LLMProvider):
    name, is_cloud = "fake", False

    def generate(self, prompt):
        raise ProviderError("rate limited")


HITS = [
    {"vector_id": "v1",
     "text": "django migrate failed AKIAIOSFODNN7EXAMPLE fix: --fake",
     "metadata": {"source_path": "/pics/a.png", "source_type": "screenshot"},
     "distance": 0.1},
    {"vector_id": "v2", "text": "unrelated react notes",
     "metadata": {"source_path": "/pics/b.png", "source_type": "screenshot"},
     "distance": 0.4},
]


def run(provider, hits=HITS, question="where did I fix django migrate?"):
    return answer_question(
        question, config=Config(), embedder=FakeEmbedder(),
        store=FakeStore(hits), provider=provider)


def test_infer_source_type():
    assert infer_source_type("show me the screenshot of that error") == "screenshot"
    assert infer_source_type("what did I read in June?") is None


def test_answer_includes_sources():
    ans = run(CapturingProvider(is_cloud=False))
    assert ans.text == "synthesized answer"
    assert ans.sources == ["/pics/a.png", "/pics/b.png"]


def test_cloud_provider_gets_redacted_chunks():
    provider = CapturingProvider(is_cloud=True)
    run(provider)
    assert "AKIAIOSFODNN7EXAMPLE" not in provider.prompts[0]
    assert "[REDACTED:aws-access-key]" in provider.prompts[0]


def test_local_provider_gets_raw_chunks():
    provider = CapturingProvider(is_cloud=False)
    run(provider)
    assert "AKIAIOSFODNN7EXAMPLE" in provider.prompts[0]


def test_prompt_marks_chunks_untrusted():
    provider = CapturingProvider(is_cloud=False)
    run(provider)
    prompt = provider.prompts[0]
    assert "untrusted" in prompt.lower()
    assert "<context>" in prompt and "</context>" in prompt


def test_provider_failure_falls_back_to_extractive():
    ans = run(FailingProvider())
    assert "django migrate failed" in ans.text     # raw snippet shown
    assert "/pics/a.png" in ans.text
    assert ans.sources == ["/pics/a.png", "/pics/b.png"]


def test_no_hits():
    ans = run(CapturingProvider(is_cloud=False), hits=[])
    assert ans.sources == []
    assert "no indexed content" in ans.text.lower()


def test_find_matches_returns_best_chunk_per_file():
    # two chunks from the same file: only the closest one is kept
    hits = HITS + [
        {"vector_id": "v3", "text": "another chunk from the same screenshot",
         "metadata": {"source_path": "/pics/a.png", "source_type": "screenshot"},
         "distance": 0.9},
    ]
    matches = find_matches("django", config=Config(), embedder=FakeEmbedder(),
                           store=FakeStore(hits))
    assert [m["path"] for m in matches] == ["/pics/a.png", "/pics/b.png"]
    assert "django migrate failed" in matches[0]["snippet"]
    assert len(matches[0]["snippet"]) <= 150


def test_find_matches_empty_store():
    matches = find_matches("anything", config=Config(),
                           embedder=FakeEmbedder(), store=FakeStore([]))
    assert matches == []


def test_infer_source_type_all_types():
    assert infer_source_type("show me the screenshot of that error") == "screenshot"
    assert infer_source_type("which images mention the dashboard") == "screenshot"
    assert infer_source_type("that pdf about invoices") == "pdf"
    assert infer_source_type("search my pdfs for the contract") == "pdf"
    assert infer_source_type("the document I saved about taxes") == "pdf"
    assert infer_source_type("my note on deployment") == "note"
    assert infer_source_type("notes about standup") == "note"
    assert infer_source_type("the code that retries requests") == "code"
    assert infer_source_type("that script for backups") == "code"
    assert infer_source_type("what did I save about invoices") is None


def test_infer_source_type_matches_words_not_substrings():
    assert infer_source_type("the encoded value in the response") is None
    assert infer_source_type("the postscript at the end") is None


def test_explicit_source_type_overrides_inference():
    store = FakeStore([])
    find_matches("screenshot of the invoice", config=Config(),
                 embedder=FakeEmbedder(), store=store, source_type="pdf")
    assert store.last_source_type == "pdf"


def test_source_type_none_falls_back_to_inference():
    store = FakeStore([])
    find_matches("screenshot of the invoice", config=Config(),
                 embedder=FakeEmbedder(), store=store)
    assert store.last_source_type == "screenshot"
