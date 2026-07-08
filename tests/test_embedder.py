from anchor.embedder import Embedder


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        import numpy as np
        return np.array([[float(len(t)), 1.0, 2.0] for t in texts])


def test_embed_texts_returns_plain_lists():
    e = Embedder()
    e._model = FakeModel()
    vectors = e.embed_texts(["ab", "abcd"])
    assert vectors == [[2.0, 1.0, 2.0], [4.0, 1.0, 2.0]]
    assert isinstance(vectors[0], list)


def test_embed_query_returns_single_vector():
    e = Embedder()
    e._model = FakeModel()
    assert e.embed_query("abc") == [3.0, 1.0, 2.0]


def test_model_is_lazy():
    e = Embedder()
    assert e._model is None  # constructing must not download/load anything
