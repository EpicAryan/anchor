from anchor.vectorstore import VectorStore


def make_store(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    store.add(
        vector_ids=["v1", "v2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        texts=["django migration error", "react query blog post"],
        metadatas=[
            {"document_id": 1, "source_type": "screenshot", "source_path": "/pics/a.png"},
            {"document_id": 2, "source_type": "note", "source_path": "/notes/b.md"},
        ],
    )
    return store


def test_query_returns_nearest_first(tmp_path):
    store = make_store(tmp_path)
    hits = store.query([1.0, 0.05], top_k=2)
    assert hits[0]["vector_id"] == "v1"
    assert hits[0]["text"] == "django migration error"
    assert hits[0]["metadata"]["source_path"] == "/pics/a.png"
    assert len(hits) == 2


def test_query_filters_by_source_type(tmp_path):
    store = make_store(tmp_path)
    hits = store.query([0.5, 0.5], top_k=5, source_type="screenshot")
    assert [h["vector_id"] for h in hits] == ["v1"]


def test_query_empty_store_returns_empty(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    assert store.query([1.0, 0.0], top_k=5) == []


def test_delete(tmp_path):
    store = make_store(tmp_path)
    store.delete(["v1"])
    hits = store.query([1.0, 0.0], top_k=5)
    assert [h["vector_id"] for h in hits] == ["v2"]
    store.delete([])  # must not raise
