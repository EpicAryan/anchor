from pathlib import Path

import pytest
from PIL import Image

import anchor.indexer as indexer_mod
from anchor.config import Config
from anchor.db import MetadataDB
from anchor.embedder import Embedder
from anchor.indexer import Indexer
from anchor.vectorstore import VectorStore


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        import numpy as np
        return np.array([[float(len(t)) % 7, 1.0, 2.0] for t in texts])


@pytest.fixture
def indexer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indexer_mod, "extract_text_from_image",
        lambda path: "TypeError: module object is not callable in views.py")
    config = Config(data_dir=tmp_path / "data")
    db = MetadataDB(config.db_path)
    embedder = Embedder()
    embedder._model = FakeModel()
    store = VectorStore(config.vector_dir)
    return Indexer(db, embedder, store, config)


def make_png(tmp_path, name="shot.png", color="white"):
    p = tmp_path / name
    Image.new("RGB", (60, 30), color).save(p)
    return p


def test_index_new_screenshot(indexer, tmp_path):
    p = make_png(tmp_path)
    assert indexer.index_file(p) == "indexed"
    doc = indexer.db.get_document(str(p.resolve()))
    assert doc is not None
    hits = indexer.store.query(indexer.embedder.embed_query("TypeError"), top_k=1)
    assert "module object is not callable" in hits[0]["text"]
    assert hits[0]["metadata"]["source_type"] == "screenshot"


def test_reindex_unchanged_is_skipped(indexer, tmp_path):
    p = make_png(tmp_path)
    indexer.index_file(p)
    assert indexer.index_file(p) == "unchanged"


def test_changed_file_reindexed_without_stale_vectors(indexer, tmp_path):
    p = make_png(tmp_path)
    indexer.index_file(p)
    make_png(tmp_path, color="black")  # same path, new bytes
    assert indexer.index_file(p) == "indexed"
    hits = indexer.store.query(indexer.embedder.embed_query("TypeError"), top_k=10)
    assert len(hits) == 1  # old vectors purged, not duplicated


def test_unsupported_extension(indexer, tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    assert indexer.index_file(p) == "unsupported"


def test_empty_ocr_result(indexer, tmp_path, monkeypatch):
    monkeypatch.setattr(indexer_mod, "extract_text_from_image", lambda path: "")
    p = make_png(tmp_path)
    assert indexer.index_file(p) == "empty"


def test_remove_file_deletes_db_row_and_vectors(indexer, tmp_path):
    p = make_png(tmp_path)
    indexer.index_file(p)
    assert indexer.remove_file(p) == "removed"
    assert indexer.db.get_document(str(p.resolve())) is None
    hits = indexer.store.query(indexer.embedder.embed_query("TypeError"), top_k=5)
    assert hits == []


def test_remove_file_unknown_path(indexer, tmp_path):
    assert indexer.remove_file(tmp_path / "never-indexed.png") == "unknown"


def test_prune_removes_only_missing_files(indexer, tmp_path):
    p1 = make_png(tmp_path, name="gone.png")
    p2 = make_png(tmp_path, name="kept.png")
    indexer.index_file(p1)
    indexer.index_file(p2)
    p1.unlink()  # simulate the user deleting a screenshot
    removed = indexer.prune()
    assert removed == [str(p1.resolve())]
    assert indexer.db.get_document(str(p2.resolve())) is not None
    hits = indexer.store.query(indexer.embedder.embed_query("TypeError"), top_k=10)
    assert {h["metadata"]["source_path"] for h in hits} == {str(p2.resolve())}


def test_prune_scoped_to_directory(indexer, tmp_path):
    inside = make_png(tmp_path, name="inside.png")
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    outside = other_dir / "outside.png"
    outside.write_bytes(inside.read_bytes())
    indexer.index_file(inside)
    indexer.index_file(outside)
    inside.unlink()
    outside.unlink()
    removed = indexer.prune(under=tmp_path / "elsewhere")
    assert removed == [str(outside.resolve())]           # scoped: only elsewhere/
    assert indexer.db.get_document(str(inside.resolve())) is not None
