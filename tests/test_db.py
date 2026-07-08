import stat

from anchor.db import MetadataDB


def test_upsert_and_get(tmp_path):
    db = MetadataDB(tmp_path / "test.db")
    doc_id = db.upsert_document("screenshot", "/pics/a.png", "hash1")
    assert db.get_document("/pics/a.png") == (doc_id, "hash1")
    assert db.get_document("/pics/missing.png") is None


def test_upsert_same_path_updates_hash_keeps_id(tmp_path):
    db = MetadataDB(tmp_path / "test.db")
    id1 = db.upsert_document("screenshot", "/pics/a.png", "hash1")
    id2 = db.upsert_document("screenshot", "/pics/a.png", "hash2")
    assert id1 == id2
    assert db.get_document("/pics/a.png") == (id1, "hash2")


def test_replace_chunks_returns_old_vector_ids(tmp_path):
    db = MetadataDB(tmp_path / "test.db")
    doc_id = db.upsert_document("screenshot", "/pics/a.png", "hash1")
    old = db.replace_chunks(doc_id, ["chunk one"], ["v1"])
    assert old == []
    old = db.replace_chunks(doc_id, ["chunk two", "chunk three"], ["v2", "v3"])
    assert old == ["v1"]


def test_db_file_mode_0600(tmp_path):
    path = tmp_path / "test.db"
    MetadataDB(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_sql_injection_in_path_is_inert(tmp_path):
    db = MetadataDB(tmp_path / "test.db")
    evil = "/pics/x'; DROP TABLE documents;--.png"
    doc_id = db.upsert_document("screenshot", evil, "h")
    assert db.get_document(evil) == (doc_id, "h")
