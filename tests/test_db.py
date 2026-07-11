import stat
import threading

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


def test_db_usable_from_worker_thread(tmp_path):
    # watchdog delivers file events on a worker thread; the connection is
    # created on the main thread. Must not raise sqlite3.ProgrammingError.
    db = MetadataDB(tmp_path / "test.db")
    errors = []

    def work():
        try:
            doc_id = db.upsert_document("screenshot", "/pics/t.png", "h")
            db.replace_chunks(doc_id, ["text"], ["v1"])
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert errors == []


def test_delete_document_returns_vector_ids_and_cascades(tmp_path):
    db = MetadataDB(tmp_path / "test.db")
    doc_id = db.upsert_document("screenshot", "/pics/a.png", "hash1")
    db.replace_chunks(doc_id, ["one", "two"], ["v1", "v2"])
    vids = db.delete_document("/pics/a.png")
    assert sorted(vids) == ["v1", "v2"]
    assert db.get_document("/pics/a.png") is None
    remaining = db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert remaining == 0  # ON DELETE CASCADE cleaned the chunks


def test_delete_document_unknown_path(tmp_path):
    db = MetadataDB(tmp_path / "test.db")
    assert db.delete_document("/pics/never-existed.png") == []


def test_all_documents(tmp_path):
    db = MetadataDB(tmp_path / "test.db")
    db.upsert_document("screenshot", "/pics/a.png", "h1")
    db.upsert_document("screenshot", "/pics/b.png", "h2")
    assert sorted(db.all_documents()) == ["/pics/a.png", "/pics/b.png"]


def test_sql_injection_in_path_is_inert(tmp_path):
    db = MetadataDB(tmp_path / "test.db")
    evil = "/pics/x'; DROP TABLE documents;--.png"
    doc_id = db.upsert_document("screenshot", evil, "h")
    assert db.get_document(evil) == (doc_id, "h")
