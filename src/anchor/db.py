from __future__ import annotations

import os
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    vector_id TEXT NOT NULL UNIQUE
);
"""


class MetadataDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the watcher indexes from watchdog's single
        # dispatch thread while the connection is created on the main thread.
        # Access stays serialized (one dispatch thread), so this is safe.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        os.chmod(path, 0o600)

    def get_document(self, source_path: str) -> tuple[int, str] | None:
        row = self.conn.execute(
            "SELECT id, content_hash FROM documents WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        return (row[0], row[1]) if row else None

    def upsert_document(self, source_type: str, source_path: str,
                        content_hash: str) -> int:
        self.conn.execute(
            """INSERT INTO documents (source_type, source_path, content_hash)
               VALUES (?, ?, ?)
               ON CONFLICT(source_path) DO UPDATE SET
                   content_hash = excluded.content_hash,
                   indexed_at = datetime('now')""",
            (source_type, source_path, content_hash),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM documents WHERE source_path = ?", (source_path,)
        ).fetchone()
        return row[0]

    def replace_chunks(self, document_id: int, texts: list[str],
                       vector_ids: list[str]) -> list[str]:
        old = [r[0] for r in self.conn.execute(
            "SELECT vector_id FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchall()]
        self.conn.execute(
            "DELETE FROM chunks WHERE document_id = ?", (document_id,))
        self.conn.executemany(
            "INSERT INTO chunks (document_id, text, vector_id) VALUES (?, ?, ?)",
            [(document_id, t, v) for t, v in zip(texts, vector_ids)],
        )
        self.conn.commit()
        return old

    def close(self) -> None:
        self.conn.close()
