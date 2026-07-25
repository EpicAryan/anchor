from __future__ import annotations

import hashlib
import threading
import uuid
from pathlib import Path

from anchor.chunking import chunk_text
from anchor.config import Config
from anchor.db import MetadataDB
from anchor.embedder import Embedder
from anchor.extractors import classify, extract
from anchor.vectorstore import VectorStore
from anchor.walker import is_secret_file


class Indexer:
    def __init__(self, db: MetadataDB, embedder: Embedder,
                 store: VectorStore, config: Config):
        self.db = db
        self.embedder = embedder
        self.store = store
        self.config = config
        # The watcher shares one Indexer across a polling and an inotify
        # observer thread (mixed /mnt + native roots), and all three writes
        # below touch a single SQLite connection, Chroma client, and embed
        # model — none guaranteed thread-safe. Serialize every mutation.
        self._lock = threading.Lock()

    def index_file(self, path: Path) -> str:
        path = path.resolve()
        if is_secret_file(path):
            return "blocked"
        if classify(path) is None:
            return "unsupported"

        with self._lock:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            existing = self.db.get_document(str(path))
            if existing and existing[1] == content_hash:
                return "unchanged"

            source_type, text = extract(path)
            doc_id = self.db.upsert_document(source_type, str(path), content_hash)

            chunks = chunk_text(text, self.config.chunk_size,
                                self.config.chunk_overlap)
            if not chunks:
                old = self.db.replace_chunks(doc_id, [], [])
                self.store.delete(old)
                return "empty"

            vector_ids = [str(uuid.uuid4()) for _ in chunks]
            embeddings = self.embedder.embed_texts(chunks)
            old = self.db.replace_chunks(doc_id, chunks, vector_ids)
            self.store.delete(old)
            self.store.add(
                vector_ids, embeddings, chunks,
                [{"document_id": doc_id, "source_type": source_type,
                  "source_path": str(path)} for _ in chunks],
            )
            return "indexed"

    def remove_file(self, path: Path) -> str:
        """Forget a file: delete its document row, chunks, and vectors."""
        path = path.resolve()
        with self._lock:
            if self.db.get_document(str(path)) is None:
                return "unknown"
            self.store.delete(self.db.delete_document(str(path)))
            return "removed"

    def prune(self, under: Path | None = None) -> list[str]:
        """Remove index entries whose files no longer exist on disk.
        `under` limits the sweep to one directory."""
        prefix = f"{under.resolve()}/" if under else None
        removed = []
        with self._lock:
            for source_path in self.db.all_documents():
                if prefix and not source_path.startswith(prefix):
                    continue
                if Path(source_path).exists():
                    continue
                self.store.delete(self.db.delete_document(source_path))
                removed.append(source_path)
        return removed
