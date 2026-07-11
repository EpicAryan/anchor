from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from anchor.chunking import chunk_text
from anchor.config import Config
from anchor.db import MetadataDB
from anchor.embedder import Embedder
from anchor.ocr import IMAGE_EXTENSIONS, extract_text_from_image
from anchor.vectorstore import VectorStore


class Indexer:
    def __init__(self, db: MetadataDB, embedder: Embedder,
                 store: VectorStore, config: Config):
        self.db = db
        self.embedder = embedder
        self.store = store
        self.config = config

    def index_file(self, path: Path) -> str:
        path = path.resolve()
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return "unsupported"

        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        existing = self.db.get_document(str(path))
        if existing and existing[1] == content_hash:
            return "unchanged"

        text = extract_text_from_image(path)
        doc_id = self.db.upsert_document("screenshot", str(path), content_hash)

        chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
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
            [{"document_id": doc_id, "source_type": "screenshot",
              "source_path": str(path)} for _ in chunks],
        )
        return "indexed"
