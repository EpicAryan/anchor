import os
from pathlib import Path

from anchor.watcher import MAX_FILE_BYTES, ScreenshotHandler


class RecordingIndexer:
    def __init__(self):
        self.calls = []

    def index_file(self, path):
        self.calls.append(path)
        return "indexed"


def make_handler(tmp_path):
    watch_dir = tmp_path / "shots"
    watch_dir.mkdir()
    idx = RecordingIndexer()
    return ScreenshotHandler(idx, watch_dir), idx, watch_dir


def test_indexes_new_png(tmp_path):
    handler, idx, watch_dir = make_handler(tmp_path)
    p = watch_dir / "a.png"
    p.write_bytes(b"fake")
    assert handler._maybe_index(p) == "indexed"
    assert idx.calls == [p.resolve()]


def test_skips_symlink(tmp_path):
    handler, idx, watch_dir = make_handler(tmp_path)
    secret = tmp_path / "outside" / "secret.png"
    secret.parent.mkdir()
    secret.write_bytes(b"fake")
    link = watch_dir / "link.png"
    os.symlink(secret, link)
    assert handler._maybe_index(link) is None
    assert idx.calls == []


def test_skips_path_escaping_watch_dir(tmp_path):
    handler, idx, watch_dir = make_handler(tmp_path)
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(b"fake")
    assert handler._maybe_index(watch_dir / ".." / "elsewhere.png") is None
    assert idx.calls == []


def test_skips_oversized_file(tmp_path):
    handler, idx, watch_dir = make_handler(tmp_path)
    p = watch_dir / "huge.png"
    p.write_bytes(b"x")
    os.truncate(p, MAX_FILE_BYTES + 1)  # sparse file, no real disk usage
    assert handler._maybe_index(p) is None
    assert idx.calls == []


def test_skips_non_image(tmp_path):
    handler, idx, watch_dir = make_handler(tmp_path)
    p = watch_dir / "notes.txt"
    p.write_text("hi")
    assert handler._maybe_index(p) is None
    assert idx.calls == []


def test_indexer_exception_does_not_propagate(tmp_path):
    class ExplodingIndexer:
        def index_file(self, path):
            raise RuntimeError("corrupt image")

    watch_dir = tmp_path / "shots"
    watch_dir.mkdir()
    handler = ScreenshotHandler(ExplodingIndexer(), watch_dir)
    p = watch_dir / "bad.png"
    p.write_bytes(b"fake")
    assert handler._maybe_index(p) is None  # must not raise
