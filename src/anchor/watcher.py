from __future__ import annotations

import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from anchor.ocr import IMAGE_EXTENSIONS

MAX_FILE_BYTES = 20_000_000


class ScreenshotHandler(FileSystemEventHandler):
    def __init__(self, indexer, watch_dir: Path):
        self.indexer = indexer
        self.watch_dir = watch_dir.resolve()

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_index(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self._maybe_index(Path(event.src_path))

    def _maybe_index(self, path: Path) -> str | None:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return None
        if path.is_symlink():
            return None
        resolved = path.resolve()
        if not resolved.is_relative_to(self.watch_dir):
            return None
        try:
            if resolved.stat().st_size > MAX_FILE_BYTES:
                return None
            status = self.indexer.index_file(resolved)
        except Exception as exc:
            # Log the path and error class only — never file content.
            print(f"[anchor] failed to index {resolved}: "
                  f"{type(exc).__name__}", file=sys.stderr)
            return None
        if status == "indexed":
            print(f"[anchor] indexed {resolved}")
        return status


def run_watcher(watch_dir: Path, indexer) -> None:
    watch_dir = watch_dir.expanduser()
    if not watch_dir.is_dir():
        raise SystemExit(f"watch directory does not exist: {watch_dir}")
    handler = ScreenshotHandler(indexer, watch_dir)
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()
    print(f"[anchor] watching {watch_dir} (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
