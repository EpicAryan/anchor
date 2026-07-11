from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from anchor.ocr import IMAGE_EXTENSIONS

MAX_FILE_BYTES = 20_000_000


def _should_poll(watch_dir: Path) -> bool:
    """inotify events don't fire for Windows drives (/mnt/*) under WSL2,
    so fall back to polling there. ANCHOR_FORCE_POLLING=1 forces it anywhere
    (e.g. network mounts)."""
    if os.environ.get("ANCHOR_FORCE_POLLING") == "1":
        return True
    return str(watch_dir).startswith("/mnt/")


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
    polling = _should_poll(watch_dir)
    observer = PollingObserver(timeout=2) if polling else Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()
    mode = "polling every 2s" if polling else "inotify"
    print(f"[anchor] watching {watch_dir} ({mode}, Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
