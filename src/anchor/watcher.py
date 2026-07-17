from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from anchor.walker import (MAX_FILE_BYTES, is_indexable,  # noqa: F401
                           passes_static_gates)

# MAX_FILE_BYTES is re-exported: Phase 1 tests import it from here.


def _should_poll(watch_dir: Path) -> bool:
    """inotify events don't fire for Windows drives (/mnt/*) under WSL2,
    so fall back to polling there. ANCHOR_FORCE_POLLING=1 forces it anywhere
    (e.g. network mounts)."""
    if os.environ.get("ANCHOR_FORCE_POLLING") == "1":
        return True
    return str(watch_dir).startswith("/mnt/")


class WatchHandler(FileSystemEventHandler):
    """One handler per watched root; every gate delegates to walker so the
    watcher and `anchor index` can never disagree about what's indexable."""

    def __init__(self, indexer, watch_dir: Path):
        self.indexer = indexer
        self.watch_dir = watch_dir.resolve()

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_index(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self._maybe_index(Path(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory:
            self._maybe_remove(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            self._maybe_remove(Path(event.src_path))
            self._maybe_index(Path(event.dest_path))

    def _maybe_remove(self, path: Path) -> str | None:
        if not passes_static_gates(path, self.watch_dir):
            return None
        resolved = path.resolve()
        try:
            status = self.indexer.remove_file(resolved)
        except Exception as exc:
            print(f"[anchor] failed to remove {resolved}: "
                  f"{type(exc).__name__}", file=sys.stderr)
            return None
        if status == "removed":
            print(f"[anchor] removed {resolved}", flush=True)
        return status

    def _maybe_index(self, path: Path) -> str | None:
        if not is_indexable(path, self.watch_dir):
            return None
        resolved = path.resolve()
        try:
            status = self.indexer.index_file(resolved)
        except Exception as exc:
            # Log the path and error class only — never file content.
            print(f"[anchor] failed to index {resolved}: "
                  f"{type(exc).__name__}", file=sys.stderr)
            return None
        if status == "indexed":
            print(f"[anchor] indexed {resolved}", flush=True)
        return status


def run_watcher(roots: list[Path], indexer) -> None:
    roots = [r.expanduser() for r in roots]
    missing = [r for r in roots if not r.is_dir()]
    if missing:
        raise SystemExit(
            f"watch directories do not exist: {', '.join(map(str, missing))}")

    observers: dict[str, object] = {}
    for root in roots:
        polling = _should_poll(root)
        backend = "polling" if polling else "inotify"
        if backend not in observers:
            observers[backend] = (PollingObserver(timeout=2) if polling
                                  else Observer())
        observers[backend].schedule(
            WatchHandler(indexer, root), str(root), recursive=True)
        mode = "polling every 2s" if polling else "inotify"
        print(f"[anchor] watching {root} ({mode})", flush=True)

    for obs in observers.values():
        obs.start()
    print("[anchor] Ctrl-C to stop", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for obs in observers.values():
            obs.stop()
    for obs in observers.values():
        obs.join()
