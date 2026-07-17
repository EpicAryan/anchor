from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterator
from pathlib import Path

from anchor.extractors import classify

MAX_FILE_BYTES = 20_000_000
MAX_PDF_BYTES = 50_000_000          # scanned PDFs run large

EXCLUDED_DIR_NAMES = frozenset({
    "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", "target", ".next",
})

# Files that must NEVER enter the index: redaction only guards cloud
# egress, so secrets are blocked at ingestion instead.
SECRET_FILE_PATTERNS = (
    ".env*", "*.pem", "*.key", "id_rsa*", "id_ed25519*",
    "*.p12", "*.pfx", "credentials*", "secrets.*",
)


def is_secret_file(path: Path) -> bool:
    name = path.name.lower()
    return any(fnmatch.fnmatch(name, pat) for pat in SECRET_FILE_PATTERNS)


def _excluded_dir(name: str) -> bool:
    return name.startswith(".") or name in EXCLUDED_DIR_NAMES


def _max_bytes(path: Path) -> int:
    return MAX_PDF_BYTES if path.suffix.lower() == ".pdf" else MAX_FILE_BYTES


def passes_static_gates(path: Path, root: Path) -> bool:
    """Gates that don't require the file to exist (usable for delete events):
    supported extension, not secret-shaped, resolves inside root, and no
    excluded/hidden directory between root and the file."""
    if classify(path) is None or is_secret_file(path):
        return False
    resolved = path.resolve()
    root = root.resolve()
    if not resolved.is_relative_to(root):
        return False
    ancestors = resolved.relative_to(root).parts[:-1]
    return not any(_excluded_dir(part) for part in ancestors)


def is_indexable(path: Path, root: Path) -> bool:
    if path.is_symlink() or not passes_static_gates(path, root):
        return False
    try:
        return path.resolve().stat().st_size <= _max_bytes(path)
    except OSError:
        return False


def iter_files(root: Path) -> Iterator[Path]:
    """Deterministic recursive walk of one root, applying every gate."""
    root = root.expanduser().resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not _excluded_dir(d))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if is_indexable(path, root):
                yield path
