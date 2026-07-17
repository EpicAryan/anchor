from __future__ import annotations

from pathlib import Path

from anchor.ocr import IMAGE_EXTENSIONS, extract_text_from_image
from anchor.pdf import extract_text_from_pdf

NOTE_EXTENSIONS = {".md", ".txt", ".rst"}
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".go", ".rs", ".rb", ".php", ".sh", ".sql", ".html", ".css",
    ".json", ".yaml", ".yml", ".toml", ".ini",
}

EXTENSION_TYPES: dict[str, str] = {
    **{ext: "screenshot" for ext in IMAGE_EXTENSIONS},
    **{ext: "note" for ext in NOTE_EXTENSIONS},
    **{ext: "code" for ext in CODE_EXTENSIONS},
    ".pdf": "pdf",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _extract_image(path: Path) -> str:
    return extract_text_from_image(path)


_EXTRACTORS = {
    "screenshot": _extract_image,
    "note": _read_text,
    "code": _read_text,
    "pdf": lambda path: extract_text_from_pdf(path),
}


def classify(path: Path) -> str | None:
    """source_type for this file, or None if anchor doesn't index it."""
    return EXTENSION_TYPES.get(path.suffix.lower())


def extract(path: Path) -> tuple[str, str]:
    source_type = classify(path)
    if source_type is None:
        raise ValueError(f"unsupported file type: {path.suffix}")
    return source_type, _EXTRACTORS[source_type](path)
