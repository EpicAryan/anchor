# Personal Knowledge OS — Phase 2 (Multi-Source Indexing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend anchor's indexing pipeline from screenshots-only to PDFs (digital + scanned), notes (markdown/text), and code files, across multiple recursively-watched folders, with secret-shaped files hard-blocked from the index.

**Architecture:** A new extractor registry (`extractors.py`) maps file extension → (source_type, extraction function); `Indexer.index_file()` routes through it and everything downstream (hash dedup, chunking, local embedding, ChromaDB+SQLite, deletion sync) is untouched Phase 1 code. A new recursive walker (`walker.py`) owns every ingestion gate (exclusion dirs, secret blocklist, symlinks, containment, size caps) and is shared by `anchor index` and the watcher. Config grows a `watch_dirs` list (back-compatible with `watch_dir`).

**Tech Stack:** Existing Phase 1 stack + `pypdf` (PDF text layer, pure Python) + `pdf2image` (page rendering for scanned-page OCR) + `poppler-utils` (system package, user installs via sudo).

**Spec:** `docs/superpowers/specs/2026-07-18-phase2-multi-source-design.md` (approved).

## Global Constraints

These apply to EVERY task. Treat them as part of each task's requirements.

- Python `>=3.11`. All pip installs go into the project venv ONLY: use `/home/amour/products/anchor/.venv/bin/pip`. Never global pip, never system Python.
- Sudo/apt commands are NEVER run by the implementer — print the exact command and ask the USER to run it manually.
- Run tests with `.venv/bin/python -m pytest` from the repo root `/home/amour/products/anchor`.
- All 79 existing Phase 1 tests must keep passing after every task. When a Phase 1 test's *premise* changes (e.g. `.txt` is now indexable), update that test in the same task and say so in the commit.
- **Privacy default:** `allow_cloud` defaults to `false`; embeddings are always local; redaction before any cloud egress (unchanged from Phase 1 — do not touch this code path).
- **Secrets never enter the index:** files matching `SECRET_FILE_PATTERNS` are blocked at BOTH the walker and the indexer (defense in depth).
- **No content in logs:** never print/log file text, OCR output, chunk text, or prompts. Paths and counts only.
- **SQL:** parameterized queries only. (No schema changes in this phase.)
- **Filesystem safety:** skip symlinks, paths resolving outside the root, files > 20 MB (50 MB for PDFs), excluded/hidden directories — at every depth.
- Git: commit after every task, message style `feat:`/`fix:`/`docs:`/`test:`. NEVER add a Co-Authored-By trailer.
- New runtime deps allowed in this phase: `pypdf`, `pdf2image`. Nothing else. No provider SDKs.

## File Structure

```
src/anchor/
├── extractors.py        # NEW — extension → (source_type, extract fn) registry
├── pdf.py               # NEW — pypdf text layer + per-page OCR fallback
├── walker.py            # NEW — recursive walk; ALL ingestion gates live here
├── config.py            # MODIFIED — watch_dirs list + legacy watch_dir compat
├── indexer.py           # MODIFIED — route via extractors; "blocked" status
├── watcher.py           # MODIFIED — multi-root, recursive, walker-gated events
├── query.py             # MODIFIED — --type override + richer type inference
├── cli.py               # MODIFIED — multi-root index/watch/prune; --type flag
└── (ocr.py, chunking.py, embedder.py, vectorstore.py, db.py,
     redact.py, providers/ — UNCHANGED)

tests/
├── test_extractors.py   # NEW
├── test_pdf.py          # NEW
├── test_walker.py       # NEW
├── test_config.py       # MODIFIED — watch_dirs cases added
├── test_indexer.py      # MODIFIED — routing/blocked cases; fixture updated
├── test_watcher.py      # MODIFIED — recursive/multi-type events; stale cases fixed
├── test_query.py        # MODIFIED — type inference + override cases
└── test_cli.py          # MODIFIED — new args parse cases
```

Dependency direction (no cycles): `walker → extractors → (ocr, pdf)`; `indexer → (extractors, walker)`; `watcher → walker`; `cli → walker`.

---

### Task 1: Dependencies (venv pip + user-run apt)

**Files:**
- Modify: `pyproject.toml:6-13`

**Interfaces:**
- Produces: importable `pypdf` and `pdf2image` inside `.venv`; `pdftoppm` binary on PATH (poppler).

- [ ] **Step 1: Add the two new runtime deps to pyproject.toml**

Change the `dependencies` list to:

```toml
dependencies = [
    "watchdog>=4",
    "pytesseract>=0.3.10",
    "Pillow>=10",
    "chromadb>=0.5",
    "sentence-transformers>=3",
    "requests>=2.31",
    "pypdf>=4",
    "pdf2image>=1.17",
]
```

- [ ] **Step 2: Install into the project venv ONLY**

Run: `/home/amour/products/anchor/.venv/bin/pip install -e ".[dev]"`
Expected: `Successfully installed ... pypdf-... pdf2image-...`

- [ ] **Step 3: Ask the USER to install poppler (do not run sudo yourself)**

Tell the user to run, in their own terminal:

```bash
sudo apt-get install -y poppler-utils
```

Wait for their confirmation before Step 4.

- [ ] **Step 4: Verify all three landed**

Run: `/home/amour/products/anchor/.venv/bin/python -c "import pypdf, pdf2image; print('py deps ok')" && pdftoppm -v 2>&1 | head -1`
Expected: `py deps ok` and a `pdftoppm version ...` line.

- [ ] **Step 5: Confirm the existing suite still passes**

Run: `.venv/bin/python -m pytest -q`
Expected: `79 passed`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pypdf and pdf2image for Phase 2 PDF extraction"
```

---

### Task 2: Extractor registry (images, notes, code)

`.pdf` joins the registry in Task 3, after `pdf.py` exists.

**Files:**
- Create: `src/anchor/extractors.py`
- Test: `tests/test_extractors.py`

**Interfaces:**
- Consumes: `anchor.ocr.IMAGE_EXTENSIONS`, `anchor.ocr.extract_text_from_image` (Phase 1).
- Produces: `EXTENSION_TYPES: dict[str, str]` (extension → source_type); `classify(path: Path) -> str | None`; `extract(path: Path) -> tuple[str, str]` returning `(source_type, text)`, raising `ValueError` on unsupported extensions. Tasks 4, 6, 8 rely on these exact names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_extractors.py`:

```python
from pathlib import Path

import pytest

import anchor.extractors as ex_mod
from anchor.extractors import EXTENSION_TYPES, classify, extract


def test_extension_types_cover_all_sources():
    assert EXTENSION_TYPES[".png"] == "screenshot"
    assert EXTENSION_TYPES[".md"] == "note"
    assert EXTENSION_TYPES[".txt"] == "note"
    assert EXTENSION_TYPES[".rst"] == "note"
    assert EXTENSION_TYPES[".py"] == "code"
    assert EXTENSION_TYPES[".yaml"] == "code"
    assert EXTENSION_TYPES[".json"] == "code"


def test_classify_is_case_insensitive_and_none_for_unknown(tmp_path):
    assert classify(Path("A.PNG")) == "screenshot"
    assert classify(Path("b.Md")) == "note"
    assert classify(Path("c.zip")) is None
    assert classify(Path("no_extension")) is None


def test_extract_note_reads_text(tmp_path):
    p = tmp_path / "todo.md"
    p.write_text("# Deployment checklist\n- rotate keys\n")
    source_type, text = extract(p)
    assert source_type == "note"
    assert "Deployment checklist" in text


def test_extract_code_reads_text(tmp_path):
    p = tmp_path / "app.py"
    p.write_text("RETRY_BACKOFF = 2  # seconds\n")
    source_type, text = extract(p)
    assert source_type == "code"
    assert "RETRY_BACKOFF" in text


def test_extract_handles_non_utf8_bytes(tmp_path):
    p = tmp_path / "weird.txt"
    p.write_bytes(b"caf\xff latte notes")
    source_type, text = extract(p)   # must not raise UnicodeDecodeError
    assert source_type == "note"
    assert "latte notes" in text


def test_extract_image_routes_to_ocr(tmp_path, monkeypatch):
    monkeypatch.setattr(ex_mod, "extract_text_from_image",
                        lambda path: "ocr says hi")
    p = tmp_path / "shot.png"
    p.write_bytes(b"not a real png; ocr is mocked")
    assert extract(p) == ("screenshot", "ocr says hi")


def test_extract_unsupported_raises(tmp_path):
    with pytest.raises(ValueError):
        extract(tmp_path / "archive.zip")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_extractors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anchor.extractors'`

- [ ] **Step 3: Write the implementation**

Create `src/anchor/extractors.py`:

```python
from __future__ import annotations

from pathlib import Path

from anchor.ocr import IMAGE_EXTENSIONS, extract_text_from_image

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
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _extract_image(path: Path) -> str:
    return extract_text_from_image(path)


_EXTRACTORS = {
    "screenshot": _extract_image,
    "note": _read_text,
    "code": _read_text,
}


def classify(path: Path) -> str | None:
    """source_type for this file, or None if anchor doesn't index it."""
    return EXTENSION_TYPES.get(path.suffix.lower())


def extract(path: Path) -> tuple[str, str]:
    source_type = classify(path)
    if source_type is None:
        raise ValueError(f"unsupported file type: {path.suffix}")
    return source_type, _EXTRACTORS[source_type](path)
```

(`_extract_image` is a wrapper so tests can monkeypatch `extract_text_from_image` on this module and have it take effect.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_extractors.py -v`
Expected: 7 passed

- [ ] **Step 5: Full suite, then commit**

Run: `.venv/bin/python -m pytest -q`
Expected: 86 passed

```bash
git add src/anchor/extractors.py tests/test_extractors.py
git commit -m "feat: extractor registry mapping extension to source type"
```

---

### Task 3: PDF extraction (text layer + per-page OCR fallback)

**Files:**
- Create: `src/anchor/pdf.py`
- Modify: `src/anchor/extractors.py` (register `.pdf`)
- Test: `tests/test_pdf.py`

**Interfaces:**
- Consumes: `pypdf.PdfReader`, `pdf2image.convert_from_path`, `pytesseract`.
- Produces: `extract_text_from_pdf(path: Path) -> str`; constants `MIN_TEXT_CHARS_PER_PAGE = 30`, `MAX_OCR_PAGES = 50`. After this task `classify(Path("x.pdf")) == "pdf"` and `extract()` handles PDFs.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pdf.py`:

```python
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import anchor.pdf as pdf_mod
from anchor.extractors import classify, extract
from anchor.pdf import extract_text_from_pdf


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakeReader:
    is_encrypted = False

    def __init__(self, pages):
        self.pages = pages


def test_digital_pdf_uses_text_layer_no_ocr(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_mod, "PdfReader",
                        lambda p: FakeReader([FakePage("A" * 40),
                                              FakePage("B" * 40)]))
    monkeypatch.setattr(pdf_mod, "_ocr_page",
                        lambda *a: pytest.fail("OCR must not run"))
    text = extract_text_from_pdf(tmp_path / "digital.pdf")
    assert "A" * 40 in text and "B" * 40 in text


def test_textless_page_falls_back_to_ocr(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_mod, "PdfReader",
                        lambda p: FakeReader([FakePage("")]))
    monkeypatch.setattr(pdf_mod, "_ocr_page",
                        lambda path, n: "scanned receipt total 42")
    assert extract_text_from_pdf(tmp_path / "scan.pdf") == \
        "scanned receipt total 42"


def test_ocr_page_cap(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pdf_mod, "MAX_OCR_PAGES", 2)
    monkeypatch.setattr(pdf_mod, "PdfReader",
                        lambda p: FakeReader([FakePage("")] * 5))
    calls = []
    monkeypatch.setattr(pdf_mod, "_ocr_page",
                        lambda path, n: calls.append(n) or f"page {n}")
    text = extract_text_from_pdf(tmp_path / "book.pdf")
    assert calls == [1, 2]                      # capped after 2 OCR'd pages
    assert "page 1" in text and "page 2" in text
    assert "OCR page cap" in capsys.readouterr().err


def test_encrypted_pdf_raises(monkeypatch, tmp_path):
    reader = FakeReader([])
    reader.is_encrypted = True
    monkeypatch.setattr(pdf_mod, "PdfReader", lambda p: reader)
    with pytest.raises(ValueError):
        extract_text_from_pdf(tmp_path / "locked.pdf")


def test_pdf_registered_in_extractors(monkeypatch, tmp_path):
    assert classify(Path("report.PDF")) == "pdf"
    monkeypatch.setattr("anchor.extractors.extract_text_from_pdf",
                        lambda path: "quarterly report body")
    p = tmp_path / "report.pdf"
    p.write_bytes(b"mocked anyway")
    assert extract(p) == ("pdf", "quarterly report body")


def test_scanned_pdf_end_to_end_with_real_poppler(tmp_path, monkeypatch):
    """PIL saves an image-only PDF (no text layer) -> pypdf finds no text ->
    page renders through REAL poppler -> tesseract call is mocked."""
    img = Image.new("RGB", (400, 120), "white")
    ImageDraw.Draw(img).text((20, 40), "SCANNED CONTENT", fill="black")
    p = tmp_path / "scan.pdf"
    img.save(p, "PDF")
    monkeypatch.setattr(pdf_mod.pytesseract, "image_to_string",
                        lambda image: "SCANNED CONTENT")
    assert extract_text_from_pdf(p) == "SCANNED CONTENT"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pdf.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anchor.pdf'`

- [ ] **Step 3: Write the implementation**

Create `src/anchor/pdf.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytesseract
from pypdf import PdfReader

MIN_TEXT_CHARS_PER_PAGE = 30
MAX_OCR_PAGES = 50


def _ocr_page(path: Path, page_number: int) -> str:
    """Render one page to an image (poppler) and OCR it (tesseract)."""
    from pdf2image import convert_from_path
    images = convert_from_path(str(path), dpi=200,
                               first_page=page_number, last_page=page_number)
    return pytesseract.image_to_string(images[0]).strip() if images else ""


def extract_text_from_pdf(path: Path) -> str:
    """Text layer per page; pages with almost no text are treated as scanned
    and OCR'd, up to MAX_OCR_PAGES per file."""
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError("encrypted PDF")
    pages: list[str] = []
    ocr_used = 0
    skipped = 0
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < MIN_TEXT_CHARS_PER_PAGE:
            if ocr_used < MAX_OCR_PAGES:
                ocr_used += 1
                text = _ocr_page(path, number)
            else:
                skipped += 1
                text = ""
        if text:
            pages.append(text)
    if skipped:
        # Path and count only — never content.
        print(f"[anchor] OCR page cap reached for {path} "
              f"({skipped} pages skipped)", file=sys.stderr)
    return "\n\n".join(pages).strip()
```

Then register it in `src/anchor/extractors.py` — add the import and the two registry entries:

```python
from anchor.pdf import extract_text_from_pdf
```

```python
EXTENSION_TYPES: dict[str, str] = {
    **{ext: "screenshot" for ext in IMAGE_EXTENSIONS},
    **{ext: "note" for ext in NOTE_EXTENSIONS},
    **{ext: "code" for ext in CODE_EXTENSIONS},
    ".pdf": "pdf",
}
```

```python
_EXTRACTORS = {
    "screenshot": _extract_image,
    "note": _read_text,
    "code": _read_text,
    "pdf": lambda path: extract_text_from_pdf(path),
}
```

(The lambda indirection lets tests monkeypatch `anchor.extractors.extract_text_from_pdf`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pdf.py tests/test_extractors.py -v`
Expected: all pass (13 tests). If `test_scanned_pdf_end_to_end_with_real_poppler` fails with `PDFInfoNotInstalledError`, poppler is missing — re-do Task 1 Step 3 with the user.

- [ ] **Step 5: Full suite, then commit**

Run: `.venv/bin/python -m pytest -q`
Expected: 92 passed

```bash
git add src/anchor/pdf.py src/anchor/extractors.py tests/test_pdf.py
git commit -m "feat: PDF extraction — text layer with per-page OCR fallback"
```

---

### Task 4: Recursive walker with every ingestion gate

**Files:**
- Create: `src/anchor/walker.py`
- Test: `tests/test_walker.py`

**Interfaces:**
- Consumes: `anchor.extractors.classify`.
- Produces (Tasks 6, 8, 9 rely on these exact names):
  - `MAX_FILE_BYTES = 20_000_000`, `MAX_PDF_BYTES = 50_000_000`
  - `EXCLUDED_DIR_NAMES: frozenset[str]`, `SECRET_FILE_PATTERNS: tuple[str, ...]`
  - `is_secret_file(path: Path) -> bool`
  - `passes_static_gates(path: Path, root: Path) -> bool` — gates that work on a *deleted* path (extension, secret, containment, excluded ancestors)
  - `is_indexable(path: Path, root: Path) -> bool` — static gates + symlink + size
  - `iter_files(root: Path) -> Iterator[Path]` — deterministic recursive walk

- [ ] **Step 1: Write the failing tests**

Create `tests/test_walker.py`:

```python
import os

from anchor.walker import (MAX_FILE_BYTES, is_indexable, is_secret_file,
                           iter_files, passes_static_gates)


def make_tree(tmp_path):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "node_modules" / "lib").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "docs").mkdir()
    (root / "notes.md").write_text("top note")
    (root / "src" / "app.py").write_text("print('hi')")
    (root / "docs" / "guide.txt").write_text("nested note")
    (root / "node_modules" / "lib" / "dep.js").write_text("junk")
    (root / ".git" / "config.ini").write_text("git internals")
    (root / ".env").write_text("API_KEY=supersecret")
    (root / "server.pem").write_text("---BEGIN---")
    (root / "readme.docx").write_text("unsupported type")
    return root


def test_iter_files_recurses_and_excludes(tmp_path):
    root = make_tree(tmp_path)
    names = {p.name for p in iter_files(root)}
    assert names == {"notes.md", "app.py", "guide.txt"}


def test_iter_files_is_deterministic(tmp_path):
    root = make_tree(tmp_path)
    assert list(iter_files(root)) == list(iter_files(root))


def test_secret_files_blocked(tmp_path):
    root = make_tree(tmp_path)
    assert is_secret_file(root / ".env")
    assert is_secret_file(root / ".env.production")
    assert is_secret_file(root / "server.pem")
    assert is_secret_file(root / "id_rsa")
    assert is_secret_file(root / "deploy.key")
    assert is_secret_file(root / "credentials.json")
    assert is_secret_file(root / "SECRETS.yaml")      # case-insensitive
    assert not is_secret_file(root / "notes.md")
    assert not is_secret_file(root / "app.py")


def test_symlink_not_indexable(tmp_path):
    root = make_tree(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("outside content")
    link = root / "link.md"
    link.symlink_to(outside)
    assert not is_indexable(link, root)


def test_path_escaping_root_not_indexable(tmp_path):
    root = make_tree(tmp_path)
    outside = tmp_path / "elsewhere.md"
    outside.write_text("x")
    assert not is_indexable(root / ".." / "elsewhere.md", root)


def test_oversized_file_not_indexable(tmp_path):
    root = make_tree(tmp_path)
    big = root / "big.md"
    big.write_text("x")
    os.truncate(big, MAX_FILE_BYTES + 1)     # sparse file, no real disk usage
    assert not is_indexable(big, root)


def test_pdf_gets_larger_cap(tmp_path):
    root = make_tree(tmp_path)
    pdf = root / "scan.pdf"
    pdf.write_bytes(b"%PDF-")
    os.truncate(pdf, MAX_FILE_BYTES + 1)     # over text cap, under PDF cap
    assert is_indexable(pdf, root)


def test_static_gates_work_for_deleted_paths(tmp_path):
    root = make_tree(tmp_path)
    assert passes_static_gates(root / "gone.md", root)          # never existed
    assert not passes_static_gates(root / "gone.env", root)
    assert not passes_static_gates(root / "node_modules" / "x.js", root)
    assert not passes_static_gates(tmp_path / "outside.md", root)


def test_excluded_dir_file_not_indexable_even_directly(tmp_path):
    root = make_tree(tmp_path)
    assert not is_indexable(root / "node_modules" / "lib" / "dep.js", root)
    assert not is_indexable(root / ".git" / "config.ini", root)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_walker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anchor.walker'`

- [ ] **Step 3: Write the implementation**

Create `src/anchor/walker.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_walker.py -v`
Expected: 10 passed

- [ ] **Step 5: Full suite, then commit**

Run: `.venv/bin/python -m pytest -q`
Expected: 102 passed

```bash
git add src/anchor/walker.py tests/test_walker.py
git commit -m "feat: recursive walker with exclusions, secret blocklist, size caps"
```

---

### Task 5: Config — multiple watch roots with back-compat

**Files:**
- Modify: `src/anchor/config.py`
- Test: `tests/test_config.py` (add cases; existing six tests must not change)

**Interfaces:**
- Produces: `Config.watch_dirs: list[Path]` (dataclass field, default `[Path.home() / "Screenshots"]`); `Config.watch_dir` becomes a read-only property returning `watch_dirs[0]` (so all Phase 1 call sites keep working); `load_config()` understands both the legacy `"watch_dir"` key and the new `"watch_dirs"` list in config.json.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_watch_dirs_defaults_to_single_screenshots_dir(tmp_path):
    cfg = load_config(tmp_path / "d")
    assert cfg.watch_dirs == [Path.home() / "Screenshots"]
    assert cfg.watch_dir == cfg.watch_dirs[0]       # back-compat property


def test_watch_dirs_list_parsed_from_config(tmp_path):
    data_dir = tmp_path / "d"
    data_dir.mkdir(mode=0o700)
    (data_dir / "config.json").write_text(
        '{"watch_dirs": ["/tmp/shots", "/tmp/notes"]}')
    cfg = load_config(data_dir)
    assert cfg.watch_dirs == [Path("/tmp/shots"), Path("/tmp/notes")]


def test_legacy_watch_dir_key_becomes_watch_dirs(tmp_path):
    data_dir = tmp_path / "d"
    data_dir.mkdir(mode=0o700)
    (data_dir / "config.json").write_text('{"watch_dir": "/tmp/old"}')
    cfg = load_config(data_dir)
    assert cfg.watch_dirs == [Path("/tmp/old")]
    assert cfg.watch_dir == Path("/tmp/old")


def test_legacy_watch_dir_prepended_when_both_present(tmp_path):
    data_dir = tmp_path / "d"
    data_dir.mkdir(mode=0o700)
    (data_dir / "config.json").write_text(
        '{"watch_dir": "/tmp/old", "watch_dirs": ["/tmp/new"]}')
    cfg = load_config(data_dir)
    assert cfg.watch_dirs == [Path("/tmp/old"), Path("/tmp/new")]
```

Also add the missing import at the top of the file if not present: `from pathlib import Path`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: the 4 new tests FAIL (`AttributeError: ... no attribute 'watch_dirs'`); the 6 existing tests PASS.

- [ ] **Step 3: Write the implementation**

In `src/anchor/config.py`, change the imports line and the dataclass:

```python
from dataclasses import dataclass, field, fields
```

```python
@dataclass
class Config:
    data_dir: Path = DEFAULT_DATA_DIR
    watch_dirs: list[Path] = field(
        default_factory=lambda: [Path.home() / "Screenshots"])
    allow_cloud: bool = False
    cloud_provider: str = "gemini"
    top_k: int = 5
    chunk_size: int = 1500
    chunk_overlap: int = 200

    @property
    def watch_dir(self) -> Path:
        """Back-compat: Phase 1 call sites read the (first) watch folder."""
        return self.watch_dirs[0]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "anchor.db"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "chroma"
```

And replace the config.json overlay block inside `load_config()` with:

```python
    cfg = Config(data_dir=data_dir)
    config_file = data_dir / "config.json"
    if config_file.exists():
        raw = json.loads(config_file.read_text())
        legacy = raw.pop("watch_dir", None)
        valid = {f.name for f in fields(Config)}
        for key, value in raw.items():
            if key not in valid or key == "data_dir":
                continue
            if key == "watch_dirs":
                value = [Path(v).expanduser() for v in value]
            setattr(cfg, key, value)
        if legacy is not None:
            legacy_path = Path(legacy).expanduser()
            if "watch_dirs" not in raw:
                cfg.watch_dirs = [legacy_path]
            elif legacy_path not in cfg.watch_dirs:
                cfg.watch_dirs.insert(0, legacy_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: 10 passed

- [ ] **Step 5: Full suite, then commit**

Run: `.venv/bin/python -m pytest -q`
Expected: 106 passed (the user's real `~/.anchor/config.json` still has `watch_dir` — the legacy path keeps it working; nothing to migrate).

```bash
git add src/anchor/config.py tests/test_config.py
git commit -m "feat: multi-root watch_dirs config with legacy watch_dir compat"
```

---

### Task 6: Indexer — route through extractors, block secrets

**Files:**
- Modify: `src/anchor/indexer.py`
- Test: `tests/test_indexer.py` (fixture + 2 stale tests updated, 3 tests added)

**Interfaces:**
- Consumes: `extractors.classify/extract`, `walker.is_secret_file`.
- Produces: `Indexer.index_file()` statuses now include `"blocked"`; stored `source_type` comes from the registry. `remove_file`/`prune` unchanged.

- [ ] **Step 1: Update the fixture and stale tests, add failing tests**

In `tests/test_indexer.py`, the fixture monkeypatches `extract_text_from_image` on the indexer module — after this task the indexer imports `extract` instead. Replace the fixture:

```python
@pytest.fixture
def indexer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indexer_mod, "extract",
        lambda path: ("screenshot",
                      "TypeError: module object is not callable in views.py"))
    config = Config(data_dir=tmp_path / "data")
    db = MetadataDB(config.db_path)
    embedder = Embedder()
    embedder._model = FakeModel()
    store = VectorStore(config.vector_dir)
    return Indexer(db, embedder, store, config)
```

Replace `test_unsupported_extension` (`.txt` is now a supported note type):

```python
def test_unsupported_extension(indexer, tmp_path):
    p = tmp_path / "archive.zip"
    p.write_text("binary-ish")
    assert indexer.index_file(p) == "unsupported"
```

Replace `test_empty_ocr_result`'s monkeypatch line to match the new import:

```python
def test_empty_ocr_result(indexer, tmp_path, monkeypatch):
    monkeypatch.setattr(indexer_mod, "extract",
                        lambda path: ("screenshot", ""))
    p = make_png(tmp_path)
    assert indexer.index_file(p) == "empty"
```

Add three new tests at the end of the file:

```python
def test_note_indexed_with_note_source_type(tmp_path, monkeypatch):
    # Real extractor path (no mock): a markdown file becomes a "note" doc.
    config = Config(data_dir=tmp_path / "data")
    db = MetadataDB(config.db_path)
    embedder = Embedder()
    embedder._model = FakeModel()
    store = VectorStore(config.vector_dir)
    idx = Indexer(db, embedder, store, config)
    p = tmp_path / "deploy.md"
    p.write_text("# Deploy checklist: rotate the api gateway keys")
    assert idx.index_file(p) == "indexed"
    hits = store.query(embedder.embed_query("deploy"), top_k=1)
    assert hits[0]["metadata"]["source_type"] == "note"
    assert hits[0]["metadata"]["source_path"] == str(p.resolve())


def test_secret_file_blocked_and_never_stored(indexer, tmp_path):
    p = tmp_path / ".env"
    p.write_text("GROQ_API_KEY=gsk_realkey")
    assert indexer.index_file(p) == "blocked"
    assert indexer.db.get_document(str(p.resolve())) is None
    assert indexer.store.query(
        indexer.embedder.embed_query("GROQ_API_KEY"), top_k=5) == []


def test_secret_file_blocked_before_unsupported_check(indexer, tmp_path):
    # .pem isn't even a supported extension, but "blocked" must win so a
    # future extension addition can't accidentally open a secrets hole.
    p = tmp_path / "server.pem"
    p.write_text("---BEGIN PRIVATE KEY---")
    assert indexer.index_file(p) == "blocked"
```

- [ ] **Step 2: Run tests to verify the new/changed ones fail**

Run: `.venv/bin/python -m pytest tests/test_indexer.py -v`
Expected: FAIL — fixture's `monkeypatch.setattr(indexer_mod, "extract", ...)` errors with `AttributeError` (module has no `extract` yet).

- [ ] **Step 3: Write the implementation**

In `src/anchor/indexer.py`, replace the ocr import with:

```python
from anchor.extractors import classify, extract
from anchor.walker import is_secret_file
```

and replace the top of `index_file` (through the `text = ...` / `doc_id = ...` lines) with:

```python
    def index_file(self, path: Path) -> str:
        path = path.resolve()
        if is_secret_file(path):
            return "blocked"
        if classify(path) is None:
            return "unsupported"

        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        existing = self.db.get_document(str(path))
        if existing and existing[1] == content_hash:
            return "unchanged"

        source_type, text = extract(path)
        doc_id = self.db.upsert_document(source_type, str(path), content_hash)
```

and in the `store.add(...)` metadata list, replace the hardcoded `"screenshot"`:

```python
            [{"document_id": doc_id, "source_type": source_type,
              "source_path": str(path)} for _ in chunks],
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_indexer.py -v`
Expected: 12 passed

- [ ] **Step 5: Full suite, then commit**

Run: `.venv/bin/python -m pytest -q`
Expected: 109 passed

```bash
git add src/anchor/indexer.py tests/test_indexer.py
git commit -m "feat: indexer routes all source types via extractor registry, blocks secrets"
```

---

### Task 7: Query — type inference for all sources + explicit override

**Files:**
- Modify: `src/anchor/query.py`
- Test: `tests/test_query.py` (add cases)

**Interfaces:**
- Produces: `infer_source_type(question) -> str | None` recognizing all four types; `find_matches(..., source_type: str | None = None)` and `answer_question(..., source_type: str | None = None)` — explicit `source_type` overrides inference. Task 9's CLI passes `args.type` straight in.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_query.py` (it already defines `FakeEmbedder`, `FakeStore`, and imports `find_matches`, `infer_source_type`):

```python
def test_infer_source_type_all_types():
    assert infer_source_type("show me the screenshot of that error") == "screenshot"
    assert infer_source_type("which images mention the dashboard") == "screenshot"
    assert infer_source_type("that pdf about invoices") == "pdf"
    assert infer_source_type("search my pdfs for the contract") == "pdf"
    assert infer_source_type("the document I saved about taxes") == "pdf"
    assert infer_source_type("my note on deployment") == "note"
    assert infer_source_type("notes about standup") == "note"
    assert infer_source_type("the code that retries requests") == "code"
    assert infer_source_type("that script for backups") == "code"
    assert infer_source_type("what did I save about invoices") is None


def test_infer_source_type_matches_words_not_substrings():
    assert infer_source_type("the encoded value in the response") is None
    assert infer_source_type("the postscript at the end") is None


def test_explicit_source_type_overrides_inference():
    store = FakeStore([])
    find_matches("screenshot of the invoice", config=Config(),
                 embedder=FakeEmbedder(), store=store, source_type="pdf")
    assert store.last_source_type == "pdf"


def test_source_type_none_falls_back_to_inference():
    store = FakeStore([])
    find_matches("screenshot of the invoice", config=Config(),
                 embedder=FakeEmbedder(), store=store)
    assert store.last_source_type == "screenshot"
```

(`FakeStore(hits)` takes the hits list and records `last_source_type` — both already defined in this test file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_query.py -v`
Expected: new tests FAIL (`infer_source_type(...pdf...)` returns None; `find_matches` got unexpected keyword `source_type`).

- [ ] **Step 3: Write the implementation**

In `src/anchor/query.py`, add `import re` to the imports, then replace `infer_source_type`:

```python
_TYPE_KEYWORDS = {
    "screenshot": "screenshot", "image": "screenshot",
    "pdf": "pdf", "document": "pdf",
    "note": "note",
    "code": "code", "script": "code",
}


def infer_source_type(question: str) -> str | None:
    for word in re.findall(r"[a-z]+", question.lower()):
        singular = word[:-1] if word.endswith("s") else word
        for candidate in (word, singular):
            if candidate in _TYPE_KEYWORDS:
                return _TYPE_KEYWORDS[candidate]
    return None
```

Then thread the override through both entry points — change the signatures and the `store.query` calls:

```python
def find_matches(question: str, *, config: Config, embedder: Embedder,
                 store: VectorStore, source_type: str | None = None) -> list[dict]:
```

```python
    hits = store.query(embedder.embed_query(question),
                       top_k=config.top_k,
                       source_type=source_type or infer_source_type(question))
```

```python
def answer_question(question: str, *, config: Config, embedder: Embedder,
                    store: VectorStore, provider: LLMProvider,
                    source_type: str | None = None) -> Answer:
```

```python
    hits = store.query(embedder.embed_query(question),
                       top_k=config.top_k,
                       source_type=source_type or infer_source_type(question))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_query.py -v`
Expected: all pass (existing + 4 new)

- [ ] **Step 5: Full suite, then commit**

Run: `.venv/bin/python -m pytest -q`
Expected: 113 passed

```bash
git add src/anchor/query.py tests/test_query.py
git commit -m "feat: source-type inference for pdf/note/code plus explicit override"
```

---

### Task 8: Watcher — multi-root, recursive, walker-gated

**Files:**
- Modify: `src/anchor/watcher.py`
- Test: `tests/test_watcher.py` (2 stale tests updated, 4 added; handler renamed)

**Interfaces:**
- Consumes: `walker.is_indexable`, `walker.passes_static_gates`, `walker.MAX_FILE_BYTES` (re-exported for the existing test import).
- Produces: `WatchHandler(indexer, watch_dir)` (renamed from `ScreenshotHandler` — it handles every type now); `run_watcher(roots: list[Path], indexer)` taking a LIST. Task 9's CLI calls `run_watcher(roots, indexer)`.

- [ ] **Step 1: Update stale tests, add failing tests**

In `tests/test_watcher.py`:

Change the import line to:

```python
from anchor.watcher import MAX_FILE_BYTES, WatchHandler, _should_poll
```

and every `ScreenshotHandler(` call to `WatchHandler(`.

Replace `test_skips_non_image` (a `.txt` is now indexable — the premise changed):

```python
def test_skips_unsupported_extension(tmp_path):
    handler, idx, watch_dir = make_handler(tmp_path)
    p = watch_dir / "archive.zip"
    p.write_text("not indexable")
    assert handler._maybe_index(p) is None
    assert idx.indexed == []
```

Replace `test_deleted_non_image_ignored`:

```python
def test_deleted_unsupported_extension_ignored(tmp_path):
    handler, idx, watch_dir = make_handler(tmp_path)
    assert handler._maybe_remove(watch_dir / "archive.zip") is None
    assert idx.removed == []
```

Add four new tests:

```python
def test_indexes_nested_note(tmp_path):
    handler, idx, watch_dir = make_handler(tmp_path)
    nested = watch_dir / "topics" / "deploy"
    nested.mkdir(parents=True)
    p = nested / "checklist.md"
    p.write_text("rotate keys")
    assert handler._maybe_index(p) == "indexed"
    assert idx.indexed == [p.resolve()]


def test_skips_file_in_excluded_dir(tmp_path):
    handler, idx, watch_dir = make_handler(tmp_path)
    dep = watch_dir / "node_modules" / "lib"
    dep.mkdir(parents=True)
    p = dep / "dep.js"
    p.write_text("junk")
    assert handler._maybe_index(p) is None
    assert idx.indexed == []


def test_secret_file_event_never_indexed(tmp_path):
    handler, idx, watch_dir = make_handler(tmp_path)
    p = watch_dir / ".env"
    p.write_text("API_KEY=oops")
    assert handler._maybe_index(p) is None
    assert idx.indexed == []


def test_deleted_secret_file_ignored(tmp_path):
    handler, idx, watch_dir = make_handler(tmp_path)
    assert handler._maybe_remove(watch_dir / ".env") is None
    assert idx.removed == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_watcher.py -v`
Expected: FAIL — `ImportError: cannot import name 'WatchHandler'`

- [ ] **Step 3: Write the implementation**

Replace `src/anchor/watcher.py` in full:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_watcher.py -v`
Expected: 15 passed

- [ ] **Step 5: Full suite, then commit**

Run: `.venv/bin/python -m pytest -q`
Expected: 117 passed

```bash
git add src/anchor/watcher.py tests/test_watcher.py
git commit -m "feat: watcher handles all source types across multiple recursive roots"
```

---

### Task 9: CLI — multi-root index/watch/prune, --type filter

**Files:**
- Modify: `src/anchor/cli.py`
- Test: `tests/test_cli.py` (extend the parser test, add wiring tests)

**Interfaces:**
- Consumes: `walker.iter_files`, `run_watcher(roots, indexer)`, `find_matches(..., source_type=)`, `answer_question(..., source_type=)`, `config.watch_dirs`.
- Produces: final CLI surface — `anchor index [path]`, `anchor watch [path]` (`--dir` kept as hidden legacy alias), `anchor prune [path]`, `anchor find q [--type T] [-k N]`, `anchor ask q [--type T] [--cloud|--local] [-k N]`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_cli.py`, extend `test_parser_commands` — replace its body with:

```python
def test_parser_commands():
    parser = build_parser()
    args = parser.parse_args(["ask", "where is that error?", "--cloud"])
    assert args.command == "ask"
    assert args.question == "where is that error?"
    assert args.cloud is True
    args = parser.parse_args(["index"])
    assert args.command == "index" and args.path is None
    args = parser.parse_args(["index", "/tmp/anywhere"])
    assert args.path == "/tmp/anywhere"
    args = parser.parse_args(["watch"])
    assert args.command == "watch" and args.path is None
    args = parser.parse_args(["watch", "/tmp/adhoc"])
    assert args.path == "/tmp/adhoc"
    args = parser.parse_args(["prune", "/tmp/scope"])
    assert args.command == "prune" and args.path == "/tmp/scope"
    args = parser.parse_args(["prune"])
    assert args.path is None
    args = parser.parse_args(["find", "dashboard", "--type", "pdf"])
    assert args.command == "find" and args.type == "pdf"
    args = parser.parse_args(["ask", "q", "--type", "note", "--local"])
    assert args.type == "note" and args.local
    args = parser.parse_args(["ask", "q"])
    assert args.type is None
```

Add an invalid-type test and an index-wiring test:

```python
def test_type_flag_rejects_unknown_values():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["find", "q", "--type", "spreadsheet"])


def test_index_command_walks_all_configured_roots(tmp_path, monkeypatch):
    import anchor.cli as cli_mod
    root_a = tmp_path / "shots"; root_a.mkdir()
    root_b = tmp_path / "notes"; root_b.mkdir()
    (root_b / "sub").mkdir()
    (root_b / "sub" / "n.md").write_text("hello")

    class FakeIndexer:
        def __init__(self):
            self.paths = []
        def index_file(self, p):
            self.paths.append(p)
            return "indexed"
        def prune(self, under=None):
            return []

    fake = FakeIndexer()
    cfg = Config(data_dir=tmp_path / "data",
                 watch_dirs=[root_a, root_b])
    monkeypatch.setattr(cli_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(cli_mod, "_make_indexer", lambda c: fake)
    assert cli_mod.main(["index"]) == 0
    assert fake.paths == [(root_b / "sub" / "n.md")]
```

Add the imports the new tests need at the top of the file: `import pytest` and (if missing) `from anchor.config import Config`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'path'` (watch), `--type` unrecognized.

- [ ] **Step 3: Write the implementation**

In `src/anchor/cli.py`:

Replace the ocr import with the walker import:

```python
from anchor.walker import iter_files
```

(delete `from anchor.ocr import IMAGE_EXTENSIONS` — no longer used.)

In `build_parser()`, replace the `watch`, `prune`, `find`, and `ask` sub-parsers:

```python
    p_watch = sub.add_parser("watch", help="watch folders for changes")
    p_watch.add_argument("path", nargs="?", default=None,
                         help="folder to watch (default: all configured "
                              "watch folders)")
    p_watch.add_argument("--dir", default=None, help=argparse.SUPPRESS)

    p_prune = sub.add_parser(
        "prune", help="remove index entries for files deleted from disk")
    p_prune.add_argument("path", nargs="?", default=None,
                         help="only prune entries under this folder")

    p_find = sub.add_parser(
        "find", help="list files matching a phrase (no LLM, fully local)")
    p_find.add_argument("question")
    p_find.add_argument("--type", choices=("screenshot", "pdf", "note", "code"),
                        default=None, help="only search this source type")
    p_find.add_argument("-k", type=int, default=None,
                        help="max results (default 10)")

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--type", choices=("screenshot", "pdf", "note", "code"),
                       default=None, help="only search this source type")
    p_ask.add_argument("--cloud", action="store_true",
                       help="one-shot consent to send redacted snippets "
                            "to the configured cloud provider")
    p_ask.add_argument("--local", action="store_true",
                       help="force local LLM even if allow_cloud is true")
    p_ask.add_argument("-k", type=int, default=None, help="top-k chunks")
```

(also update `p_index`'s help text: `help="file or folder (default: all configured watch folders)"`.)

Replace the `index`, `prune`, and `watch` command blocks in `main()`:

```python
    if args.command == "index":
        indexer = _make_indexer(config)
        targets = ([Path(args.path).expanduser()] if args.path
                   else list(config.watch_dirs))
        printed = 0
        for target in targets:
            files = [target] if target.is_file() else list(iter_files(target))
            for f in files:
                try:
                    status = indexer.index_file(f)
                except Exception as exc:
                    status = f"error:{type(exc).__name__}"
                print(f"{status:>12}  {f}")
                printed += 1
            if target.is_dir():
                for p in indexer.prune(under=target):
                    print(f"{'removed':>12}  {p}")
                    printed += 1
        if not printed:
            print("nothing to index", file=sys.stderr)
            return 1
        return 0

    if args.command == "prune":
        under = Path(args.path).expanduser() if args.path else None
        removed = _make_indexer(config).prune(under=under)
        if not removed:
            print("nothing to prune — index matches disk")
        for p in removed:
            print(f"{'removed':>12}  {p}")
        return 0

    if args.command == "watch":
        chosen = args.path or args.dir
        roots = ([Path(chosen).expanduser()] if chosen
                 else list(config.watch_dirs))
        run_watcher(roots, _make_indexer(config))
        return 0
```

And pass the type filter through in `find` and `ask`:

```python
        matches = find_matches(
            args.question, config=config, embedder=Embedder(),
            store=VectorStore(config.vector_dir), source_type=args.type)
```

```python
        ans = answer_question(
            args.question, config=config, embedder=Embedder(),
            store=VectorStore(config.vector_dir), provider=provider,
            source_type=args.type)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: 7 passed

- [ ] **Step 5: Full suite, then commit**

Run: `.venv/bin/python -m pytest -q`
Expected: 120 passed

```bash
git add src/anchor/cli.py tests/test_cli.py
git commit -m "feat: CLI multi-root index/watch/prune and --type filter"
```

---

### Task 10: Documentation + final suite

**Files:**
- Modify: `README.md`
- Modify: `GUIDE.md` (LOCAL ONLY — gitignored, never commit it)

**Interfaces:** none (docs).

- [ ] **Step 1: Update README.md**

In Setup, change the apt line to:

```bash
sudo apt-get install -y tesseract-ocr poppler-utils
```

Replace the Usage section:

```markdown
## Usage

```bash
anchor index                        # sync ALL watch folders (adds new, removes deleted)
anchor index ~/any/folder           # index any folder on demand (recursive)
anchor watch                        # live: indexes new files, forgets deleted ones
anchor ask "screenshot of the module object is not callable error"
anchor ask "what does my note say about the deploy checklist"
anchor find "dashboard" --type pdf  # list matching PDFs (no LLM, offline)
anchor prune                        # drop index entries for deleted files
```

Indexes screenshots (OCR), PDFs (digital + scanned via OCR), notes
(`.md .txt .rst`), and code/config files. Folders are walked recursively;
`.git`, `node_modules`, virtualenvs, hidden and build directories are
skipped, and secret-shaped files (`.env*`, `*.pem`, `*.key`, `id_rsa*`,
`credentials*`, …) are never indexed.

Config lives in `~/.anchor/config.json`:

```json
{
  "watch_dirs": [
    "/mnt/c/Users/YOU/Pictures/Screenshots",
    "/mnt/c/Users/YOU/Documents/pdfs",
    "/mnt/c/Users/YOU/Documents/notes"
  ],
  "allow_cloud": true,
  "cloud_provider": "gemini"
}
```

(The old single `"watch_dir"` key still works.)
```

Add one line to the Security model section's filesystem bullet: watcher and
indexer skip symlinks, paths outside watched roots, oversized files
(20 MB; 50 MB for PDFs), excluded directories — and secret-shaped files are
blocked from the index entirely (redaction only guards cloud egress).

- [ ] **Step 2: Update GUIDE.md (local only)**

Update the table (watch folders now plural), the "Daily usage" section
(mention PDFs/notes/code folders, `--type` flag, recursive behavior,
`anchor index <path>` for ad-hoc folders), and Troubleshooting (add: scanned
PDF slow to index → OCR runs per page, capped at 50 pages;
`PDFInfoNotInstalledError` → `sudo apt-get install -y poppler-utils`).
Do NOT `git add` this file.

- [ ] **Step 3: Full suite one more time**

Run: `.venv/bin/python -m pytest -q`
Expected: 120 passed

- [ ] **Step 4: Commit (README only)**

```bash
git add README.md
git commit -m "docs: Phase 2 usage — multi-folder, PDFs/notes/code, --type filter"
```

---

### Task 11: Live verification on real folders

No new code — prove the whole phase works against reality, mirroring Phase 1's live checks. Uses the scratchpad for throwaway fixtures.

- [ ] **Step 1: Build a real mixed-content test folder**

```bash
S=/tmp/claude-1000/-home-amour-products-anchor/*/scratchpad/phase2-live
mkdir -p "$S"/project/src "$S"/project/node_modules
printf '# Deploy checklist\nRotate the gateway keys every quarter.\n' > "$S"/notes.md
printf 'RETRY_BACKOFF = [2, 4]  # seconds between provider retries\n' > "$S"/project/src/retry.py
printf 'SHOULD_NEVER_APPEAR=1\n' > "$S"/project/.env
printf 'junk\n' > "$S"/project/node_modules/dep.js
.venv/bin/python - <<'EOF'
from pathlib import Path
from PIL import Image, ImageDraw
import glob
s = Path(glob.glob("/tmp/claude-1000/-home-amour-products-anchor/*/scratchpad/phase2-live")[0])
img = Image.new("RGB", (500, 150), "white")
ImageDraw.Draw(img).text((20, 50), "INVOICE TOTAL 4242 EUR", fill="black")
img.save(s / "scan.pdf", "PDF")
EOF
```

- [ ] **Step 2: Index it ad-hoc and check every gate**

Run: `.venv/bin/anchor index "$S"` (expand `$S` to the real path).
Expected output lines: `indexed` for `notes.md`, `retry.py`, `scan.pdf`; NO line for `.env` or `node_modules/dep.js` (walker never yields them).

Then verify the index directly:

```bash
.venv/bin/python - <<'EOF'
import sqlite3, pathlib
db = sqlite3.connect(pathlib.Path.home() / ".anchor" / "anchor.db")
rows = db.execute("SELECT source_type, source_path FROM documents "
                  "WHERE source_path LIKE '%phase2-live%'").fetchall()
for r in rows: print(r)
assert not any(".env" in r[1] or "node_modules" in r[1] for r in rows)
assert {r[0] for r in rows} == {"note", "code", "pdf"}
print("gates OK")
EOF
```

- [ ] **Step 3: Query across types**

```bash
.venv/bin/anchor find "gateway keys" --type note     # notes.md listed
.venv/bin/anchor find "retry backoff"                # retry.py listed
.venv/bin/anchor find "invoice total" --type pdf     # scan.pdf listed (OCR'd)
.venv/bin/anchor ask "what is the invoice total in my pdf?"   # Groq answer citing scan.pdf
.venv/bin/anchor find "SHOULD_NEVER_APPEAR"          # expected: no matches
```

- [ ] **Step 4: Deletion sync still holds for new types**

```bash
rm "$S"/notes.md
.venv/bin/anchor index "$S"        # prints: removed .../notes.md
.venv/bin/anchor find "gateway keys"   # notes.md no longer listed
```

- [ ] **Step 5: Live watch on a real folder (polling + recursion)**

Start `.venv/bin/anchor watch "$S"` in the background redirected to a log;
create `"$S"/sub/new-note.md`, wait ~5s, confirm the log shows
`[anchor] indexed .../sub/new-note.md`; delete it, confirm
`[anchor] removed ...`; stop the watcher.

- [ ] **Step 6: Offer the user real watch_dirs**

Ask the user which Windows folders to add (e.g.
`C:\Users\ARYAN KUMAR\Documents\anchor-pdfs` and `...\anchor-notes`), create
them if needed, then update `~/.anchor/config.json` to the `watch_dirs`
list form including the existing screenshots folder. Clean up the
`phase2-live` scratch folder and prune its entries:

```bash
rm -rf "$S" && .venv/bin/anchor prune
```

- [ ] **Step 7: Final commit if any fixes were needed, then report**

If live verification exposed fixes, they were committed as `fix:` commits
during this task. Report results to the user with the verified outputs.

---

## Self-Review (done at plan-writing time)

- **Spec coverage:** FR1→Task 2/3, FR2→Task 3, FR3→Task 4, FR4→Task 5,
  FR5→Task 6, FR6→Task 8, FR7→Task 9 (+Task 7 for query internals),
  FR8→Tasks 6/8/9 reuse Phase 1 deletion paths; §7 security→Tasks 4/6/8
  tests; §10 success criteria→Task 11; §11 rollout→Tasks 1/10/11.
- **Type consistency:** `extract() -> tuple[str, str]` consumed in Task 6;
  `source_type: str | None = None` kwarg consistent across Task 7/9;
  `run_watcher(roots: list[Path], indexer)` consistent Task 8/9;
  `WatchHandler` renamed once, tests updated in the same task.
- **Test-count checkpoints** (79→86→92→102→106→109→113→117→120) are
  expectations, not gates — if a count differs because a task added an extra
  test, update the expectation and continue; a FAILURE is always a stop.
