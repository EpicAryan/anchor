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
