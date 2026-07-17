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
