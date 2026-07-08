from pathlib import Path

from PIL import Image

import anchor.ocr as ocr_mod
from anchor.chunking import chunk_text
from anchor.ocr import IMAGE_EXTENSIONS, extract_text_from_image


def test_image_extensions():
    assert ".png" in IMAGE_EXTENSIONS
    assert ".pdf" not in IMAGE_EXTENSIONS


def test_extract_text_calls_tesseract(tmp_path, monkeypatch):
    img_path = tmp_path / "shot.png"
    Image.new("RGB", (100, 40), "white").save(img_path)
    monkeypatch.setattr(
        ocr_mod.pytesseract, "image_to_string",
        lambda img: "  TypeError: module object is not callable  \n")
    assert extract_text_from_image(img_path) == "TypeError: module object is not callable"


def test_chunk_empty():
    assert chunk_text("   ") == []


def test_chunk_short_text_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_chunk_long_text_overlaps():
    text = "x" * 3200
    chunks = chunk_text(text, chunk_size=1500, overlap=200)
    assert all(len(c) <= 1500 for c in chunks)
    assert chunks[0][-200:] == chunks[1][:200]          # overlap preserved
    assert "".join([chunks[0]] + [c[200:] for c in chunks[1:]]) == text  # no loss
