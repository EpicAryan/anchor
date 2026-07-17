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
