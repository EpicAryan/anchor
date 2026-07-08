from __future__ import annotations

from pathlib import Path

import pytesseract
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def extract_text_from_image(path: Path) -> str:
    with Image.open(path) as img:
        return pytesseract.image_to_string(img).strip()
