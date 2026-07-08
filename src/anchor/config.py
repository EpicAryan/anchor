from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / ".anchor"


@dataclass
class Config:
    data_dir: Path = DEFAULT_DATA_DIR
    watch_dir: Path = Path.home() / "Screenshots"
    allow_cloud: bool = False
    cloud_provider: str = "gemini"
    top_k: int = 5
    chunk_size: int = 1500
    chunk_overlap: int = 200

    @property
    def db_path(self) -> Path:
        return self.data_dir / "anchor.db"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "chroma"


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ. Refuses insecure files."""
    if not path.exists():
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(
            f"{path} is readable by group/other; fix with: chmod 600 {path}"
        )
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_config(data_dir: Path | None = None) -> Config:
    data_dir = data_dir or DEFAULT_DATA_DIR
    data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(data_dir, 0o700)

    cfg = Config(data_dir=data_dir)
    config_file = data_dir / "config.json"
    if config_file.exists():
        raw = json.loads(config_file.read_text())
        valid = {f.name for f in fields(Config)}
        for key, value in raw.items():
            if key not in valid or key == "data_dir":
                continue
            if key == "watch_dir":
                value = Path(value)
            setattr(cfg, key, value)

    load_env_file(data_dir / "env")
    return cfg
