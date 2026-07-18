import os
import stat
from pathlib import Path

import pytest

from anchor.config import Config, load_config, load_env_file


def test_defaults_are_private(tmp_path):
    cfg = load_config(data_dir=tmp_path / "anchor-data")
    assert cfg.allow_cloud is False
    assert cfg.top_k == 5
    assert cfg.db_path == tmp_path / "anchor-data" / "anchor.db"
    assert cfg.vector_dir == tmp_path / "anchor-data" / "chroma"


def test_data_dir_created_with_0700(tmp_path):
    cfg = load_config(data_dir=tmp_path / "anchor-data")
    mode = stat.S_IMODE(cfg.data_dir.stat().st_mode)
    assert mode == 0o700


def test_config_json_overlay(tmp_path):
    data_dir = tmp_path / "anchor-data"
    data_dir.mkdir()
    (data_dir / "config.json").write_text(
        '{"allow_cloud": true, "cloud_provider": "groq", "top_k": 3}'
    )
    cfg = load_config(data_dir=data_dir)
    assert cfg.allow_cloud is True
    assert cfg.cloud_provider == "groq"
    assert cfg.top_k == 3


def test_env_file_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env_file = tmp_path / "env"
    env_file.write_text("GEMINI_API_KEY=abc123\n# comment\n\n")
    os.chmod(env_file, 0o600)
    load_env_file(env_file)
    assert os.environ["GEMINI_API_KEY"] == "abc123"


def test_env_file_rejected_if_world_readable(tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text("GEMINI_API_KEY=abc123\n")
    os.chmod(env_file, 0o644)
    with pytest.raises(PermissionError):
        load_env_file(env_file)


def test_missing_env_file_is_fine(tmp_path):
    load_env_file(tmp_path / "does-not-exist")  # must not raise


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


def test_malformed_watch_dirs_string_raises(tmp_path):
    data_dir = tmp_path / "d"
    data_dir.mkdir(mode=0o700)
    (data_dir / "config.json").write_text('{"watch_dirs": "/tmp/x"}')
    with pytest.raises(ValueError):
        load_config(data_dir)
