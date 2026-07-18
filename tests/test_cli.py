import os

import pytest

from anchor.cli import build_parser, resolve_provider_name
from anchor.config import Config
from anchor.walker import MAX_FILE_BYTES


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


def test_index_command_single_indexable_file(tmp_path, monkeypatch, capsys):
    import anchor.cli as cli_mod
    f = tmp_path / "note.md"
    f.write_text("hello")

    class FakeIndexer:
        def __init__(self):
            self.paths = []
        def index_file(self, p):
            self.paths.append(p)
            return "indexed"
        def prune(self, under=None):
            return []

    fake = FakeIndexer()
    cfg = Config(data_dir=tmp_path / "data")
    monkeypatch.setattr(cli_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(cli_mod, "_make_indexer", lambda c: fake)
    assert cli_mod.main(["index", str(f)]) == 0
    assert fake.paths == [f]
    out = capsys.readouterr().out
    assert "indexed" in out


def test_index_command_single_oversized_file_is_skipped(tmp_path, monkeypatch, capsys):
    import anchor.cli as cli_mod
    f = tmp_path / "big.md"
    f.write_text("x")
    os.truncate(f, MAX_FILE_BYTES + 1)     # sparse file, no real disk usage

    class FakeIndexer:
        def __init__(self):
            self.calls = 0
        def index_file(self, p):
            self.calls += 1
            return "indexed"
        def prune(self, under=None):
            return []

    fake = FakeIndexer()
    cfg = Config(data_dir=tmp_path / "data")
    monkeypatch.setattr(cli_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(cli_mod, "_make_indexer", lambda c: fake)
    assert cli_mod.main(["index", str(f)]) == 0
    assert fake.calls == 0
    out = capsys.readouterr().out
    assert "skipped" in out


def test_index_command_single_secret_file_still_flows_to_indexer(tmp_path, monkeypatch, capsys):
    import anchor.cli as cli_mod
    f = tmp_path / ".env"
    f.write_text("API_KEY=supersecret")

    class FakeIndexer:
        def __init__(self):
            self.paths = []
        def index_file(self, p):
            self.paths.append(p)
            return "blocked"
        def prune(self, under=None):
            return []

    fake = FakeIndexer()
    cfg = Config(data_dir=tmp_path / "data")
    monkeypatch.setattr(cli_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(cli_mod, "_make_indexer", lambda c: fake)
    assert cli_mod.main(["index", str(f)]) == 0
    assert fake.paths == [f]
    out = capsys.readouterr().out
    assert "blocked" in out


def test_provider_resolution_defaults_to_local():
    cfg = Config()  # allow_cloud False
    assert resolve_provider_name(cfg, cloud_flag=False) == "ollama"


def test_provider_resolution_flag_is_one_shot_consent():
    cfg = Config()
    assert resolve_provider_name(cfg, cloud_flag=True) == "gemini"


def test_provider_resolution_standing_consent():
    cfg = Config(allow_cloud=True, cloud_provider="groq")
    assert resolve_provider_name(cfg, cloud_flag=False) == "groq"
