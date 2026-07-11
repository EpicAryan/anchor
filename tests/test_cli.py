from anchor.cli import build_parser, resolve_provider_name
from anchor.config import Config


def test_parser_commands():
    parser = build_parser()
    args = parser.parse_args(["ask", "where is that error?", "--cloud"])
    assert args.command == "ask"
    assert args.question == "where is that error?"
    assert args.cloud is True

    args = parser.parse_args(["index", "/pics"])
    assert args.command == "index" and args.path == "/pics"

    args = parser.parse_args(["watch"])
    assert args.command == "watch"


def test_provider_resolution_defaults_to_local():
    cfg = Config()  # allow_cloud False
    assert resolve_provider_name(cfg, cloud_flag=False) == "ollama"


def test_provider_resolution_flag_is_one_shot_consent():
    cfg = Config()
    assert resolve_provider_name(cfg, cloud_flag=True) == "gemini"


def test_provider_resolution_standing_consent():
    cfg = Config(allow_cloud=True, cloud_provider="groq")
    assert resolve_provider_name(cfg, cloud_flag=False) == "groq"
