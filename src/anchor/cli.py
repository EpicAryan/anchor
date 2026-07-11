from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
import warnings
from pathlib import Path

# Keep CLI output clean: the ML libraries are chatty by default. These must
# be set before torch/transformers load (which happens lazily on first embed).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning)

from anchor.config import Config, load_config
from anchor.db import MetadataDB
from anchor.embedder import Embedder
from anchor.indexer import Indexer
from anchor.ocr import IMAGE_EXTENSIONS
from anchor.providers import ProviderError, get_provider
from anchor.query import answer_question, find_matches
from anchor.vectorstore import VectorStore
from anchor.watcher import run_watcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anchor", description="Local-first personal knowledge search")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index a file or directory")
    p_index.add_argument("path", nargs="?", default=None,
                         help="file or folder (default: configured watch_dir)")

    p_watch = sub.add_parser("watch", help="watch the screenshots folder")
    p_watch.add_argument("--dir", default=None,
                         help="override configured watch directory")

    p_find = sub.add_parser(
        "find", help="list screenshots matching a phrase (no LLM, fully local)")
    p_find.add_argument("question")
    p_find.add_argument("-k", type=int, default=None,
                        help="max results (default 10)")

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--cloud", action="store_true",
                       help="one-shot consent to send redacted snippets "
                            "to the configured cloud provider")
    p_ask.add_argument("--local", action="store_true",
                       help="force local LLM even if allow_cloud is true")
    p_ask.add_argument("-k", type=int, default=None, help="top-k chunks")
    return parser


def resolve_provider_name(config: Config, cloud_flag: bool,
                          local_flag: bool = False) -> str:
    if local_flag:
        return "ollama"
    if cloud_flag or config.allow_cloud:
        return config.cloud_provider
    return "ollama"


def _make_indexer(config: Config) -> Indexer:
    return Indexer(MetadataDB(config.db_path), Embedder(),
                   VectorStore(config.vector_dir), config)


def _print_answer(ans) -> None:
    width = min(shutil.get_terminal_size((100, 20)).columns, 100)
    print()
    for line in ans.text.splitlines():
        print(textwrap.fill(line, width=width, subsequent_indent="  ")
              if len(line) > width else line)
    if ans.sources:
        print("\nSources:")
        for i, s in enumerate(ans.sources, 1):
            print(f"  {i}. {s}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()

    if args.command == "index":
        indexer = _make_indexer(config)
        target = Path(args.path).expanduser() if args.path else config.watch_dir
        files = ([target] if target.is_file() else
                 sorted(p for p in target.iterdir()
                        if p.suffix.lower() in IMAGE_EXTENSIONS))
        if not files:
            print("nothing to index", file=sys.stderr)
            return 1
        for f in files:
            print(f"{indexer.index_file(f):>12}  {f}")
        return 0

    if args.command == "watch":
        watch_dir = Path(args.dir).expanduser() if args.dir else config.watch_dir
        run_watcher(watch_dir, _make_indexer(config))
        return 0

    if args.command == "find":
        config.top_k = args.k or 10
        matches = find_matches(
            args.question, config=config, embedder=Embedder(),
            store=VectorStore(config.vector_dir))
        if not matches:
            print("no matches")
            return 1
        print()
        for i, m in enumerate(matches, 1):
            print(f"  {i}. {m['path']}")
            print(f"     {m['snippet']}")
        print()
        return 0

    if args.command == "ask":
        if args.k:
            config.top_k = args.k
        name = resolve_provider_name(config, args.cloud, args.local)
        try:
            provider = get_provider(name)
        except ProviderError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if provider.is_cloud:
            print(f"[anchor] using cloud provider '{name}' "
                  f"(redacted snippets will be sent)", file=sys.stderr)
        ans = answer_question(
            args.question, config=config, embedder=Embedder(),
            store=VectorStore(config.vector_dir), provider=provider)
        _print_answer(ans)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
