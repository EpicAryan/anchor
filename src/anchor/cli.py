from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anchor.config import Config, load_config
from anchor.db import MetadataDB
from anchor.embedder import Embedder
from anchor.indexer import Indexer
from anchor.ocr import IMAGE_EXTENSIONS
from anchor.providers import ProviderError, get_provider
from anchor.query import answer_question
from anchor.vectorstore import VectorStore
from anchor.watcher import run_watcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anchor", description="Local-first personal knowledge search")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index a file or directory")
    p_index.add_argument("path")

    p_watch = sub.add_parser("watch", help="watch the screenshots folder")
    p_watch.add_argument("--dir", default=None,
                         help="override configured watch directory")

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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()

    if args.command == "index":
        indexer = _make_indexer(config)
        target = Path(args.path).expanduser()
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
        print(ans.text)
        if ans.sources:
            print("\nSources:")
            for s in ans.sources:
                print(f"  {s}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
