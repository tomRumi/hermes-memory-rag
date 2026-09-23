#!/usr/bin/env python3
"""Re-index the written pages after they change.

A page copied into `~/.hermes/wikis/<project>/` is just a file: searching finds
nothing until the project's wiki layer is rebuilt. The importer cannot do this
itself — it runs inside Hermes, which has no vector library — so it is a separate
step, and the wrapper that runs the importer runs this straight afterwards.

Runs with the engine's interpreter:

    <venv>/bin/python scripts/reindex_wiki.py
    <venv>/bin/python scripts/reindex_wiki.py --project my-project

Rebuilding is per project: only the pages that changed matter, and rebuilding all
of them on a large install would be wasted work. Pass no --project to rebuild each
project that has a wiki directory.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_WIKI_ROOT = Path(os.environ.get("HERMES_WIKI_ROOT",
                                        str(Path.home() / ".hermes" / "wikis")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild a project's wiki search layer")
    ap.add_argument("--wiki-root", default=str(DEFAULT_WIKI_ROOT))
    ap.add_argument("--project", action="append", default=[],
                    help="project to rebuild (repeatable; default: every wiki directory)")
    args = ap.parse_args(argv)

    from hermes_memory_rag import server

    wiki_root = Path(args.wiki_root).expanduser()
    if not wiki_root.is_dir():
        print(f"no wiki directory at {wiki_root} — nothing to do")
        return 0

    if args.project:
        targets = [(name, wiki_root / name) for name in args.project]
    else:
        targets = [(child.name, child) for child in sorted(wiki_root.iterdir()) if child.is_dir()]

    started = time.time()
    failures = 0
    for name, directory in targets:
        if not directory.is_dir():
            print(f"  {name}: no such directory ({directory})")
            failures += 1
            continue
        pages = sum(1 for _ in directory.rglob("*.md"))
        if pages == 0:
            print(f"  {name}: no pages, skipped")
            continue
        try:
            result = server.ingest_wiki(str(directory), name)
            print(f"  {name}: {pages} page(s) -> {result}")
        except Exception as exc:  # noqa: BLE001 - report, do not crash the job
            print(f"  {name}: FAILED ({exc})")
            failures += 1

    print(f"re-indexed {len(targets)} project(s) in {round(time.time() - started, 1)}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())