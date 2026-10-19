#!/usr/bin/env python3
"""Re-index the written pages after they change.

A page copied into `~/.hermes/wikis/<project>/` is just a file: searching finds
nothing until the project's wiki layer is rebuilt. The importer cannot do this
itself — it runs inside Hermes, which has no vector library — so it is a separate
step, and the wrapper that runs the importer runs this straight afterwards.

Runs with the engine's interpreter:

    <venv>/bin/python scripts/reindex_wiki.py
    <venv>/bin/python scripts/reindex_wiki.py --project my-project
    <venv>/bin/python scripts/reindex_wiki.py --wiki-root DIR --project P --namespace vps-host
    <venv>/bin/python scripts/reindex_wiki.py --incremental --prune --dry-run

The layer lands in the namespace given by --namespace (it sets
MEMORY_RAG_LOCAL_NAMESPACE for this in-process call); without it, in whatever that
variable already says, and failing that in "legacy", the unprefixed layout. Point
HERMES_RAG_STORE at the store the server uses.

Rebuilding is per project: only the pages that changed matter, and rebuilding all
of them on a large install would be wasted work. Pass no --project to rebuild each
project that has a wiki directory.

By default each project's wiki layer is rebuilt from scratch (`ingest_wiki`).
--incremental updates it in place instead (`reindex_wiki`, the same function the
server offers as a tool): pages that did not change cost a hash and no embedding, an
edited page replaces its own chunks, and --prune also removes pages that are gone
from the folder. --dry-run prints the plan and writes nothing; --exclude takes a
comma list of top-level subfolders to leave out. Either way the folder must be one
the namespace may index (see namespaces.json in the README).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_WIKI_ROOT = Path(os.environ.get("HERMES_WIKI_ROOT",
                                        str(Path.home() / ".hermes" / "wikis")))


def _reindex(server, directory: Path, name: str, args) -> str:
    """Run the incremental update to completion (it works in time-boxed passes) and
    summarise it in one line."""
    while True:
        report = json.loads(server.reindex_wiki_blocking(
            str(directory), name, mode="incremental", prune=args.prune,
            dry_run=args.dry_run, exclude=args.exclude, budget_seconds=300))
        if report.get("busy"):
            return "busy: another re-index of this project is running"
        if not report["partial"]:
            break
    files, nodes = report["files"], report["nodes"]
    return (f"{'would index' if args.dry_run else 'indexed'} {files['indexed']} page(s), "
            f"{files['unchanged']} unchanged, {files['removed']} removed, "
            f"{files['skipped']} skipped; nodes +{nodes['added']} -{nodes['deleted']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild a project's wiki search layer")
    ap.add_argument("--wiki-root", default=str(DEFAULT_WIKI_ROOT))
    ap.add_argument("--project", action="append", default=[],
                    help="project to rebuild (repeatable; default: every wiki directory)")
    ap.add_argument("--namespace", default="",
                    help="namespace to write into, e.g. vps-host (default: "
                         "$MEMORY_RAG_LOCAL_NAMESPACE, else legacy)")
    ap.add_argument("--incremental", action="store_true",
                    help="update the layer in place instead of rebuilding it")
    ap.add_argument("--prune", action="store_true",
                    help="with --incremental: remove pages that are gone from the folder")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --incremental: print the plan, write nothing")
    ap.add_argument("--exclude", default="",
                    help="with --incremental: comma list of top-level subfolders to skip")
    args = ap.parse_args(argv)
    if not args.incremental and (args.prune or args.dry_run or args.exclude):
        ap.error("--prune, --dry-run and --exclude need --incremental")

    from hermes_memory_rag import server

    if args.namespace:
        if not server._valid_namespace(args.namespace):
            print(f"--namespace {args.namespace!r} is not a valid namespace "
                  f"(lowercase letters, digits and single hyphens, at most 40)")
            return 2
        os.environ[server.LOCAL_NAMESPACE_ENV] = args.namespace
    try:
        namespace = server._local_namespace()
    except PermissionError as exc:
        print(exc)
        return 2
    print(f"namespace: {namespace}")

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
            if args.incremental:
                result = _reindex(server, directory, name, args)
            else:
                result = server.ingest_wiki(str(directory), name)
            print(f"  {name}: {pages} page(s) -> {result}")
        except Exception as exc:  # noqa: BLE001 - report, do not crash the job
            print(f"  {name}: FAILED ({exc})")
            failures += 1

    print(f"re-indexed {len(targets)} project(s) in {round(time.time() - started, 1)}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())