#!/usr/bin/env python3
"""Copy a store into a new one, with distance measured by angle instead of size.

Why this exists
---------------
This store holds two different kinds of number lists. The code part keeps the
numbers exactly as the embedding model produced them (list size ~20); the wiki
and memory parts keep lists resized to 1. Distance was measured by size
difference, which is affected by that resizing — so recomputing the numbers from
the text on another machine returned a different search order for the wiki and
memory parts (measured: ranks 2 and 3 swapped on both).

Chroma lets a collection measure distance by angle instead ("cosine"), which
ignores how big the lists are. With that, both kinds of list behave the same,
recomputing numbers anywhere gives the same order, and the embedding-numbers file
stops being needed for a rebuild to be faithful.

This script copies collection by collection: it reads the numbers already stored
(no re-embedding, seconds not minutes) and writes them into a new store whose
collections use the angle measure. The original store is never modified.

Usage:
    python scripts/migrate_to_cosine.py --from ~/hermes-rag --to ~/hermes-rag-cosine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import chromadb

PAGE = 500


def migrate(src: Path, dst: Path) -> None:
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} already exists and is not empty — refusing to write into it")

    from_client = chromadb.PersistentClient(path=str(src))
    dst.mkdir(parents=True, exist_ok=True)
    to_client = chromadb.PersistentClient(path=str(dst))

    collections = sorted(from_client.list_collections(), key=lambda c: c.name)
    if not collections:
        raise SystemExit(f"no collections in {src}")

    for col in collections:
        count = col.count()
        target = to_client.get_or_create_collection(
            col.name, metadata={"hnsw:space": "cosine"},
        )
        offset = 0
        while offset < count:
            got = col.get(limit=PAGE, offset=offset,
                          include=cast(Any, ["documents", "metadatas", "embeddings"]))
            ids = got.get("ids") or []
            if not ids:
                break
            embeddings = got.get("embeddings")
            if embeddings is None:
                raise SystemExit(f"{col.name}: some nodes carry no numbers; re-ingest instead")
            target.upsert(
                ids=ids,
                documents=cast(Any, got.get("documents") or [None] * len(ids)),
                metadatas=cast(Any, got.get("metadatas") or [None] * len(ids)),
                embeddings=cast(Any, [list(v) for v in embeddings]),
            )
            offset += len(ids)
        print(f"  {col.name}: copied {count} nodes -> angle-measured collection")

    print(f"copied {len(collections)} collections into {dst} (original untouched)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Copy a store with angle-based distance")
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    args = ap.parse_args(argv)

    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    if not src.exists():
        print(f"no store at {src}", file=sys.stderr)
        return 2
    migrate(src, dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())