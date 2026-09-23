#!/usr/bin/env python3
"""Give every stored node the fields the engine now relies on.

Nodes written before the status field existed carry no ``status``, no ``source``
and (in the code and wiki layers) no ``created``. Recall treats a missing status
as live, so an un-migrated store still works — but nothing can be marked as
replaced or withdrawn, and the merge counter cannot tell a real learning from a
mirrored note.

This walks every collection and fills in the missing fields. It is idempotent:
nodes that already carry a status are left untouched, so running it twice changes
nothing. Nothing is deleted or rewritten — only fields are added.

Dry run by default; pass --apply to write.

Usage:
    python scripts/migrate_status.py
    python scripts/migrate_status.py --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, cast

import chromadb

DEFAULT_STORE = Path(os.environ.get("HERMES_RAG_STORE", str(Path.home() / "hermes-rag")))
PAGE = 500

# Source recorded per layer for nodes that predate the field.
_SOURCE_BY_LAYER = {"code": "ingest_code", "wiki": "ingest_wiki", "memory": "learn"}


def migrate(store: Path, apply: bool) -> int:
    client = chromadb.PersistentClient(path=str(store))
    collections = sorted(client.list_collections(), key=lambda c: c.name)
    if not collections:
        raise SystemExit(f"no collections in {store}")

    total = 0
    for col in collections:
        stamped = 0
        offset = 0
        count = col.count()
        while offset < count:
            got = col.get(limit=PAGE, offset=offset,
                          include=cast(Any, ["metadatas"]))
            ids = got.get("ids") or []
            if not ids:
                break
            metas = got.get("metadatas") or [{}] * len(ids)
            updates: list[str] = []
            new_metas: list[dict] = []
            for node_id, meta in zip(ids, metas):
                meta = dict(meta or {})
                if meta.get("status"):
                    continue
                layer = str(meta.get("layer") or "")
                meta["status"] = "active"
                meta.setdefault("source", _SOURCE_BY_LAYER.get(layer, "unknown"))
                meta.setdefault("created", "")
                updates.append(node_id)
                new_metas.append(meta)
            if updates:
                stamped += len(updates)
                if apply:
                    col.update(ids=updates, metadatas=cast(Any, new_metas))
            offset += len(ids)
        total += stamped
        verb = "stamped" if apply else "would stamp"
        print(f"  {col.name}: {verb} {stamped} of {count} nodes")

    print(("migrated " if apply else "would migrate ") + f"{total} nodes in {store}")
    if not apply and total:
        print("dry run — nothing written. Re-run with --apply.")
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Add status/source/created to stored nodes")
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    ap.add_argument("--apply", action="store_true", help="write the changes")
    args = ap.parse_args(argv)

    store = Path(args.store).expanduser()
    if not store.exists():
        print(f"no store at {store}", file=sys.stderr)
        return 2
    migrate(store, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())