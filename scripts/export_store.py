#!/usr/bin/env python3
"""Export every collection from a hermes-rag store to plain-text files.

Writes two files:

  <out>                 readable: one line per node with id, text and metadata.
                        This is the file to read, diff and commit.
  <out>.vectors.jsonl   the stored vectors, one line per node (id + embedding).

Both are needed, for a reason measured on this machine: the code layer holds raw
ollama vectors while the wiki and memory layers hold L2-normalised ones (unit
norm). Chroma's default L2 distance is scale-sensitive, so re-embedding documents
at restore time does NOT reproduce the original rankings for the normalised
layers. Carrying the vectors makes a restore exact; the readable file stays the
thing you inspect.

Vectors can be skipped with --no-vectors for a text-only export; a restore from
one re-embeds and may rank differently, and says so.

Usage:
    python scripts/export_store.py --out backup.jsonl
    python scripts/export_store.py --out backup.jsonl --no-vectors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import chromadb

DEFAULT_STORE = Path(os.environ.get("HERMES_RAG_STORE", str(Path.home() / "hermes-rag")))
EMBED_MODEL = os.environ.get("HERMES_RAG_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.environ.get("HERMES_RAG_OLLAMA_URL", "http://localhost:11434")
PAGE = 500


def _norm(vector: list[float]) -> float:
    return sum(float(x) * float(x) for x in vector) ** 0.5


def sort_key(row: dict) -> tuple[str, str]:
    return (row["collection"], row["id"])


def export(store: Path, out: Path, with_vectors: bool) -> int:
    """Write *store* to *out* (+ vectors sidecar); return nodes written."""
    client = chromadb.PersistentClient(path=str(store))
    collections = sorted(client.list_collections(), key=lambda c: c.name)

    header = {
        "kind": "header",
        "hermes_rag_export": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "store": str(store),
        "embed_model": EMBED_MODEL,
        "ollama_url": OLLAMA_URL,
        "vectors_included": bool(with_vectors),
        "collections": {},
    }

    text_rows: list[dict] = []
    vector_rows: list[dict] = []

    for col in collections:
        count = col.count()
        norm_sample: list[float] = []
        offset = 0
        include = cast(Any, ["documents", "metadatas", "embeddings"])
        while offset < count:
            got = col.get(limit=PAGE, offset=offset, include=include)
            ids = got.get("ids") or []
            if not ids:
                break
            docs = got.get("documents") or [None] * len(ids)
            metas = got.get("metadatas") or [None] * len(ids)
            embs = got.get("embeddings")
            for i, node_id in enumerate(ids):
                text_rows.append({
                    "kind": "node",
                    "collection": col.name,
                    "id": node_id,
                    "document": docs[i],
                    "metadata": metas[i] or {},
                })
                if embs is not None:
                    vector = [float(x) for x in embs[i]]
                    if len(norm_sample) < 5:
                        norm_sample.append(round(_norm(vector), 4))
                    if with_vectors:
                        vector_rows.append({"id": node_id, "collection": col.name,
                                            "embedding": vector})
            offset += len(ids)

        header["collections"][col.name] = {
            "count": count,
            "metadata": dict(col.metadata or {}),
            # The size of each fact's number list, measured. 1.0 means this part
            # of the store stores lists sized to 1; about 20 means it stores them
            # as the model produced them. The two behave differently under the
            # distance measure, which is why an exact rebuild needs the numbers file.
            "embedding_list_sizes": norm_sample,
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    # Stable order so an unchanged store produces byte-identical files: without
    # this, a daily backup looks like a change and every commit carries the whole
    # file again.
    text_rows.sort(key=sort_key)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
        for row in text_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    if with_vectors:
        numbers = Path(str(out) + ".embedding-numbers.jsonl")
        vector_rows.sort(key=lambda r: (r["collection"], r["id"]))
        with numbers.open("w", encoding="utf-8") as fh:
            for row in vector_rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"wrote {len(vector_rows)} embedding numbers -> {numbers}")

    print(f"exported {len(text_rows)} nodes from {len(collections)} collections -> {out}")
    for name, meta in header["collections"].items():
        sizes = meta["embedding_list_sizes"]
        print(f"  {name}: {meta['count']} nodes, number-list sizes {sizes or '—'}")
    return len(text_rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export a hermes-rag store to plain-text files")
    ap.add_argument("--out", required=True, help="destination .jsonl file (readable)")
    ap.add_argument("--store", default=str(DEFAULT_STORE), help="store directory to export")
    ap.add_argument("--no-vectors", action="store_true",
                    help="skip the vectors sidecar; a restore then re-embeds and may rank differently")
    args = ap.parse_args(argv)

    store = Path(args.store).expanduser()
    if not store.exists():
        print(f"no store at {store}", file=sys.stderr)
        return 2
    export(store, Path(args.out).expanduser(), not args.no_vectors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())