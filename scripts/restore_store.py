#!/usr/bin/env python3
"""Restore a hermes-rag export into a store (normally a fresh, empty one).

Prefers the vectors sidecar written by export_store.py, which makes the restore
exact — same vectors, same rankings, and it does not depend on this machine's
ollama behaving as the original did. Without the sidecar the documents are
re-embedded instead, which reproduces the original rankings only if the vectors
were raw (the code layer) and not normalised (the wiki and memory layers); that
case prints a warning rather than pretending.

Nothing is written until the whole export has parsed, so a truncated file leaves
the target store untouched.

Usage:
    python scripts/restore_store.py --in backup.jsonl --store /tmp/restored
    python scripts/restore_store.py --in backup.jsonl --store ~/hermes-rag --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import chromadb

DEFAULT_STORE = Path(os.environ.get("HERMES_RAG_STORE", str(Path.home() / "hermes-rag")))
EMBED_MODEL = os.environ.get("HERMES_RAG_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.environ.get("HERMES_RAG_OLLAMA_URL", "http://localhost:11434")
BATCH = 64


def read_export(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    """Parse the whole export; returns (header, {collection: [node, ...]})."""
    header: dict | None = None
    by_collection: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: not valid JSON ({exc})") from exc
            if row.get("kind") == "header":
                if header is not None:
                    raise SystemExit(f"{path}:{lineno}: second header line")
                header = row
                continue
            if header is None:
                raise SystemExit(f"{path}:{lineno}: node before the header line")
            if row.get("kind") != "node":
                raise SystemExit(f"{path}:{lineno}: unknown line kind {row.get('kind')!r}")
            by_collection[row["collection"]].append(row)
    if header is None:
        raise SystemExit(f"{path}: no header line — not a hermes-rag export")
    return header, dict(by_collection)


def read_vectors(src: Path) -> dict[str, list[float]]:
    """Load the optional embedding-numbers file; {} when it is absent."""
    sidecar = Path(str(src) + ".embedding-numbers.jsonl")
    if not sidecar.exists():
        return {}
    vectors: dict[str, list[float]] = {}
    with sidecar.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{sidecar}:{lineno}: not valid JSON ({exc})") from exc
            vectors[row["id"]] = row["embedding"]
    return vectors


def restore(src: Path, store: Path, force: bool) -> int:
    header, by_collection = read_export(src)
    vectors = read_vectors(src)

    if vectors:
        print(f"using {len(vectors)} stored number lists from {src.name}.embedding-numbers.jsonl")
    else:
        if (exported_model := header.get("embed_model")) and exported_model != EMBED_MODEL:
            raise SystemExit(
                f"refusing to restore: the export was embedded with {exported_model!r} but this "
                f"machine would embed with {EMBED_MODEL!r}. Set HERMES_RAG_EMBED_MODEL to match, or "
                f"the restored store will rank differently."
            )
        print(f"no numbers file — recomputing {EMBED_MODEL!r} numbers from the text. WARNING: parts of "
              f"the store whose number lists are sized 1.0 (see the export header) will NOT return the "
              f"same order.")

    store.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store))

    model = None
    for name, nodes in sorted(by_collection.items()):
        # Read-only existence check. get_or_create would create the collection
        # here, without the settings recorded in the export — and the real
        # get_or_create below would then find it already there and silently keep
        # the default distance measure, producing a store that answers differently
        # from the original.
        existing = 0
        try:
            existing = client.get_collection(name).count()
        except Exception:
            existing = 0  # not present yet
        if existing and not force:
            raise SystemExit(
                f"collection {name!r} already holds {existing} nodes in {store}. "
                f"Restore into an empty store, or pass --force to overwrite."
            )
        if force and existing:
            client.delete_collection(name)

        settings = (header.get("collections", {}).get(name) or {}).get("metadata") or {}
        col = client.get_or_create_collection(name, metadata=settings or None)

        # Only meaningful when the numbers file was used: without it every node
        # is recomputed, so "missing" would name all of them.
        missing = [n["id"] for n in nodes if n["id"] not in vectors] if vectors else []
        if not vectors and model is None:
            from llama_index.embeddings.ollama import OllamaEmbedding
            model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_URL)

        done = 0
        for start in range(0, len(nodes), BATCH):
            batch = nodes[start:start + BATCH]
            if vectors:
                batch = [n for n in batch if n["id"] in vectors]
                if not batch:
                    continue
                embeddings = [vectors[n["id"]] for n in batch]
            else:
                assert model is not None
                embeddings = model.get_text_embedding_batch([n["document"] or "" for n in batch])
            col.upsert(
                ids=[n["id"] for n in batch],
                documents=[n["document"] for n in batch],
                metadatas=[n["metadata"] or {} for n in batch],
                embeddings=cast(Any, embeddings),
            )
            done += len(batch)
        note = f" ({len(missing)} without a stored vector, skipped)" if missing else ""
        print(f"  {name}: restored {done}/{len(nodes)} nodes{note}")
        if missing and not force:
            raise SystemExit(
                f"{len(missing)} nodes of {name!r} have no vector in the sidecar — the export and "
                f"its vectors file are out of sync. Re-export."
            )

    total = sum(len(v) for v in by_collection.values())
    print(f"restored {total} nodes into {store}")
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Restore a hermes-rag export")
    ap.add_argument("--in", dest="src", required=True, help="the .jsonl export to restore")
    ap.add_argument("--store", default=str(DEFAULT_STORE), help="target store directory")
    ap.add_argument("--force", action="store_true",
                    help="overwrite collections that already hold nodes")
    args = ap.parse_args(argv)

    src = Path(args.src).expanduser()
    if not src.exists():
        print(f"no export at {src}", file=sys.stderr)
        return 2
    restore(src, Path(args.store).expanduser(), args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())