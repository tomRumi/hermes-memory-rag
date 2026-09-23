#!/usr/bin/env python3
"""Compare two hermes-rag stores: node counts and the top hits for a query.

The point of a backup is that a restored store answers identically to the
original, so this is the check the restore drill uses.

Usage:
    python scripts/compare_stores.py --a ~/hermes-rag --b /tmp/restored \
        --collection rag_myproject__wiki --query "how are learnings consolidated" --top 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import chromadb

EMBED_MODEL = os.environ.get("HERMES_RAG_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.environ.get("HERMES_RAG_OLLAMA_URL", "http://localhost:11434")


def counts(store: Path) -> dict[str, int]:
    client = chromadb.PersistentClient(path=str(store))
    return {c.name: c.count() for c in client.list_collections()}


def top_hits(store: Path, collection: str, query: str, top: int, embed) -> list[tuple[str, float]]:
    client = chromadb.PersistentClient(path=str(store))
    col = client.get_or_create_collection(collection)
    if col.count() == 0:
        return []
    vector = embed.get_text_embedding(query)
    got = col.query(query_embeddings=[vector], n_results=min(top, col.count()))
    ids = (got.get("ids") or [[]])[0]
    dists = (got.get("distances") or [[]])[0]
    return list(zip(ids, [round(d, 6) for d in dists]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare two hermes-rag stores")
    ap.add_argument("--a", required=True, help="original store")
    ap.add_argument("--b", required=True, help="restored store")
    ap.add_argument("--collection", help="collection to compare hits on")
    ap.add_argument("--query", help="query text for the hit comparison")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args(argv)

    a, b = Path(args.a).expanduser(), Path(args.b).expanduser()
    ca, cb = counts(a), counts(b)

    ok = True
    names = sorted(set(ca) | set(cb))
    print("counts:")
    for name in names:
        na, nb = ca.get(name, 0), cb.get(name, 0)
        flag = "ok" if na == nb else "MISMATCH"
        if na != nb:
            ok = False
        print(f"  {flag:9} {name}: {na} vs {nb}")

    if args.collection and args.query:
        from llama_index.embeddings.ollama import OllamaEmbedding
        embed = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_URL)
        ha = top_hits(a, args.collection, args.query, args.top, embed)
        hb = top_hits(b, args.collection, args.query, args.top, embed)
        print(f"\ntop {args.top} for {args.collection!r} query={args.query!r}:")
        for store, hits in (("original", ha), ("restored", hb)):
            for rank, (node_id, dist) in enumerate(hits, 1):
                print(f"  {store:9} #{rank} {node_id} distance={dist}")
        same = [i for i, _ in ha] == [i for i, _ in hb]
        print(f"  top-{args.top} id order identical: {same}")
        ok = ok and same

    print("\nRESULT:", "identical" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())