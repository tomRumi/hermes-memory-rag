#!/usr/bin/env python3
"""Copy every collection of one namespace into another, in the same store.

Typical use: the store as it was before namespaces ("legacy", unprefixed names
such as `rag_todo-api__memory`) is copied to a real namespace ("vps-host",
`rag_vps-host.todo-api__memory`), so a token for that namespace sees exactly what
the legacy token saw.

What it does, and does not do:

- copies ids, documents, metadatas and the stored embeddings exactly. Nothing is
  re-embedded, so search order is unchanged and no Ollama is needed. Each target
  collection gets the source's metadata and index configuration (distance etc.);
- rewrites the namespace only where data names it: a metadata field called
  `namespace` holding the source namespace (also inside llama-index's
  `_node_content` copy of the metadata). The server itself writes no such field,
  so on a store it wrote, no node changes. The usage ledger (usage.db) does name
  it: its `project` column holds the namespaced scope ("todo-api" for legacy,
  "vps-host.todo-api" otherwise), so the source's rows are copied with the scope
  rewritten, keeping stats' counts for the new namespace. Rows the target already
  has (its own usage, or an earlier run) are kept and never duplicated;
- refuses if any target collection already exists, unless --force (which deletes
  those TARGET collections first);
- verifies every collection afterwards: count, the full id set, and a sample of
  ids whose documents, metadatas and embeddings must match exactly;
- never deletes or changes the source. Retiring it is a separate, later step;
- refuses to run while another process holds the store open (the memory-rag
  service does, all the time), unless --i-stopped-the-service.

Operator procedure (live store):

    systemctl --user stop memory-rag
    <venv>/bin/python scripts/backup.py --store <store>
    cp -a <store> <store>.pre-migration  # raw rollback copy
    <venv>/bin/python scripts/migrate_namespace.py --store <store> \\
        --from <from> --to <to> --dry-run
    <venv>/bin/python scripts/migrate_namespace.py --store <store> \\
        --from <from> --to <to> --i-stopped-the-service
    # add an entry for the target namespace to tokens.json, repoint clients, then
    systemctl --user start memory-rag

Exit status: 0 done (or dry run), 1 refused or verification failed, 2 bad usage.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, cast

LEGACY = "legacy"
LAYERS = ("code", "wiki", "memory")
PAGE = 500
SAMPLE = 25
_NAMESPACE_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")  # as the server's
_SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"


class Refused(Exception):
    """A reason not to go on; nothing has been written when it is raised."""


def valid_namespace(ns: str) -> bool:
    return len(ns) <= 40 and _NAMESPACE_RE.fullmatch(ns) is not None


def scope_prefix(ns: str) -> str:
    return "" if ns == LEGACY else f"{ns}."


def split_name(name: str, ns: str) -> tuple[str, str] | None:
    """(slug, layer) if collection `name` belongs to namespace `ns`, else None."""
    m = re.fullmatch(rf"rag_{re.escape(scope_prefix(ns))}({_SLUG})__({'|'.join(LAYERS)})", name)
    return (m.group(1), m.group(2)) if m else None


def target_name(slug: str, layer: str, ns: str) -> str:
    return f"rag_{scope_prefix(ns)}{slug}__{layer}"


# ── is anyone else holding the store open? ────────────────────────────────


def holders(store: Path) -> list[tuple[int, str]]:
    """(pid, command name) of other processes with a file or cwd inside `store`.

    Only processes this user may inspect are seen, which covers a service run as
    the same user (memory-rag is a user unit).
    """
    store = store.resolve()
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        links = []
        try:
            links.append(os.readlink(entry / "cwd"))
            for fd in (entry / "fd").iterdir():
                try:
                    links.append(os.readlink(fd))
                except OSError:
                    pass
        except OSError:
            continue
        for link in links:
            path = Path(link)
            if path == store or store in path.parents:
                try:
                    comm = (entry / "comm").read_text().strip()
                except OSError:
                    comm = "?"
                found.append((int(entry.name), comm))
                break
    return found


# ── metadata ──────────────────────────────────────────────────────────────


def rewrite_metadata(meta: dict | None, src: str, dst: str) -> tuple[dict | None, bool]:
    """The metadata with a `namespace` field naming `src` changed to `dst`."""
    if not meta:
        return meta, False
    out, changed = dict(meta), False
    if out.get("namespace") == src:
        out["namespace"], changed = dst, True
    raw = out.get("_node_content")
    if isinstance(raw, str) and '"namespace"' in raw:
        try:
            node = json.loads(raw)
        except ValueError:
            node = None
        if isinstance(node, dict) and isinstance(node.get("metadata"), dict) \
                and node["metadata"].get("namespace") == src:
            node["metadata"]["namespace"] = dst
            out["_node_content"], changed = json.dumps(node), True
    return out, changed


# ── the ledger ────────────────────────────────────────────────────────────


def _ledger_rows(conn: sqlite3.Connection, ns: str) -> list[tuple]:
    rows = conn.execute("SELECT ts, project, event, detail FROM events").fetchall()
    prefix = scope_prefix(ns)
    if ns == LEGACY:
        return [r for r in rows if r[1] is not None and "." not in r[1]]
    return [r for r in rows if r[1] is not None and r[1].startswith(prefix)]


def _rescope(project: str, src: str, dst: str) -> str:
    return scope_prefix(dst) + project[len(scope_prefix(src)):]


def _open_ledger(store: Path) -> sqlite3.Connection | None:
    path = store / "usage.db"
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    has = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'").fetchone()
    if not has:
        conn.close()
        return None
    return conn


# ── copy and verify ───────────────────────────────────────────────────────


def _create_like(client, source, name: str):
    config = source.configuration or {}
    create: dict[str, Any] = {k: config[k] for k in ("hnsw", "spann") if config.get(k)}
    return client.create_collection(
        name, metadata=source.metadata or None,
        configuration=cast(Any, create or None),
        embedding_function=config.get("embedding_function"),
    )


def _copy(source, target, src: str, dst: str) -> int:
    rewritten, offset, total = 0, 0, source.count()
    while offset < total:
        got = source.get(limit=PAGE, offset=offset,
                         include=cast(Any, ["documents", "metadatas", "embeddings"]))
        ids = got["ids"]
        if not ids:
            break
        embeddings = got.get("embeddings")
        if embeddings is None or len(embeddings) != len(ids):
            raise Refused(f"{source.name}: some nodes carry no embedding; refusing to re-embed")
        metas = []
        for meta in got.get("metadatas") or [None] * len(ids):
            meta, changed = rewrite_metadata(meta, src, dst)
            metas.append(meta)
            rewritten += changed
        target.add(ids=ids, documents=cast(Any, got.get("documents")),
                   metadatas=cast(Any, metas), embeddings=cast(Any, embeddings))
        offset += len(ids)
    return rewritten


def _verify(source, target, src: str, dst: str) -> list[str]:
    import numpy as np

    problems = []
    if source.count() != target.count():
        problems.append(f"count {source.count()} != {target.count()}")
    src_ids = set(source.get(include=[])["ids"])
    dst_ids = set(target.get(include=[])["ids"])
    if src_ids != dst_ids:
        problems.append(f"id sets differ ({len(src_ids ^ dst_ids)} ids)")
    ids = sorted(src_ids)
    sample = ids if len(ids) <= SAMPLE else random.Random(0).sample(ids, SAMPLE)
    if not sample:
        return problems
    include = cast(Any, ["documents", "metadatas", "embeddings"])
    a, b = source.get(ids=sample, include=include), target.get(ids=sample, include=include)
    a_at = {i: n for n, i in enumerate(a["ids"])}
    b_at = {i: n for n, i in enumerate(b["ids"])}
    for i in sample:
        if i not in a_at or i not in b_at:
            problems.append(f"{i}: missing")
            continue
        x, y = a_at[i], b_at[i]
        if a["documents"][x] != b["documents"][y]:
            problems.append(f"{i}: document differs")
        if rewrite_metadata(a["metadatas"][x], src, dst)[0] != b["metadatas"][y]:
            problems.append(f"{i}: metadata differs")
        if not np.array_equal(np.asarray(a["embeddings"][x]), np.asarray(b["embeddings"][y])):
            problems.append(f"{i}: embedding differs")
    return problems


def migrate(store: Path, src: str, dst: str, *, dry_run: bool = False,
            force: bool = False, service_stopped: bool = False,
            out=sys.stdout) -> int:
    def say(line: str = "") -> None:
        print(line, file=out)

    for ns in (src, dst):
        if not valid_namespace(ns):
            raise Refused(f"{ns!r} is not a valid namespace")
    if src == dst:
        raise Refused("source and target namespace are the same")
    if not (store / "chroma.sqlite3").is_file():
        raise Refused(f"no Chroma store at {store}")
    held = holders(store)
    if held and not service_stopped:
        who = ", ".join(f"pid {pid} ({comm})" for pid, comm in held)
        raise Refused(f"{store} is held open by {who}. Stop the service first "
                      f"(systemctl --user stop memory-rag), back up with scripts/backup.py, "
                      f"then pass --i-stopped-the-service")

    import chromadb

    client = chromadb.PersistentClient(path=str(store))
    existing = {c.name: c for c in client.list_collections()}
    plan = []
    for name in sorted(existing):
        parts = split_name(name, src)
        if parts:
            plan.append((name, target_name(*parts, dst)))
    ledger = _open_ledger(store)
    src_rows = _ledger_rows(ledger, src) if ledger else []
    dst_rows = _ledger_rows(ledger, dst) if ledger else []

    say(f"{'DRY RUN: ' if dry_run else ''}{src} -> {dst} in {store}")
    total = 0
    clashes = []
    for name, new in plan:
        n = client.get_collection(name).count()
        total += n
        clash = new in existing
        clashes += [new] if clash else []
        say(f"  {name} ({n}) -> {new}{'  [TARGET EXISTS]' if clash else ''}")
    say(f"{len(plan)} collection(s), {total} node(s); ledger: {len(src_rows)} row(s) to copy, "
        f"{len(dst_rows)} already under {dst}")
    if not plan:
        raise Refused(f"namespace {src!r} has no collections in {store}")
    if dry_run:
        return 0
    if clashes and not force:
        raise Refused(f"{len(clashes)} target collection(s) already exist; pass --force to "
                      f"replace them (the source is never touched)")

    failed = False
    for name, new in plan:
        if new in existing:
            client.delete_collection(new)  # --force: the target only
        source = client.get_collection(name)
        target = _create_like(client, source, new)
        rewritten = _copy(source, target, src, dst)
        problems = _verify(source, client.get_collection(new), src, dst)
        failed |= bool(problems)
        say(f"  {new}: {target.count()} node(s), {rewritten} metadata rewritten, "
            + ("verified" if not problems else "VERIFY FAILED: " + "; ".join(problems[:5])))

    if ledger:
        wanted = [(ts, _rescope(p, src, dst), ev, d) for ts, p, ev, d in src_rows]
        have = set(dst_rows)
        new_rows = [row for row in wanted if row not in have]
        with ledger:
            ledger.executemany(
                "INSERT INTO events (ts, project, event, detail) VALUES (?,?,?,?)", new_rows)
        after = set(_ledger_rows(ledger, dst))
        ledger.close()
        missing = [row for row in wanted if row not in after]
        failed |= bool(missing)
        say(f"  ledger: {len(new_rows)} row(s) copied, {len(after)} now under {dst}"
            + (f"  VERIFY FAILED: {len(missing)} missing" if missing else ", verified"))

    say("FAILED verification — the source is untouched; inspect, then rerun with --force"
        if failed else f"done: {len(plan)} collection(s) copied and verified; {src} untouched")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Copy a namespace's collections to another namespace")
    ap.add_argument("--store", required=True, help="Chroma store directory")
    ap.add_argument("--from", dest="src", default=LEGACY, help="source namespace (default legacy)")
    ap.add_argument("--to", dest="dst", required=True, help="target namespace, e.g. vps-host")
    ap.add_argument("--dry-run", action="store_true", help="print collection names and counts only")
    ap.add_argument("--force", action="store_true",
                    help="replace target collections / ledger rows that already exist")
    ap.add_argument("--i-stopped-the-service", dest="stopped", action="store_true",
                    help="run even though another process holds the store open")
    args = ap.parse_args(argv)
    try:
        return migrate(Path(args.store).expanduser(), args.src, args.dst,
                       dry_run=args.dry_run, force=args.force, service_stopped=args.stopped)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
