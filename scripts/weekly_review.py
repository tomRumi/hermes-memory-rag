#!/usr/bin/env python3
"""Weekly look at what memory is accumulating, and what has gone stale.

Four things, in order of how much they matter:

1. Inventory — what each project holds, and how many notes are waiting to be
   merged into its written pages.
2. Merge — for any project whose staging has reached the threshold, merge the
   notes into the wiki. The merge validates its own output before writing and
   commits the wiki beforehand, so a bad merge is one command away from undone.
   Use --no-apply to look without merging.
3. Candidates to replace — notes that are near duplicates of each other. Reported
   only, never actioned: deciding that two facts contradict each other is a
   judgement, and the point of the report is to bring the judgement to a human.
4. Names that no longer exist — pages that mention a collection or a store that is
   not in the store any more. This is how a rename leaves wrong details behind in
   the written knowledge, which is worse than leaving nothing.

Runs with the engine's interpreter (it needs the vector library):

    python scripts/weekly_review.py
    python scripts/weekly_review.py --no-apply
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

DEFAULT_STORE = Path(os.environ.get("HERMES_RAG_STORE", str(Path.home() / "hermes-rag")))
WIKI_ROOT = Path(os.environ.get("HERMES_WIKI_ROOT", str(Path.home() / ".hermes" / "wikis")))
COLLECTION_RE = re.compile(r"\brag_[a-z0-9-]+__[a-z]+\b")
LEGACY_STORE_RE = re.compile(r"~/[a-z0-9-]*chroma[a-z0-9-]*")
# Two notes this close are worth a human look. Deliberately conservative: a false
# candidate costs a sentence of reading, a missed one costs a stale wiki.
CANDIDATE_DISTANCE = 0.35


def projects(client) -> dict[str, dict[str, str]]:
    """{project slug: {layer: collection name}} from the collections present."""
    found: dict[str, dict[str, str]] = {}
    for collection in client.list_collections():
        match = re.match(r"^rag_(?P<slug>.+)__(?P<layer>code|wiki|memory)$", collection.name)
        if match:
            found.setdefault(match.group("slug"), {})[match.group("layer")] = collection.name
    return found


def inventory(client, slug: str, layers: dict[str, str]) -> dict:
    counts = {}
    for layer, name in layers.items():
        try:
            counts[layer] = client.get_or_create_collection(name).count()
        except Exception:
            counts[layer] = -1

    staged = active = 0
    if "memory" in layers:
        col = client.get_or_create_collection(layers["memory"])
        got = col.get(include=["metadatas"]) or {}
        for meta in got.get("metadatas") or []:
            meta = meta or {}
            if str(meta.get("kind") or "").startswith("memory"):
                continue
            if (meta.get("status") or "active") == "active":
                staged += 1
            elif (meta.get("status") or "") == "active":
                active += 1
    return {"counts": counts, "staged": staged}


def supersede_candidates(client, name: str, embed, limit: int = 8) -> list[tuple[str, str, float]]:
    """Pairs of active notes close enough to be worth reading side by side."""
    col = client.get_or_create_collection(name)
    got = col.get(include=cast(Any, ["documents", "metadatas", "embeddings"]))
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    vectors = got.get("embeddings")

    live = [(i, d, m) for i, d, m in zip(ids, docs, metas)
            if ((m or {}).get("status") or "active") == "active"]
    if vectors is None or len(live) < 2:
        return []

    def cosine(a, b) -> float:
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = sum(float(x) * float(x) for x in a) ** 0.5
        nb = sum(float(y) * float(y) for y in b) ** 0.5
        return 1.0 - (dot / (na * nb) if na and nb else 0.0)

    index = {node_id: vec for node_id, vec in zip(ids, vectors)}
    pairs = []
    for a in range(len(live)):
        for b in range(a + 1, len(live)):
            distance = cosine(index[live[a][0]], index[live[b][0]])
            if distance <= CANDIDATE_DISTANCE:
                pairs.append((live[a][1][:110], live[b][1][:110], round(distance, 4)))
    pairs.sort(key=lambda pair: pair[2])
    return pairs[:limit]


# A page that says a name was removed is being accurate, not stale. Without this
# the report would flag its own history every week and be ignored, which is worse
# than not reporting at all.
RETIRED_CONTEXT = re.compile(
    r"(\bremoved\b|\bretired\b|\bno longer\b|\bdeleted\b|\breplaced by\b|"
    r"\bpreviously\b|\bused to\b|\bdeprecated\b|\bis gone\b|\bmoved\b|\bformer\b)",
    re.IGNORECASE)


def stale_names(project_dir: Path, known: set[str]) -> list[tuple[str, str]]:
    """(file, name) for names on a page that do not exist and are not marked as gone.

    A name is only reported when the page presents it as current. Mentions
    accompanied by \"was removed\" / \"no longer\" / \"retired\" are history, and the
    history is worth keeping.
    """
    found = []
    if not project_dir.exists():
        return found
    for page in sorted(project_dir.rglob("*.md")):
        try:
            text = page.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        for number, line in enumerate(lines):
            # Look at the line and its neighbour: prose wraps, so the word that
            # says this name is gone often sits on the line above or below.
            window = " ".join(lines[max(0, number - 1):number + 2])
            if RETIRED_CONTEXT.search(window):
                continue
            for name in set(COLLECTION_RE.findall(line)) | set(LEGACY_STORE_RE.findall(line)):
                if name.startswith("rag_") and name in known:
                    continue
                found.append((str(page.relative_to(project_dir)), name))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Weekly memory review")
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    ap.add_argument("--wiki-root", default=str(WIKI_ROOT))
    ap.add_argument("--no-apply", action="store_true", help="report only; do not merge")
    ap.add_argument("--out", default="", help="write the report here as markdown")
    args = ap.parse_args(argv)

    import chromadb
    from llama_index.embeddings.ollama import OllamaEmbedding

    store = Path(args.store).expanduser()
    wiki_root = Path(args.wiki_root).expanduser()
    client = chromadb.PersistentClient(path=str(store))
    embedding_model = os.environ.get("HERMES_RAG_EMBED_MODEL", "nomic-embed-text")
    embed = OllamaEmbedding(model_name=embedding_model,
                            base_url=os.environ.get("HERMES_RAG_OLLAMA_URL",
                                                    "http://localhost:11434"))
    threshold = int(os.environ.get("HERMES_RAG_CONSOLIDATE_THRESHOLD", "10"))

    lines: list[str] = []
    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"# Memory review — {time.strftime('%Y-%m-%d %H:%M')}")
    emit()
    found = projects(client)
    if not found:
        emit("The store has no collections yet.")
        return 0

    known = {name for layers in found.values() for name in layers.values()}
    due: list[str] = []

    emit("## Inventory")
    emit()
    emit("| project | code | wiki | notes stored | staged for merge |")
    emit("|---|---|---|---|---|")
    for slug in sorted(found):
        info = inventory(client, slug, found[slug])
        counts = info["counts"]
        col = client.get_or_create_collection(found[slug]["memory"]) if "memory" in found[slug] else None
        stored = col.count() if col is not None else 0
        emit(f"| {slug} | {counts.get('code', 0)} | {counts.get('wiki', 0)} | {stored} | {info['staged']} |")
        if info["staged"] >= threshold:
            due.append(slug)
    emit()

    emit("## Merges")
    emit()
    if not due:
        emit(f"Nothing to merge (the threshold is {threshold} staged notes).")
    for slug in due:
        if args.no_apply:
            emit(f"- {slug}: {threshold}+ staged — would merge (report-only run)")
            continue
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("consolidate.py")), slug, "--apply"],
            capture_output=True, text=True)
        summary = [line for line in (result.stdout or "").splitlines() if "SUMMARY" in line]
        emit(f"- {slug}: {summary[0] if summary else 'merge attempted'}"
             + ("" if result.returncode == 0 else f" (exit {result.returncode})"))
    emit()

    emit("## Notes worth reading side by side")
    emit()
    emit("Reported only. Deciding that two notes disagree is a judgement, not a computation.")
    emit()
    any_candidates = False
    for slug in sorted(found):
        if "memory" not in found[slug]:
            continue
        pairs = supersede_candidates(client, found[slug]["memory"], embed)
        for first, second, distance in pairs:
            any_candidates = True
            emit(f"- **{slug}** (closeness {distance}):")
            emit(f"  - {first}")
            emit(f"  - {second}")
    if not any_candidates:
        emit("None found.")
    emit()

    emit("## Names that no longer exist")
    emit()
    flagged = 0
    for slug in sorted(found):
        project_dir = None
        for candidate in wiki_root.iterdir() if wiki_root.exists() else []:
            if re.sub(r"[^a-z0-9]+", "-", candidate.name.lower()).strip("-") == slug:
                project_dir = candidate
                break
        if project_dir is None:
            continue
        for page, name in stale_names(project_dir, known):
            flagged += 1
            emit(f"- `{name}` in {slug}/{page}")
    if not flagged:
        emit("None. Every collection and store named on a page still exists.")

    report = "\n".join(lines)
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())