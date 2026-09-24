#!/usr/bin/env python3
"""Score how well memory search finds the right thing.

A question list holds questions, each naming the note that should answer it. This
asks the real search paths — not a copy of them — and writes down the position the
right note came back in.

    the facts half   runs inside Hermes, because the fact search lives there
    the pages half   runs with the server's interpreter, because search over
                     pages and code uses the vector library

So there are two modes, each run by its own interpreter, and one merge step:

    <hermes python> scripts/recall_score.py --half facts   --questions q.json --out facts.json
    <server python> scripts/recall_score.py --half vectors --questions q.json --out vectors.json
    <any python>    scripts/recall_score.py --merge facts.json vectors.json

scripts/recall_score.sh does all three in order.

Question file: a list of {"half", "question", "marker", ...} entries. The marker is
a short piece of text that appears only in the note that should answer the question
— that is how the position is measured, so it is checked against real content and
not against a filename. See examples/questions.example.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# How many results to look at. The facts half is asked for the same number the
# assistant itself gets before each message, so the score reflects what it sees.
DEFAULT_TOP = 5
REPO_ROOT = Path(__file__).resolve().parents[1]


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()


def load_questions(path: Path, half: str | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    if half:
        questions = [q for q in questions if q.get("half") == half]
    return questions


def position_of(texts: list[str], marker: str) -> tuple[int | None, str]:
    """Where the marker first appears, 1-based. Also return what came first."""
    for index, text in enumerate(texts, start=1):
        if marker.lower() in (text or "").lower():
            return index, ""
    first = (texts[0] if texts else "").strip().replace("\n", " ")[:70]
    return None, first


def score_facts(questions: list[dict], top: int, db_path: Path) -> list[dict]:
    """Ask the fact store the way the assistant asks it before each message."""
    sys.path.insert(0, str(hermes_home() / "hermes-agent"))
    from plugins.memory.holographic.retrieval import FactRetriever
    from plugins.memory.holographic.store import MemoryStore

    store = MemoryStore(db_path=db_path)
    # The same shape the provider uses for its per-message lookup.
    retriever = FactRetriever(store, hrr_dim=1024)

    rows: list[dict] = []
    for q in questions:
        try:
            hits = retriever.search(q["question"], limit=top)
        except Exception as exc:  # noqa: BLE001
            rows.append({**q, "position": None, "note": f"search failed: {exc}"})
            continue
        contents = [h.get("content", "") for h in hits]
        position, first = position_of(contents, q["marker"])
        rows.append({**q, "position": position, "note": first, "returned": len(contents)})
    return rows


def _node_text(hit) -> str:
    node = getattr(hit, "node", hit)
    for attr in ("get_content", "text"):
        value = getattr(node, attr, None)
        if callable(value):
            return value() or ""
        if isinstance(value, str):
            return value
    return ""


def score_vectors(questions: list[dict], top: int, store_dir: Path) -> list[dict]:
    """Ask the page and code search the way a recall request does.

    The server's own functions are used rather than a query against the
    collection: it searches through the vector index, and going around that (a
    plain collection query) silently picks a different embedding model and
    measures something the user never experiences.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from hermes_memory_rag import server

    client = server._client()
    embed = server._embed_model()

    rows: list[dict] = []
    for q in questions:
        collection = q.get("collection")
        if not collection:
            rows.append({**q, "position": None, "note": "no collection named"})
            continue
        try:
            # Over-fetch, then drop superseded or retired notes, then take the
            # number the caller asked for — the same shape the recall path uses.
            hits = [h for h in server._search_collection(client, collection, q["question"],
                                                         top * 3, embed)
                    if server._is_live(h)][:top]
            documents = [_node_text(h) for h in hits]
        except Exception as exc:  # noqa: BLE001
            rows.append({**q, "position": None, "note": f"search failed: {exc}"})
            continue
        position, first = position_of(documents, q["marker"])
        rows.append({**q, "position": position, "note": first, "returned": len(documents)})
    return rows


def report(rows: list[dict], top: int) -> str:
    """A plain table, then the count that matters: right note came back first."""
    lines: list[str] = []
    for half in ("facts", "wiki", "code"):
        subset = [r for r in rows if r.get("half") == half]
        if not subset:
            continue
        lines.append("")
        lines.append(f"{half.upper()}  ({len(subset)} questions)")
        lines.append("  pos  what the question was           notes")
        for r in subset:
            pos = r.get("position")
            where = {1: "1st", 2: "2nd", 3: "3rd"}.get(pos, f"{pos}th") if pos else "not found"
            lines.append(f"  {where:<4} {r['question'][:34]:<34} {r.get('note','')[:40]}")
        first = sum(1 for r in subset if r.get("position") == 1)
        within = sum(1 for r in subset if r.get("position") and r["position"] <= top)
        lines.append(f"  -> first position: {first}/{len(subset)}   "
                     f"anywhere in top {top}: {within}/{len(subset)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score memory search against a question list")
    ap.add_argument("--questions", default="", help="question list (JSON)")
    ap.add_argument("--half", choices=["facts", "vectors"], default="",
                    help="facts = run inside Hermes; vectors = run with the server interpreter")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP, help=f"results to look at (default {DEFAULT_TOP})")
    ap.add_argument("--out", default="", help="write the results here as JSON")
    ap.add_argument("--merge", nargs="*", default=[], help="merge result files and print the score")
    ap.add_argument("--facts-db", default="", help="fact store (default: HERMES_HOME/memory_store.db)")
    ap.add_argument("--store", default="", help="store directory (default: ~/hermes-rag)")
    args = ap.parse_args(argv)

    if args.merge:
        rows: list[dict] = []
        for path in args.merge:
            rows.extend(json.loads(Path(path).read_text(encoding="utf-8")))
        print(report(rows, args.top))
        found = sum(1 for r in rows if r.get("position") == 1)
        print(f"\n{found} of {len(rows)} questions came back first.")
        return 0

    if not args.questions or not args.half:
        ap.error("--questions and --half are required unless --merge is used")

    questions = load_questions(Path(args.questions), None if args.half == "facts" else None)
    if args.half == "facts":
        questions = [q for q in questions if q.get("half") == "facts"]
        db = Path(args.facts_db).expanduser() if args.facts_db else hermes_home() / "memory_store.db"
        rows = score_facts(questions, args.top, db)
    else:
        questions = [q for q in questions if q.get("half") in ("wiki", "code")]
        store = Path(args.store).expanduser() if args.store else Path.home() / "hermes-rag"
        rows = score_vectors(questions, args.top, store)

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(report(rows, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())