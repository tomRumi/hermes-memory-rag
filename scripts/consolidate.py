#!/usr/bin/env python3
"""LLM-assisted wiki consolidation for hermes-rag.

Drains the staging memory collection (rag_<slug>__memory) into the project's
wiki markdown files (~/.hermes/wikis/<project>/modules/<prefix>.md), using an
Ollama editor model with a hard pure-Python validation gate and git rollback.

Usage:
    python scripts/consolidate.py <project>            # dry-run: print plan
    python scripts/consolidate.py <project> --apply    # perform the merge

Safety:
- wiki dir is git-initialized if needed; pre-consolidation snapshot commit
  first, so a bad merge is `git checkout` away.
- validation gate (headings preserved, length ratio, note coverage, fence
  count) → any failure falls back to append-only '## Consolidated learnings'.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

WIKI_ROOT = Path(os.environ.get("HERMES_WIKI_ROOT", str(Path.home() / ".hermes" / "wikis")))
EDITOR_MODEL = os.environ.get("HERMES_RAG_EDITOR_MODEL", "granite4:3b")
OLLAMA_URL = os.environ.get("HERMES_RAG_OLLAMA_URL", "http://localhost:11434")
LEN_MIN, LEN_MAX = 0.60, 1.60


# ── staging access ────────────────────────────────────────────────────────


def _client():
    import chromadb

    store = os.environ.get("HERMES_RAG_STORE", str(Path.home() / "hermes-rag"))
    return chromadb.PersistentClient(path=store)


def _slugify(project: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", project.strip().lower()).strip("-")
    return slug or "default"


def _staging_nodes(project: str) -> dict[str, dict]:
    """id -> {text, kind, created} from the staging collection."""
    names = {
        "memory": f"rag_{_slugify(project)}__memory",
    }
    col = _client().get_or_create_collection(names["memory"])
    data = col.get(include=["documents", "metadatas"])
    nodes = {}
    for i, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        nodes[i] = {
            "text": (doc or "").strip(),
            "kind": str((meta or {}).get("kind", "learning")),
            "created": str((meta or {}).get("created", "")),
        }
    return nodes


def _delete_ids(project: str, ids: list[str]) -> None:
    names = {"memory": f"rag_{_slugify(project)}__memory"}
    col = _client().get_or_create_collection(names["memory"])
    col.delete(ids=ids)


# ── routing ───────────────────────────────────────────────────────────────


def _target_file(kind: str, wiki_dir: Path) -> Path | None:
    """'rag:gotcha' -> modules/rag.md; None when no resolvable module."""
    prefix = kind.split(":", 1)[0] if ":" in kind else ""
    if not prefix or not re.fullmatch(r"[a-z0-9_-]+", prefix):
        return None
    return wiki_dir / "modules" / f"{prefix}.md"


def route(nodes: dict[str, dict], wiki_dir: Path):
    """Group node ids by target file; the rest goes to manual routing."""
    by_target: dict[Path, list[str]] = defaultdict(list)
    manual: list[tuple[str, dict]] = []
    for nid, node in sorted(nodes.items(), key=lambda kv: kv[1]["created"]):
        target = _target_file(node["kind"], wiki_dir)
        if target is None:
            manual.append((nid, node))
        else:
            by_target[target].append(nid)
    return by_target, manual


# ── validation gate (pure Python, hard) ───────────────────────────────────


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _count_fences(text: str) -> int:
    return len(re.findall(r"^\s*```", text, flags=re.MULTILINE))


def validate(original: str, new: str, notes: list[str]) -> tuple[bool, list[str]]:
    problems = []
    orig_headings = re.findall(r"^## .*$", original, flags=re.MULTILINE)
    new_headings = set(re.findall(r"^## .*$", new, flags=re.MULTILINE))
    for h in orig_headings:
        if h not in new_headings:
            problems.append(f"heading lost: {h!r}")
    lo, hi = LEN_MIN * len(original), LEN_MAX * len(original)
    if not (lo <= len(new) <= hi):
        problems.append(f"length {len(new)} outside [{int(lo)},{int(hi)}] of {len(original)}")
    new_norm = _norm(new)
    for i, note in enumerate(notes):
        words = _norm(note).split(" ")
        found = False
        for start in range(len(words) - 7):
            frag = " ".join(words[start:start + 8])
            if frag in new_norm:
                found = True
                break
        if not found:
            problems.append(f"note {i} has no distinctive 8+ word substring in output")
    fences_orig, fences_new = _count_fences(original), _count_fences(new)
    if fences_orig != fences_new or fences_new % 2 != 0:
        problems.append(f"fence count {fences_orig}->{fences_new} (changed or odd)")
    return (not problems), problems


# ── LLM merge ─────────────────────────────────────────────────────────────


MERGE_PROMPT = """You are editing a project wiki markdown file. Weave the NEW LEARNINGS into the CURRENT FILE as follows:
- Place each learning into the most fitting existing '##' section, rewritten as natural prose that fits the section's voice.
- Remove only text that a learning directly contradicts. Keep everything else, including every '##' heading and all code fences.
- Output the COMPLETE new file content and nothing else: no commentary, no explanations, no extra headings beyond the file's own.

CURRENT FILE:
<<<FILE
{current}
FILE>>>

NEW LEARNINGS:
{notes}

Output the complete updated file now:"""


def llm_merge(current: str, notes: list[str]) -> str:
    from ollama import Client

    client = Client(host=OLLAMA_URL)
    resp = client.chat(
        model=EDITOR_MODEL,
        messages=[{
            "role": "user",
            "content": MERGE_PROMPT.format(
                current=current,
                notes="\n".join(f"- {n}" for n in notes),
            ),
        }],
        options={"temperature": 0.2},
    )
    return resp["message"]["content"].strip()


def fallback_merge(current: str, notes: list[str]) -> str:
    lines = [current.rstrip("\n"), "", "## Consolidated learnings", ""]
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines)


# ── git ───────────────────────────────────────────────────────────────────


def git(wiki_dir: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(wiki_dir), *args],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def ensure_git_repo(wiki_dir: Path) -> None:
    if not (wiki_dir / ".git").exists():
        git(wiki_dir, "init")
    git(wiki_dir, "add", "-A")
    r = subprocess.run(["git", "-C", str(wiki_dir), "status", "--porcelain"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        git(wiki_dir, "-c", "user.name=hermes-memory-rag", "-c",
            "user.email=hermes-memory-rag@local", "commit", "-m", "initial commit")


def git_commit(wiki_dir: Path, message: str) -> None:
    git(wiki_dir, "add", "-A")
    r = subprocess.run(
        ["git", "-C", str(wiki_dir), "-c", "user.name=hermes-memory-rag", "-c",
         "user.email=hermes-memory-rag@local", "commit", "-m", message],
        capture_output=True, text=True,
    )
    # rc 1 with a clean tree ("nothing to commit") is fine — e.g. the
    # pre-snapshot can be a no-op right after the initial commit.
    if r.returncode != 0 and "nothing to commit" not in r.stdout:
        raise RuntimeError(f"git commit: {(r.stderr or r.stdout).strip()}")


# ── main ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="hermes-rag wiki consolidation")
    ap.add_argument("project")
    ap.add_argument("--apply", action="store_true", help="perform the merge")
    args = ap.parse_args(argv)

    project = args.project
    wiki_dir = WIKI_ROOT / project
    nodes = _staging_nodes(project)
    if not nodes:
        print(f"staging empty for [{project}] — nothing to consolidate")
        return 0

    by_target, manual = route(nodes, wiki_dir)
    if manual:
        print("MANUAL ROUTING (never auto-merged):")
        for nid, node in manual:
            print(f"  [{node['kind']}] {node['text'][:100]}{'…' if len(node['text']) > 100 else ''}")

    plan: list[tuple[Path, list[str], list[str]]] = []
    for target, ids in sorted(by_target.items()):
        notes = [nodes[i]["text"] for i in ids]
        plan.append((target, ids, notes))
        exists = target.exists()
        print(f"\nPLAN: {len(ids)} note(s) -> {target} ({'existing' if exists else 'MISSING — will create'})")
        for n in notes:
            print(f"  • {n[:100]}{'…' if len(n) > 100 else ''}")

    if not args.apply:
        print(f"\nDRY RUN — {len(nodes)} staged note(s), {len(plan)} target file(s), "
              f"{len(manual)} manual-routing. Re-run with --apply to merge.")
        return 0

    ensure_git_repo(wiki_dir)
    git_commit(wiki_dir, "pre-consolidation snapshot")

    merged_ids: list[str] = []
    print("\n── applying ──")
    for target, ids, notes in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            current = target.read_text(encoding="utf-8")
        else:
            # New module file: deterministic scaffold, no LLM, no gate needed.
            scaffold = [f"# {target.stem}", "", "## Consolidated learnings", ""]
            scaffold += [f"- {n}" for n in notes] + [""]
            target.write_text("\n".join(scaffold), encoding="utf-8")
            merged_ids.extend(ids)
            print(f"  {target.name}: created (scaffold, {len(ids)} note(s))")
            continue
        outcome = "merged"
        try:
            new = llm_merge(current, notes)
            ok, problems = validate(current, new, notes)
            if not ok:
                print(f"  GATE FAILED for {target.name}: {'; '.join(problems)}")
                print("  → falling back to append-only merge")
                new, outcome = fallback_merge(current, notes), "fallback"
        except Exception as exc:  # noqa: BLE001 — gate errors must not lose notes
            print(f"  EDITOR ERROR for {target.name}: {exc} → fallback")
            new, outcome = fallback_merge(current, notes), "fallback"
        target.write_text(new, encoding="utf-8")
        merged_ids.extend(ids)
        print(f"  {target.name}: {outcome} ({len(ids)} note(s))")

    git_commit(wiki_dir, f"consolidation: {len(merged_ids)} notes merged")

    _delete_ids(project, merged_ids)
    try:
        from hermes_memory_rag.server import ingest_wiki
        result = ingest_wiki(str(wiki_dir), project)
        print(f"wiki layer refreshed: {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: wiki layer refresh failed: {exc}")

    print(f"\nSUMMARY [{project}]: {len(merged_ids)} merged (LLM or fallback) "
          f"→ staging drained; {len(manual)} manual-routing kept; "
          f"{len(plan)} file(s) updated. Rollback: git -C {wiki_dir} revert/reset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
