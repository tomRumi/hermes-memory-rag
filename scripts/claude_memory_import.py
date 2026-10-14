#!/usr/bin/env python3
"""Bring Claude Code's own memory into this project's memory.

Claude Code keeps curated notes per project at
``~/.claude/projects/<slug>/memory/*.md`` — one topic per file. They are good
material: written by an agent that was working in the same repository, about the
same repository. This reads them and files them where they will actually be used.

Where each file goes, and why:

- files named ``feedback*`` are about how to work (a correction, a preference).
  Those go into the fact store, which is searched before every message, so they
  can influence behaviour without anyone asking for them.
- everything else is project knowledge, and goes to
  ``~/.hermes/wikis/<project>/imported/`` as a wiki page. It is then indexed with
  the rest of the wiki, which is where written knowledge is searched and where it
  can be read by a human.

Deliberately not read: the raw conversation transcripts next to these files
(hundreds of megabytes of logs), and repository ``CLAUDE.md`` files, which the
code layer already indexes.

Dry run by default. Idempotent: a file whose content has not changed is skipped.

The scheduled wrapper provisions NumPy with uv for the selected Python.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import re
import sys
import types
from pathlib import Path

DEFAULT_CLAUDE_ROOT = Path.home() / ".claude" / "projects"
FEEDBACK_FILE = re.compile(r"^feedback[-_]", re.IGNORECASE)


def hermes_home(explicit: str = "") -> Path:
    import os
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()


def slugify_path(path: str) -> str:
    """How Claude Code names a project directory.

    It replaces every character that is not a letter or a digit with a dash, so
    /Users/me/code/_proj becomes -Users-me-code--proj (note the doubled dash).
    Deriving the slug from a path this way is exact; going the other direction is
    not, since a dash in a name is indistinguishable from a separator.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", path.rstrip("/"))


def project_paths(home: Path, extra_paths: list[str]) -> dict[str, str]:
    """{slug: project name} from the projects file plus any paths given on the CLI."""
    pairs: list[tuple[str, str]] = []
    mapping = home / "projects.yaml"
    if mapping.exists():
        path = project = None
        for raw in mapping.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("- "):
                path = project = None
                line = line[2:].strip()
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip().strip("'\"")
            if key in ("path", "prefix", "dir", "root"):
                path = value
            elif key in ("project", "name", "slug"):
                project = value
            if path and project:
                pairs.append((path, project))
                path = project = None

    for raw in extra_paths:
        expanded = str(Path(raw).expanduser())
        pairs.append((expanded, Path(expanded).name))

    # Longest path first so a nested directory wins over its parent.
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return {slugify_path(path): project for path, project in pairs}


def find_projects(claude_root: Path) -> list[tuple[str, Path]]:
    if not claude_root.exists():
        return []
    found = []
    for child in sorted(claude_root.iterdir()):
        memory = child / "memory"
        if memory.is_dir() and any(memory.glob("*.md")):
            found.append((child.name, memory))
    return found


def import_store_module(home: Path):
    """Load the installed store's own files, not the Hermes app or its site-packages.

    Current Hermes discovers bundled plugins before user plugins; loading the
    provider entry module here would import the whole app and its private deps.
    A synthetic package lets store.py resolve its relative imports without that.
    """
    shared = Path.home() / ".hermes"
    override = os.environ.get("HERMES_RAG_HERMES_SOURCE")
    core = Path(override) if override else next(
        (candidate for candidate in (home / "hermes-agent", shared / "hermes-agent")
         if (candidate / "hermes_state_wal.py").is_file()), home / "hermes-agent")
    if not (core / "hermes_state_wal.py").is_file():
        raise RuntimeError(f"Hermes source helpers missing at {core}; cannot open holographic store")
    # Hermes source is pure Python here; never add its venv/site-packages. The
    # provider calls its own journal policy and migration helpers when opening.
    sys.path.insert(0, str(core))
    candidates = (home / "hermes-agent/plugins/memory/holographic",
                  home / "plugins/holographic",
                  shared / "hermes-agent/plugins/memory/holographic",
                  shared / "plugins/holographic")
    for candidate in candidates:
        if (candidate / "store.py").is_file():
            package = types.ModuleType("_claude_import_holographic")
            package.__path__ = [str(candidate)]
            sys.modules[package.__name__] = package
            return importlib.import_module(f"{package.__name__}.store").MemoryStore
    raise RuntimeError(f"holographic fact-store plugin not installed under {home}; "
                       "refusing to skip Claude feedback facts")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Import Claude Code's memory files")
    ap.add_argument("--claude-root", default=str(DEFAULT_CLAUDE_ROOT))
    ap.add_argument("--home", default="", help="Hermes home (default: $HERMES_HOME or ~/.hermes)")
    ap.add_argument("--project-path", action="append", default=[],
                    help="a repo path to map to a project (repeatable)")
    ap.add_argument("--apply", action="store_true", help="write the changes")
    args = ap.parse_args(argv)

    home = hermes_home(args.home)
    claude_root = Path(args.claude_root).expanduser()
    slugs = project_paths(home, args.project_path)
    wiki_root = home / "wikis"

    projects = find_projects(claude_root)
    if not projects:
        print(f"no Claude Code memory found under {claude_root} — nothing to do")
        return 0

    # Use one input snapshot for both the provider preflight and the writes.
    # A source can change or disappear while the provider is being opened.
    sources = [(slug, [(source, source.read_text(encoding="utf-8", errors="replace"))
                       for source in sorted(memory.glob("*.md"))])
               for slug, memory in projects]
    feedback = any(FEEDBACK_FILE.match(source.name) and body.strip()
                   for _, files in sources for source, body in files)
    MemoryStore = import_store_module(home) if feedback else None
    store = MemoryStore(db_path=home / "memory_store.db") if MemoryStore and args.apply else None

    facts_before = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] if store else 0
    facts_added = 0
    copied = updated = skipped = 0
    unmapped = []

    for slug, files in sources:
        project = slugs.get(slug)
        if project is None:
            unmapped.append(slug)
            project = slug.rsplit("-", 1)[-1].lower() or "imported"
        print(f"\n{slug}  ->  project {project!r}")
        target_dir = wiki_root / project / "imported"

        for source, body in files:
            # Compare and write the same bytes: comparing stripped text against
            # copied text made every file look changed, so every run rewrote all
            # of them and nothing was ever reported as unchanged.
            if not body.strip():
                continue
            if FEEDBACK_FILE.match(source.name):
                if store is not None:
                    store.add_fact(body.strip()[:400], category="user_pref")
                facts_added += 1
                print(f"   fact   {source.name}")
                continue

            destination = target_dir / source.name
            existed = destination.exists()
            if existed and destination.read_text(encoding="utf-8", errors="replace") == body:
                skipped += 1
                print(f"   keep   {source.name} (unchanged)")
                continue
            if args.apply:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(body, encoding="utf-8")
            if existed:
                updated += 1
                print(f"   update {source.name} -> {destination}")
            else:
                copied += 1
                print(f"   add    {source.name} -> {destination}")

    facts_after = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] if store else 0
    if store:
        store.close()

    print("\nsummary")
    print(f"  projects found:        {len(projects)}")
    print(f"  wiki pages added:      {copied}")
    print(f"  wiki pages updated:    {updated}")
    print(f"  unchanged (skipped):   {skipped}")
    print(f"  facts added:           {facts_after - facts_before} (offered {facts_added})")
    if unmapped:
        print(f"  unmapped project dirs: {len(unmapped)} -> {unmapped}")
        print("    (add them to projects.yaml to control the project name)")
    if not args.apply:
        print("\ndry run — nothing written. Re-run with --apply, then re-index the wiki.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())