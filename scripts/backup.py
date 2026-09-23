#!/usr/bin/env python3
"""Write everything worth keeping into one git repository.

What a backup has to contain to be a real one — each of these was a gap found by
actually trying to rebuild a store on another machine:

- every fact, as readable text plus **the numbers search compares**. Recomputing
  those numbers from the text does not reproduce the original search order for
  parts of the store that keep them resized, so they travel with the text.
- the wiki, which is the written knowledge and the part a human reads.
- MEMORY.md and USER.md for each profile, which are not in the store at all.
- the fact store, as JSON, so the episodic memory is portable too.
- the small files that make a rebuild possible: the project map and the wiring
  notes (which profiles had which server registered).

Plain text throughout, so the history is readable and a diff means something. The
raw Chroma database is deliberately *not* committed: it is binary, it does not
diff, and it can be rebuilt from the text plus the numbers.

Pushing to a remote is OFF unless the config says otherwise — that file is what
the desktop window toggles. The first run also refuses to push to a public repo.

    python scripts/backup.py                     # write, commit, do not push
    python scripts/backup.py --push              # push if the config allows it
    python scripts/backup.py --dir ~/.hermes/memory-backup --store ~/hermes-rag
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_STORE = Path(os.environ.get("HERMES_RAG_STORE", str(Path.home() / "hermes-rag")))
GIT_ID = ["-c", "user.name=hermes-memory-rag", "-c", "user.email=hermes-memory-rag@local"]


def git(directory: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(directory), *args],
                            capture_output=True, text=True)
    if check and result.returncode != 0 and "nothing to commit" not in result.stdout:
        raise RuntimeError(f"git {' '.join(args)}: {(result.stderr or result.stdout).strip()}")
    return result


def read_config(directory: Path) -> dict:
    path = directory / "config.yaml"
    if not path.exists():
        return {"push": False}
    config: dict = {"push": False}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" in line:
            key, _, value = line.partition(":")
            config[key.strip()] = value.strip().lower() in ("true", "yes", "1", "on")
    return config


def export_store(store: Path, destination: Path, repo_root: Path) -> dict:
    """Reuse the store exporter so both paths stay in step."""
    sys.path.insert(0, str(repo_root / "scripts"))
    from export_store import export  # type: ignore

    counts = {}
    nodes = export(store, destination / "store.jsonl", with_vectors=True)
    header = json.loads((destination / "store.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for name, meta in header.get("collections", {}).items():
        counts[name] = meta.get("count", 0)
    return {"nodes": nodes, "collections": counts}


def export_facts(home: Path, destination: Path) -> int:
    """The episodic facts, as JSON so they are readable and portable."""
    db = home / "memory_store.db"
    if not db.exists():
        return 0
    import sqlite3
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT content, category, trust_score, created_at FROM facts").fetchall()
    except sqlite3.OperationalError:
        rows = conn.execute("SELECT content, category FROM facts").fetchall()
    finally:
        conn.close()

    payload = [{"content": r[0], "category": r[1]} for r in rows]
    (destination / "facts.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(payload)


def copy_tree(source: Path, destination: Path) -> int:
    if not source.exists():
        return 0
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    return sum(1 for _ in destination.rglob("*") if _.is_file())


def copy_memory_files(home: Path, destination: Path) -> list[str]:
    """MEMORY.md / USER.md for the default profile and every named profile."""
    written = []
    candidates = [(home, "default")]
    profiles = home / "profiles"
    if profiles.is_dir():
        candidates += [(p, p.name) for p in sorted(profiles.iterdir()) if p.is_dir()]

    for profile_home, label in candidates:
        for filename in ("MEMORY.md", "USER.md"):
            source = profile_home / "memories" / filename
            if not source.exists():
                continue
            target = destination / "memories" / label / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            written.append(f"{label}/{filename}")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Back the memory up into a git repository")
    ap.add_argument("--dir", default=str(Path.home() / ".hermes" / "memory-backup"))
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    ap.add_argument("--home", default="", help="Hermes home (default: $HERMES_HOME or ~/.hermes)")
    ap.add_argument("--push", action="store_true", help="push if the config permits it")
    args = ap.parse_args(argv)

    home = Path(args.home).expanduser() if args.home else Path(
        os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()
    repo = Path(args.dir).expanduser()
    store = Path(args.store).expanduser()
    repo_root = Path(__file__).resolve().parents[1]
    started = time.time()

    repo.mkdir(parents=True, exist_ok=True)
    config = read_config(repo)
    (repo / "config.yaml").write_text(
        "# Toggled from the desktop window. Push stays off unless this says true.\n"
        f"push: {str(config.get('push', False)).lower()}\n", encoding="utf-8")

    summary: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if store.exists():
        summary["store"] = export_store(store, repo, repo_root)
        print(f"store: {summary['store']['nodes']} nodes from "
              f"{len(summary['store']['collections'])} collections")
    else:
        summary["store"] = {"error": f"no store at {store}"}
        print(f"store: MISSING at {store}")

    summary["facts"] = export_facts(home, repo)
    print(f"facts: {summary['facts']}")

    copied = copy_tree(home / "wikis", repo / "wiki")
    summary["wiki_files"] = copied
    print(f"wiki: {copied} files")

    memory_files = copy_memory_files(home, repo / "memories")
    summary["memory_files"] = memory_files
    print(f"memory files: {len(memory_files)} ({', '.join(memory_files) or 'none'})")

    # The project map is paths, and the wiring is a list of names. The Hermes
    # config is NOT copied: it holds credentials (server passwords, API keys), and
    # a backup repository that ends up on a remote must never carry those.
    projects = home / "projects.yaml"
    if projects.exists():
        shutil.copyfile(projects, repo / "hermes-projects.yaml")
    wiring = {"memory_provider": "", "profiles_with_server": []}
    try:
        import yaml  # type: ignore
        raw = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
        wiring["memory_provider"] = (raw.get("memory") or {}).get("provider", "")
        wiring["profiles_with_server"] = sorted((raw.get("mcp_servers") or {}).keys())
        profiles_dir = home / "profiles"
        if profiles_dir.is_dir():
            for profile in sorted(profiles_dir.iterdir()):
                cfg = profile / "config.yaml"
                if not cfg.is_file():
                    continue
                praw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
                wiring.setdefault("per_profile", {})[profile.name] = {
                    "server": sorted((praw.get("mcp_servers") or {}).keys()),
                    "memory_provider": (praw.get("memory") or {}).get("provider", ""),
                }
    except Exception as exc:  # noqa: BLE001 - a backup must not fail on this
        wiring["error"] = f"could not read the config: {exc}"
    (repo / "wiring.json").write_text(json.dumps(wiring, indent=1) + "\n", encoding="utf-8")
    print("wiring: project map copied, config reduced to names (no credentials)")

    # Commit
    if not (repo / ".git").exists():
        git(repo, "init", "-q")
        git(repo, "symbolic-ref", "HEAD", "refs/heads/main", check=False)
    git(repo, "add", "-A")
    status = git(repo, "status", "--porcelain").stdout.strip()
    if status:
        git(repo, *GIT_ID, "commit", "-q", "-m",
            f"memory backup {time.strftime('%Y-%m-%d %H:%M')}")
        changed = len(status.splitlines())
        print(f"git: committed {changed} changed path(s)")
        summary["committed"] = changed
    else:
        print("git: nothing changed since the last backup")
        summary["committed"] = 0

    # Push, only when allowed
    remote = git(repo, "remote", "get-url", "origin", check=False).stdout.strip()
    push_allowed = bool(config.get("push", False)) and args.push
    if push_allowed and remote:
        visibility = subprocess.run(["gh", "repo", "view", remote, "--json", "visibility"],
                                    capture_output=True, text=True)
        if visibility.returncode == 0 and "PUBLIC" in visibility.stdout.upper():
            print("push REFUSED: the remote is public. Memory must not be published.")
            summary["pushed"] = "refused-public-remote"
        else:
            git(repo, *GIT_ID, "push", "-q", "origin", "HEAD:main")
            print(f"git: pushed to {remote}")
            summary["pushed"] = remote
    elif not config.get("push", False):
        print("push: off (enable it in the desktop window)")
        summary["pushed"] = "off"
    elif not args.push:
        print("push: skipped (no --push flag)")
        summary["pushed"] = "skipped"
    elif not remote:
        print("push: no remote configured")
        summary["pushed"] = "no-remote"

    summary["seconds"] = round(time.time() - started, 1)
    (repo / "last-run.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"done in {summary['seconds']}s -> {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())