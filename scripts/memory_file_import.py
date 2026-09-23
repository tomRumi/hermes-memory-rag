#!/usr/bin/env python3
"""Load existing MEMORY.md / USER.md entries into the holographic fact store.

The memory provider mirrors writes that happen while it is active, but everything
already sitting in those files predates it. Without this the agent starts with an
empty fact store and the standing rules it has been following are simply not there.

Idempotent: facts are keyed by content, so running it twice adds nothing.

Must run with the interpreter that has Hermes on its path:

    ~/.hermes/hermes-agent/venv/bin/python scripts/memory_file_import.py
    ~/.hermes/hermes-agent/venv/bin/python scripts/memory_file_import.py --home ~/.hermes/profiles/work

Exit codes: 0 = done (including "nothing to do"), 2 = Hermes could not be found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DELIMITER = "\n§\n"


def hermes_home(explicit: str = "") -> Path:
    """The Hermes home to import into: --home, else $HERMES_HOME, else ~/.hermes."""
    import os
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()


def import_store_module(home: Path):
    """Import the provider's own store class.

    Reaching into Hermes rather than re-implementing its schema keeps this honest:
    if their schema changes, this fails loudly instead of writing rows nothing reads.
    """
    for candidate in (home / "hermes-agent", Path.home() / ".hermes" / "hermes-agent"):
        if (candidate / "plugins" / "memory" / "holographic" / "store.py").exists():
            sys.path.insert(0, str(candidate))
            from plugins.memory.holographic.store import MemoryStore  # type: ignore
            return MemoryStore
    return None


def entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [e.strip() for e in path.read_text(encoding="utf-8").split(DELIMITER) if e.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Load memory-file entries into the fact store")
    ap.add_argument("--home", default="", help="Hermes home (default: $HERMES_HOME or ~/.hermes)")
    ap.add_argument("--db", default="", help="fact database (default: <home>/memory_store.db)")
    args = ap.parse_args(argv)

    home = hermes_home(args.home)
    MemoryStore = import_store_module(home)
    if MemoryStore is None:
        print("could not find the Hermes install (expected <home>/hermes-agent)", file=sys.stderr)
        return 2

    db = Path(args.db).expanduser() if args.db else home / "memory_store.db"
    store = MemoryStore(db_path=db)

    before = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    offered = 0
    for filename, category in (("MEMORY.md", "general"), ("USER.md", "user_pref")):
        for entry in entries(home / "memories" / filename):
            store.add_fact(entry, category=category)
            offered += 1
            print(f"  [{category}] {entry[:72].replace(chr(10), ' ')}")

    after = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    print(f"\n{home}")
    print(f"  entries read: {offered}")
    print(f"  facts before: {before}   after: {after}   new: {after - before}")
    if offered and after == before:
        print("  (already imported — nothing to do)")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())