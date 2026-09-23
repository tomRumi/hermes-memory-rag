"""Backend routes for the memory window, mounted at /api/plugins/hermes-memory-rag/.

The desktop window cannot run shell commands itself, and it must not: the memory
server's libraries are deliberately kept out of Hermes's own environment. So the
window asks these routes to do the work, and these routes run the project's own
scripts in the environment they were installed into.

Everything here shells out. Nothing imports the vector library, because this file
runs inside the Hermes process.

Installed alongside this file: ``install.json``, written by the installer, saying
where the repository, the server's environment, the store and the backup live.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

PLUGIN_DIR = Path(__file__).resolve().parents[1]
INSTALL_FILE = PLUGIN_DIR / "install.json"
HOME = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()
TIMEOUT = 900


def install_info() -> dict:
    if not INSTALL_FILE.exists():
        raise HTTPException(status_code=503,
                            detail="not installed yet: install.json is missing")
    return json.loads(INSTALL_FILE.read_text(encoding="utf-8"))


def run_script(name: str, args: list[str], use_server_env: bool = True) -> dict:
    """Run one of the project's scripts and return what it said."""
    info = install_info()
    repo = Path(info["repo"])
    script = repo / "scripts" / name
    if not script.exists():
        raise HTTPException(status_code=500, detail=f"missing script: {script}")

    python = Path(info["venv"]) / "bin" / "python" if use_server_env else Path(sys.executable)
    if not python.exists():
        raise HTTPException(status_code=500, detail=f"missing interpreter: {python}")

    try:
        result = subprocess.run([str(python), str(script), *args],
                                capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"{name} did not finish within {TIMEOUT}s"}

    return {
        "ok": result.returncode == 0,
        "exit_code": result.returncode,
        "output": (result.stdout or "")[-4000:],
        "error": (result.stderr or "")[-2000:] if result.returncode != 0 else "",
    }


@router.get("/status")
def status() -> dict:
    """What the window shows: what is stored, and when it was last backed up."""
    info = install_info()
    store = Path(info["store"])
    backup_dir = Path(info["backup_dir"])

    payload: dict = {
        "store": str(store),
        "store_exists": store.exists(),
        "backup_dir": str(backup_dir),
        "push_enabled": False,
        "last_backup": None,
        "projects": [],
        "wiki_pages": 0,
        "facts": 0,
    }

    backup_config = backup_dir / "config.yaml"
    if backup_config.exists():
        for line in backup_config.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("push:"):
                payload["push_enabled"] = line.split(":", 1)[1].strip().lower() in (
                    "true", "yes", "on", "1")

    last_run = backup_dir / "last-run.json"
    if last_run.exists():
        try:
            payload["last_backup"] = json.loads(last_run.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload["last_backup"] = {"error": "unreadable"}

    facts_file = backup_dir / "facts.json"
    if facts_file.exists():
        try:
            payload["facts"] = len(json.loads(facts_file.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass

    wiki_root = HOME / "wikis"
    if wiki_root.is_dir():
        payload["wiki_pages"] = sum(1 for _ in wiki_root.rglob("*.md"))

    # Per-project counts come from the store, read with the server's interpreter
    # (this process has no vector library).
    python = Path(info["venv"]) / "bin" / "python"
    if python.exists() and store.exists():
        script = (
            "import chromadb, json, re\n"
            "c = chromadb.PersistentClient(path=%r)\n"
            "out = {}\n"
            "for col in c.list_collections():\n"
            "    m = re.match(r'^rag_(.+)__(code|wiki|memory)$', col.name)\n"
            "    if m:\n"
            "        out.setdefault(m.group(1), {})[m.group(2)] = col.count()\n"
            "print(json.dumps(out))\n"
        ) % str(store)
        try:
            result = subprocess.run([str(python), "-c", script],
                                    capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                counts = json.loads(result.stdout.strip().splitlines()[-1])
                payload["projects"] = [
                    {"project": name, **layers} for name, layers in sorted(counts.items())
                ]
        except Exception:  # noqa: BLE001 - the window must render even if this fails
            pass

    return payload


class PushSetting(BaseModel):
    enabled: bool


@router.post("/backup")
def backup_now() -> dict:
    """Write a backup and commit it. Pushes only if the setting allows it."""
    info = install_info()
    args = ["--dir", info["backup_dir"], "--store", info["store"]]
    if info.get("push_enabled"):
        args.append("--push")
    return run_script("backup.py", args)


@router.post("/push")
def set_push(setting: PushSetting) -> dict:
    """Turn pushing to a remote on or off. Off by default; the window toggles it."""
    info = install_info()
    backup_dir = Path(info["backup_dir"])
    backup_dir.mkdir(parents=True, exist_ok=True)
    config = backup_dir / "config.yaml"
    config.write_text(
        "# Toggled from the desktop window. Push stays off unless this says true.\n"
        f"push: {str(setting.enabled).lower()}\n", encoding="utf-8")
    return {"ok": True, "push_enabled": setting.enabled}


class PathOnly(BaseModel):
    path: str = ""


@router.post("/export")
def export(body: PathOnly) -> dict:
    """Write a copy of everything, readable, to a file the user picks."""
    info = install_info()
    destination = body.path or str(Path(info["backup_dir"]) / "manual-export.jsonl")
    return run_script("export_store.py", ["--out", destination, "--store", info["store"]])


@router.post("/import")
def import_store(body: PathOnly) -> dict:
    """Rebuild the store from an export. Emptying it first is the caller's choice."""
    info = install_info()
    if not body.path:
        raise HTTPException(status_code=400, detail="choose a file to import")
    source = Path(body.path)
    if not source.exists():
        raise HTTPException(status_code=400, detail=f"no such file: {source}")
    return run_script("restore_store.py",
                      ["--in", str(source), "--store", info["store"], "--force"])


@router.post("/claude-import")
def claude_import() -> dict:
    """Read Claude Code's memory files into the wiki and the fact store."""
    info = install_info()
    return run_script("claude_memory_import.py", ["--apply"], use_server_env=False)


@router.post("/review")
def review() -> dict:
    """Report what is accumulating and what has gone stale."""
    info = install_info()
    return run_script("weekly_review.py",
                      ["--no-apply", "--store", info["store"],
                       "--out", str(Path(info["backup_dir"]) / "review.md")])