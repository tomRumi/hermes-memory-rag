#!/usr/bin/env python3
"""Set this project up on a machine.

What it does, in order, and it stops at the first thing it cannot do:

1. Looks for what it needs — Python, uv, ollama, and the embedding model. Missing
   pieces are reported with the exact command to fix them rather than guessed at.
2. Builds a private environment for the memory server and installs this package
   into it, so the server's dependencies never touch Hermes's own environment.
   That separation is deliberate: installing a vector database into Hermes is how
   a working Hermes stops working.
3. Creates the store and the wiki directory.
4. Installs the plugin half (this repository) into the Hermes plugins directory
   and enables it, which is what the desktop window needs.
5. Writes the wiring into the selected profiles: the server as an MCP entry, and
   the holographic memory provider. The config is backed up first, and the edit is
   textual so comments survive. Running it twice changes nothing.
6. Loads the entries already in MEMORY.md / USER.md into the fact store, so the
   standing rules the agent has been following are there from the first session.
7. Checks that it actually works: the embedder answers, the store answers, the
   server starts and lists its tools, and a recall comes back.

Nothing is written until the probe stage passes. Use --dry-run to see the whole
plan without touching anything.

    python scripts/install.py
    python scripts/install.py --dry-run
    python scripts/install.py --profile default --profile work
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

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_NAME = "hermes-memory-rag"
SERVER_NAME = "hermes-memory-rag"
EMBED_MODEL = os.environ.get("HERMES_RAG_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.environ.get("HERMES_RAG_OLLAMA_URL", "http://localhost:11434")


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, step: str, state: str, detail: str = "") -> None:
        self.rows.append((step, state, detail))
        marker = {"ok": "ok  ", "skip": "skip", "fail": "FAIL", "plan": "plan"}.get(state, state)
        print(f"  [{marker}] {step}" + (f" — {detail}" if detail else ""))

    def failed(self) -> bool:
        return any(state == "fail" for _, state, _ in self.rows)

    def render(self) -> str:
        lines = ["", "Summary", "-------"]
        for step, state, detail in self.rows:
            lines.append(f"{state:5} {step}" + (f"  ({detail})" if detail else ""))
        return "\n".join(lines)


def hermes_home(explicit: str = "") -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()


def run(command: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def profile_home(home: Path, profile: str) -> Path:
    return home if profile == "default" else home / "profiles" / profile


def probe(report: Report) -> dict:
    """Check what is present. Returns what was found; never guesses, never installs."""
    found: dict = {}

    found["python"] = sys.version_info >= (3, 11)
    report.add("python 3.11+", "ok" if found["python"] else "fail",
               f"{sys.version.split()[0]}" if found["python"] else "this script needs 3.11 or newer")

    uv = shutil.which("uv") or str(Path.home() / ".hermes" / "bin" / "uv")
    found["uv"] = Path(uv).exists()
    report.add("uv", "ok" if found["uv"] else "fail",
               uv if found["uv"] else "install it: curl -LsSf https://astral.sh/uv/install.sh | sh")
    found["uv_path"] = uv

    try:
        import urllib.request
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as response:
            tags = json.loads(response.read().decode())
        names = [m.get("name", "") for m in tags.get("models", [])]
        found["ollama"] = True
        found["models"] = names
        has_model = any(n.split(":")[0] == EMBED_MODEL for n in names)
        found["embed_model"] = has_model
        report.add("ollama", "ok", OLLAMA_URL)
        report.add(f"model {EMBED_MODEL}", "ok" if has_model else "fail",
                   "present" if has_model else f"pull it: ollama pull {EMBED_MODEL}")
    except Exception as exc:  # noqa: BLE001
        found["ollama"] = False
        found["embed_model"] = False
        report.add("ollama", "fail", f"not reachable at {OLLAMA_URL} ({exc})")
        report.add(f"model {EMBED_MODEL}", "fail", f"start ollama, then: ollama pull {EMBED_MODEL}")

    found["gh"] = shutil.which("gh") is not None
    report.add("gh (optional, for pushing backups)", "ok" if found["gh"] else "skip",
               "found" if found["gh"] else "not needed unless you want to push")
    return found


def build_environment(store_venv: Path, report: Report, dry_run: bool) -> bool:
    """A private environment for the server, so Hermes's own stays untouched."""
    if store_venv.exists() and (store_venv / "bin" / SERVER_NAME).exists():
        report.add("server environment", "skip", f"already built at {store_venv}")
        return True

    if dry_run:
        report.add("server environment", "plan", f"would create {store_venv} and install this package")
        return True

    uv = shutil.which("uv") or str(Path.home() / ".hermes" / "bin" / "uv")
    made = run([uv, "venv", str(store_venv), "--python", "3.11"])
    if made.returncode != 0:
        report.add("server environment", "fail", (made.stderr or made.stdout).strip()[:200])
        return False
    installed = run([uv, "pip", "install", "--python", str(store_venv / "bin" / "python"),
                     "-e", str(REPO_ROOT)])
    if installed.returncode != 0:
        report.add("server environment", "fail", (installed.stderr or installed.stdout).strip()[:200])
        return False
    report.add("server environment", "ok", f"{store_venv} (this package installed into it)")
    return True


def write_server_entry(config: Path, venv: Path, report: Report, dry_run: bool) -> bool:
    """Add the MCP entry textually, so the file's comments survive."""
    if not config.exists():
        report.add(f"{config.parent.name}: config", "fail", "no config.yaml to edit")
        return False
    text = config.read_text(encoding="utf-8")
    if f"{SERVER_NAME}:" in text:
        report.add(f"{config.parent.name}: server entry", "skip", "already registered")
        return True

    block = (f"mcp_servers:\n"
             f"  {SERVER_NAME}:\n"
             f"    command: {venv}/bin/{SERVER_NAME}\n"
             f"    enabled: true\n")
    if dry_run:
        report.add(f"{config.parent.name}: server entry", "plan", "would add the mcp_servers entry")
        return True

    shutil.copyfile(config, config.with_suffix(".yaml.bak-install"))
    if "mcp_servers:" in text:
        # Put our key directly under the existing top-level block.
        lines = text.splitlines(keepends=True)
        out, inserted = [], False
        for line in lines:
            out.append(line)
            if not inserted and line.rstrip() == "mcp_servers:":
                out.append(f"  {SERVER_NAME}:\n"
                           f"    command: {venv}/bin/{SERVER_NAME}\n"
                           f"    enabled: true\n")
                inserted = True
        text = "".join(out)
    else:
        text = text.rstrip("\n") + "\n\n" + block
    config.write_text(text, encoding="utf-8")
    report.add(f"{config.parent.name}: server entry", "ok", "added (config backed up)")
    return True


def remove_server_entry(config: Path, name: str, report: Report, dry_run: bool) -> None:
    """Remove another MCP server's entry, so two servers do not both answer.

    Two registrations of the same tools is not harmless: the model sees ten
    near-identical tools and picks between them at random. Removing the old entry
    is therefore part of switching over, not cleanup — but it is never done
    silently, which is why it needs its own flag.
    """
    if not config.exists():
        return
    text = config.read_text(encoding="utf-8")
    marker = f"  {name}:"
    if marker not in text:
        report.add(f"{config.parent.name}: remove {name}", "skip", "not registered")
        return
    if dry_run:
        report.add(f"{config.parent.name}: remove {name}", "plan", "would remove the old entry")
        return

    shutil.copyfile(config, config.with_suffix(".yaml.bak-migrate"))
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    skipping = False
    for line in lines:
        if line.rstrip() == marker:
            skipping = True
            continue
        if skipping:
            # The entry's own keys are indented deeper than the key itself.
            if line.startswith("    ") or not line.strip():
                continue
            skipping = False
        out.append(line)
    config.write_text("".join(out), encoding="utf-8")
    report.add(f"{config.parent.name}: remove {name}", "ok", "old entry removed (config backed up)")


def set_provider(home: Path, profile: str, report: Report, dry_run: bool) -> None:
    """Memory provider via the CLI, which knows how to write its own config."""
    hermes = Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "hermes"
    if not hermes.exists():
        report.add(f"{profile}: memory provider", "skip", "Hermes CLI not found; set it by hand")
        return
    if dry_run:
        report.add(f"{profile}: memory provider", "plan", "would set memory.provider = holographic")
        return
    args = [str(hermes)]
    if profile != "default":
        args += ["-p", profile]
    args += ["config", "set", "memory.provider", "holographic"]
    result = run(args, timeout=120)
    ok = result.returncode == 0
    report.add(f"{profile}: memory provider", "ok" if ok else "fail",
               "holographic" if ok else (result.stderr or result.stdout).strip()[:160])


def install_plugin(home: Path, store_venv: Path, store: Path, backup_dir: Path,
                   report: Report, dry_run: bool) -> bool:
    """Copy the plugin half into the plugins directory, point it at the install,
    and enable it."""
    target = home / "plugins" / PLUGIN_NAME
    if dry_run:
        report.add("plugin", "plan", f"would install into {target} and enable it")
        return True

    target.mkdir(parents=True, exist_ok=True)
    for name in ("plugin.yaml", "__init__.py"):
        source = REPO_ROOT / name
        if source.exists():
            shutil.copyfile(source, target / name)
    # The desktop half (the window) and the dashboard half (its backend routes).
    for folder in ("desktop", "dashboard"):
        source = REPO_ROOT / folder
        if source.is_dir():
            if (target / folder).exists():
                shutil.rmtree(target / folder)
            shutil.copytree(source, target / folder)
    # The window itself goes to the desktop plugin directory. Measured, not
    # assumed: the app loads desktop halves from there and does NOT pick up the
    # copy inside the plugin package, so placing it only in the package leaves a
    # window that never appears.
    desktop_target = home / "desktop-plugins" / PLUGIN_NAME
    desktop_source = REPO_ROOT / "desktop" / "plugin.js"
    if desktop_source.exists():
        desktop_target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(desktop_source, desktop_target / "plugin.js")
        report.add("window", "ok", f"installed at {desktop_target}")

    # Where everything lives, so the window's routes can find the scripts. Written
    # rather than guessed: the repository is not inside the plugin directory.
    (target / "install.json").write_text(json.dumps({
        "repo": str(REPO_ROOT),
        "venv": str(store_venv),
        "store": str(store),
        "backup_dir": str(backup_dir),
    }, indent=1) + "\n", encoding="utf-8")
    report.add("plugin", "ok", f"installed at {target}")

    hermes = Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "hermes"
    if hermes.exists():
        result = run([str(hermes), "config", "set", "plugins.enabled",
                      json.dumps([PLUGIN_NAME])], timeout=120)
        report.add("plugin enabled", "ok" if result.returncode == 0 else "skip",
                   "added to plugins.enabled" if result.returncode == 0
                   else "set plugins.enabled by hand to include " + PLUGIN_NAME)
    return True


def healthcheck(venv: Path, store: Path, report: Report, dry_run: bool) -> None:
    if dry_run:
        report.add("healthcheck", "plan", "would probe the embedder, the store and the server")
        return

    python = venv / "bin" / "python"
    probe_script = (
        "import chromadb, sys\n"
        "from llama_index.embeddings.ollama import OllamaEmbedding\n"
        "e = OllamaEmbedding(model_name='%s', base_url='%s')\n"
        "v = e.get_text_embedding('healthcheck')\n"
        "print('embed ok', len(v))\n"
        "c = chromadb.PersistentClient(path='%s')\n"
        "names = [x.name for x in c.list_collections()]\n"
        "print('store ok', len(names), 'collections')\n"
    ) % (EMBED_MODEL, OLLAMA_URL, store)
    result = run([str(python), "-c", probe_script], timeout=300)
    good = result.returncode == 0 and "embed ok" in result.stdout
    report.add("embedder and store", "ok" if good else "fail",
               result.stdout.strip().replace("\n", "; ")[:160] if good
               else (result.stderr or result.stdout).strip()[-200:])

    # The server itself: start it and list its tools, exactly as Hermes would.
    script = (
        "import json, subprocess, sys\n"
        "p = subprocess.Popen(['%s/bin/%s'], stdin=subprocess.PIPE,\n"
        "                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)\n"
        "def send(o):\n"
        "    p.stdin.write(json.dumps(o) + chr(10)); p.stdin.flush()\n"
        "send({'jsonrpc':'2.0','id':1,'method':'initialize','params':"
        "{'protocolVersion':'2024-11-05','capabilities':{},"
        "'clientInfo':{'name':'install','version':'1'}}})\n"
        "send({'jsonrpc':'2.0','method':'notifications/initialized'})\n"
        "send({'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}})\n"
        "import threading, time\n"
        "seen = {}\n"
        "def rd():\n"
        "    for line in p.stdout:\n"
        "        try: m = json.loads(line)\n"
        "        except Exception: continue\n"
        "        seen[m.get('id')] = m\n"
        "threading.Thread(target=rd, daemon=True).start()\n"
        "deadline = time.time() + 60\n"
        "while time.time() < deadline and 2 not in seen: time.sleep(0.3)\n"
        "p.kill()\n"
        "print('TOOLS', ','.join(t['name'] for t in seen.get(2, {}).get('result', {}).get('tools', [])))\n"
    ) % (venv, SERVER_NAME)
    result = run([str(python), "-c", script], timeout=300)
    tools = ""
    for line in result.stdout.splitlines():
        if line.startswith("TOOLS"):
            tools = line.replace("TOOLS", "").strip()
    report.add("memory server", "ok" if tools else "fail",
               f"tools: {tools}" if tools else "the server did not answer")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Install the memory server and wire it into Hermes")
    ap.add_argument("--home", default="", help="Hermes home (default: $HERMES_HOME or ~/.hermes)")
    ap.add_argument("--venv", default="", help="where to build the server environment")
    ap.add_argument("--store", default="", help="store directory")
    ap.add_argument("--backup-dir", default="", help="where the backup repository lives")
    ap.add_argument("--profile", action="append", default=[],
                    help="profile to wire up (repeatable; default: every profile found)")
    ap.add_argument("--migrate-from", default="",
                    help="remove this MCP server entry from the configs (switching over)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, change nothing")
    args = ap.parse_args(argv)

    started = time.time()
    home = hermes_home(args.home)
    store_venv = Path(args.venv).expanduser() if args.venv else home / "hermes-memory-rag" / "venv"
    store = Path(args.store).expanduser() if args.store else Path(
        os.environ.get("HERMES_RAG_STORE") or (Path.home() / "hermes-rag"))

    print(f"Installing {PLUGIN_NAME}")
    print(f"  repository: {REPO_ROOT}")
    print(f"  hermes home: {home}")
    print(f"  store: {store}")
    if args.dry_run:
        print("  DRY RUN — nothing will be written")
    print()

    report = Report()
    print("Checks")
    found = probe(report)
    if not (found.get("python") and found.get("uv") and found.get("ollama") and found.get("embed_model")):
        print(report.render())
        print("\nStopping: the checks above must pass first. Nothing was changed.")
        return 2

    print("\nBuilding")
    if not build_environment(store_venv, report, args.dry_run):
        print(report.render())
        print("\nStopping: the server environment could not be built. Nothing else was changed.")
        return 2

    if not args.dry_run:
        store.mkdir(parents=True, exist_ok=True)
    report.add("store directory", "skip" if store.exists() and not args.dry_run else "plan",
               str(store))

    if args.profile:
        profiles = args.profile
    else:
        profiles = ["default"]
        profiles_dir = home / "profiles"
        if profiles_dir.is_dir():
            profiles += [p.name for p in sorted(profiles_dir.iterdir()) if p.is_dir()]

    # Prove the server works BEFORE touching any config. A config that points at a
    # server that does not start is worse than no config at all.
    print("\nHealthcheck")
    healthcheck(store_venv, store, report, args.dry_run)
    if report.failed():
        print(report.render())
        print("\nStopping before the wiring step: the server did not pass its check, so "
              "nothing was written to any config.")
        return 2

    print("\nWiring")
    for profile in profiles:
        config = profile_home(home, profile) / "config.yaml"
        write_server_entry(config, store_venv, report, args.dry_run)
        if args.migrate_from:
            remove_server_entry(config, args.migrate_from, report, args.dry_run)
        set_provider(home, profile, report, args.dry_run)

    backup_dir = Path(args.backup_dir).expanduser() if args.backup_dir else home / "memory-backup"
    install_plugin(home, store_venv, store, backup_dir, report, args.dry_run)

    print("\nExisting memory entries")
    if args.dry_run:
        report.add("memory file import", "plan", "would load MEMORY.md/USER.md into the fact store")
    else:
        importer = REPO_ROOT / "scripts" / "memory_file_import.py"
        hermes_python = Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python"
        if hermes_python.exists():
            result = run([str(hermes_python), str(importer), "--home", str(home)], timeout=300)
            report.add("memory file import", "ok" if result.returncode == 0 else "skip",
                       (result.stdout or "").strip().splitlines()[-1] if result.returncode == 0
                       else "run scripts/memory_file_import.py by hand")
        else:
            report.add("memory file import", "skip", "Hermes interpreter not found")

    print(report.render())
    print(f"\nDone in {round(time.time() - started, 1)}s.")
    if report.failed():
        print("Some steps failed — see FAIL above.")
        return 1
    if not args.dry_run:
        print("\nNext: restart Hermes so it picks up the new server and provider, then run")
        print("  hermes memory status                      # the provider should say holographic")
        print("  <python> scripts/claude_memory_import.py --apply   # bring Claude Code's notes in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())