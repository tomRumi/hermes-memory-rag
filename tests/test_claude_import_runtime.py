"""The fact importer must load the installed provider without Hermes's venv."""
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
IMPORTER = ROOT / "scripts/claude_memory_import.py"


class ClaudeImportRuntime(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "hermes"
        self.memory = self.root / "claude" / "-sample-project" / "memory"
        self.memory.mkdir(parents=True)
        (self.memory / "feedback-work.md").write_text("Prefer precise, verifiable claims.")
        self.source = self.root / "hermes-source"
        self.source.mkdir()
        (self.source / "hermes_state_wal.py").write_text("# test source\n")

    def invoke(self, python=None, with_numpy=False, source=None, uv=None):
        python = python or sys.executable
        cmd = ([str(uv), "run", "--no-project", "--no-config", "--no-python-downloads",
                "--python", python, "--with", "numpy", "python", "-I"] if with_numpy
               else [python, "-I"])
        return subprocess.run([*cmd, str(IMPORTER), "--home", str(self.home),
                               "--claude-root", str(self.memory.parent.parent), "--apply"],
                              cwd=ROOT, capture_output=True, text=True, timeout=120,
                              env={**os.environ, "HOME": str(self.root),
                                   "HERMES_RAG_HERMES_SOURCE": str(source or self.source)})

    def test_missing_provider_never_silently_skips_feedback(self):
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("holographic", result.stderr + result.stdout)
        self.assertFalse((self.home / "memory_store.db").exists())

    def test_feedback_uses_the_preflight_snapshot(self):
        from scripts import claude_memory_import as importer

        class Store:
            offered = []

            def __init__(self, db_path):
                self._conn = sqlite3.connect(":memory:")
                self._conn.execute("CREATE TABLE facts (content TEXT)")

            def add_fact(self, content, category):
                self.offered.append((content, category))
                self._conn.execute("INSERT INTO facts VALUES (?)", (content,))

            def close(self):
                self._conn.close()

        source = self.memory / "feedback-work.md"
        note = self.memory / "notes.md"
        note.write_text("Initial project knowledge.\n")
        def change_after_preflight(home):
            source.write_text("Changed while importing.")
            note.write_text("Changed project knowledge.\n")
            return Store

        with patch.object(importer, "import_store_module", side_effect=change_after_preflight):
            self.assertEqual(importer.main(["--home", str(self.home), "--claude-root",
                                            str(self.memory.parent.parent), "--apply"]), 0)
        self.assertEqual(Store.offered, [("Prefer precise, verifiable claims.", "user_pref")])
        self.assertEqual((self.home / "wikis/project/imported/notes.md").read_text(),
                         "Initial project knowledge.\n")

    def test_profile_home_uses_shared_source_and_plugin_but_own_database(self):
        shared = self.root / ".hermes"
        profile = shared / "profiles" / "work"
        core = shared / "hermes-agent"
        core.mkdir(parents=True)
        (core / "hermes_state_wal.py").write_text("# fixture\n")
        plugin = shared / "plugins" / "holographic"
        plugin.mkdir(parents=True)
        (plugin / "store.py").write_text(
            "import sqlite3\n"
            "class MemoryStore:\n"
            "    def __init__(self, db_path):\n"
            "        self._conn = sqlite3.connect(db_path)\n"
            "        self._conn.execute('CREATE TABLE facts (content TEXT)')\n"
            "    def add_fact(self, content, category):\n"
            "        self._conn.execute('INSERT INTO facts VALUES (?)', (content,))\n"
            "        self._conn.commit()\n"
            "    def close(self): self._conn.close()\n")
        profile.mkdir(parents=True)
        env = {**os.environ, "HOME": str(self.root), "HERMES_HOME": str(profile)}
        env.pop("HERMES_RAG_HERMES_SOURCE", None)
        result = subprocess.run([sys.executable, "-I", str(IMPORTER), "--claude-root",
                                 str(self.memory.parent.parent), "--apply"], cwd=ROOT,
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with sqlite3.connect(profile / "memory_store.db") as db:
            self.assertEqual(db.execute("SELECT count(*) FROM facts").fetchone()[0], 1)
        self.assertFalse((shared / "memory_store.db").exists())
        # An explicit source override remains authoritative, not a fallback hint.
        env["HERMES_RAG_HERMES_SOURCE"] = str(self.root / "missing-source")
        failure = subprocess.run([sys.executable, "-I", str(IMPORTER), "--claude-root",
                                  str(self.memory.parent.parent), "--apply"], cwd=ROOT,
                                 env=env, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("Hermes source helpers missing", failure.stderr)

    def test_provider_import_and_fact_store_under_selected_pythons(self):
        uv = shutil.which("uv") or str(Path.home() / ".hermes/bin/uv")
        source = Path(os.environ.get("HERMES_RAG_TEST_HERMES_SOURCE", str(Path.home() / ".hermes/hermes-agent")))
        provider = Path(os.environ.get("HERMES_RAG_TEST_PROVIDER", str(Path.home() / ".hermes/plugins/holographic")))
        if not (Path(uv).is_file() and (source / "hermes_state_wal.py").is_file()
                and all((provider / name).is_file() for name in ("store.py", "text.py", "holographic.py"))):
            self.skipTest("installed uv, Hermes source and holographic provider required for integration")
        found = subprocess.run([uv, "python", "find", "3.14", "--no-python-downloads"],
                               capture_output=True, text=True, timeout=20)
        optional = [found.stdout.strip()] if found.returncode == 0 else []
        for python in (sys.executable, *optional):
            with self.subTest(python=python):
                self.home.mkdir(exist_ok=True)
                package = self.home / "plugins/holographic"
                package.mkdir(parents=True, exist_ok=True)
                for file in ("store.py", "text.py", "holographic.py"):
                    shutil.copyfile(provider / file, package / file)
                # Importing the provider entry module would require all Hermes dependencies.
                (package / "__init__.py").write_text("raise RuntimeError('do not import Hermes')\n")
                result = self.invoke(python, with_numpy=True, source=source, uv=uv)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("facts added:", result.stdout)
                with sqlite3.connect(self.home / "memory_store.db") as db:
                    self.assertEqual(db.execute("select count(*) from facts where content = ?",
                                                ("Prefer precise, verifiable claims.",)).fetchone()[0], 1)
