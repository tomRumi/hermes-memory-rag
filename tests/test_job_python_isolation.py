"""The installed no-agent jobs must not import packages from their scheduler."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
UV = shutil.which("uv") or str(Path.home() / ".hermes/bin/uv")


def optional_python314():
    if not Path(UV).is_file():
        return None
    found = subprocess.run([UV, "python", "find", "3.14", "--no-python-downloads"],
                           capture_output=True, text=True, timeout=20)
    return found.stdout.strip() if found.returncode == 0 else None


class ScheduledPythonIsolation(unittest.TestCase):
    def test_installer_generates_scripts_without_pinned_interpreters(self):
        from scripts.install import Report, write_job_scripts
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            report = Report()
            write_job_scripts(home, report, False)
            self.assertFalse(report.failed())
            for script in ("memory_claude_import.sh", "memory_backup.sh", "memory_review.sh"):
                text = (home / "scripts" / script).read_text()
                self.assertNotIn("__", text)
                self.assertNotIn("old-server-venv", text)
                self.assertNotIn("/venv/bin/python", text)
                self.assertIn("python_runtime.sh", text)

    def run_job(self, job: str, scripts: dict[str, str], polluted: bool,
                args: tuple[str, ...] = (), pythonhome: bool = False,
                selected: str = PYTHON, expected_status: int | tuple[int, ...] = 0,
                expected_pending: tuple[bool, ...] = (), real_uv: bool = False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            home = root / "home"
            home.mkdir()
            repo = root / "repo"
            (repo / "scripts").mkdir(parents=True)
            (repo / "scripts/jobs").mkdir()
            shutil.copyfile(ROOT / "scripts/jobs/python_runtime.sh",
                            repo / "scripts/jobs/python_runtime.sh")
            for name, body in scripts.items():
                (repo / "scripts" / name).write_text(
                    "import os, sys\n"
                    "assert sys.version_info >= (3, 11)\n"
                    "assert sys.flags.isolated == 1\n"
                    "assert 'PYTHONPATH' not in os.environ, os.environ['PYTHONPATH']\n"
                    "assert 'PYTHONHOME' not in os.environ\n"
                    "assert sys.version_info[:2] == tuple(map(int, os.environ['EXPECTED_PYTHON_VERSION'].split('.')))\n"
                    f"{body}\n", encoding="utf-8")
            template = (ROOT / "scripts/jobs" / job).read_text(encoding="utf-8")
            wrapper = root / job
            for token, value in {
                "__REPO__": str(repo),
                "__HERMES_HOME__": str(home),
            }.items():
                template = template.replace(token, value)
            wrapper.write_text(template, encoding="utf-8")
            env = os.environ.copy()
            env["HOME"] = str(home)
            env["HERMES_RAG_PYTHON"] = selected
            if real_uv:
                self.assertTrue(Path(UV).is_file(), "uv integration prerequisite")
                env["HERMES_RAG_UV"] = UV
            else:
                # Fast seam for wrapper state tests; the real uv/Python path is
                # exercised by test_import_and_reindex_isolate_both_interpreters.
                fake_uv = root / "uv"
                fake_uv.write_text("#!/bin/sh\n"
                                   "while [ \"$#\" -gt 0 ]; do\n"
                                   "  case \"$1\" in\n"
                                   "    --python) selected=\"$2\"; shift 2;;\n"
                                   "    python) shift; exec \"$selected\" \"$@\";;\n"
                                   "    *) shift;;\n"
                                   "  esac\n"
                                   "done\nexit 2\n")
                fake_uv.chmod(0o755)
                env["HERMES_RAG_UV"] = str(fake_uv)
            env["HERMES_RAG_SERVER_DEP"] = str(ROOT)
            env["EXPECTED_PYTHON_VERSION"] = subprocess.run(
                [selected, "-I", "-c", "import sys; print('%s.%s' % sys.version_info[:2])"],
                capture_output=True, text=True).stdout.strip() if Path(selected).exists() else ""
            if polluted:
                poison = root / "python3.14-site-packages"
                poison.mkdir()
                (poison / "numpy.py").write_text("raise RuntimeError('alien NumPy')\n")
                env["PYTHONPATH"] = str(poison)
            else:
                env.pop("PYTHONPATH", None)
            if pythonhome:
                env["PYTHONHOME"] = str(root / "alien-python-home")
            else:
                env.pop("PYTHONHOME", None)
            statuses = (expected_status,) if isinstance(expected_status, int) else expected_status
            output = ""
            for index, status in enumerate(statuses):
                result = subprocess.run(["bash", str(wrapper), *args], cwd=ROOT, env=env,
                                        text=True, capture_output=True, timeout=30)
                output += result.stdout + result.stderr
                self.assertEqual(result.returncode, status, result.stdout + result.stderr)
                if expected_pending:
                    self.assertEqual((home / ".hermes/memory-backup/claude-reindex.pending").exists(),
                                     expected_pending[index], f"pending after run {index + 1}: {output}")
            log_path = home / ".hermes/memory-backup/claude-import.log"
            return output, log_path.read_text() if log_path.exists() else ""

    def test_import_and_reindex_isolate_both_interpreters(self):
        if not Path(UV).is_file():
            self.skipTest("uv required for runtime integration")
        scripts = {
            "claude_memory_import.py": "import numpy; print('wiki pages added: 1\\nwiki pages updated: 0\\nfacts added: 0')",
            "reindex_wiki.py": "import numpy, hermes_memory_rag; print('fixture: 1 page(s) -> indexed')",
        }
        for selected in filter(None, (PYTHON, optional_python314())):
            with self.subTest(selected=selected):
                output, log = self.run_job("memory_claude_import.sh", scripts, True,
                                           selected=selected, real_uv=True)
                self.assertIn("Indexed:", output)
                self.assertIn("fixture: 1 page(s) -> indexed", log)

    def test_unchanged_import_stays_quiet_and_skips_reindex(self):
        scripts = {
            "claude_memory_import.py": "print('wiki pages added: 0\\nwiki pages updated: 0\\nfacts added: 0')",
            "reindex_wiki.py": "raise AssertionError('unchanged pages should not reindex')",
        }
        for polluted in (False, True):
            with self.subTest(polluted=polluted):
                output, log = self.run_job("memory_claude_import.sh", scripts, polluted)
                self.assertEqual(output, "")
                self.assertIn("wiki pages added: 0", log)
                self.assertNotIn("Indexed:", log)

    def test_reindex_only_recovers_previously_copied_pages_without_import(self):
        scripts = {
            "claude_memory_import.py": "raise AssertionError('do not import a second time')",
            "reindex_wiki.py": "print('fixture: 1 page(s) -> indexed')",
        }
        for polluted in (False, True):
            with self.subTest(polluted=polluted):
                output, log = self.run_job("memory_claude_import.sh", scripts, polluted,
                                           ("--reindex-only",))
                self.assertIn("fixture: 1 page(s) -> indexed", output)
                self.assertIn("fixture: 1 page(s) -> indexed", log)

    def test_failed_reindex_retries_next_scheduled_run_even_when_import_unchanged(self):
        scripts = {
            "claude_memory_import.py": (
                "from pathlib import Path\n"
                "flag = Path(os.environ['HOME']) / 'imported'\n"
                "changed = not flag.exists()\n"
                "flag.touch()\n"
                "print(f'wiki pages added: {int(changed)}\\nwiki pages updated: 0\\nfacts added: 0')"
            ),
            "reindex_wiki.py": (
                "from pathlib import Path\n"
                "flag = Path(os.environ['HOME']) / 'index-attempted'\n"
                "if not flag.exists():\n"
                "    flag.touch()\n"
                "    raise RuntimeError('index unavailable')\n"
                "print('fixture: 1 page(s) -> indexed')"
            ),
        }
        output, log = self.run_job("memory_claude_import.sh", scripts, False,
                                   expected_status=(1, 0, 0),
                                   expected_pending=(True, False, False))
        self.assertIn("index unavailable", output)
        self.assertEqual(log.count("fixture: 1 page(s) -> indexed"), 1)
        self.assertIn("Indexed:", output)

    def test_pending_is_not_cleared_by_empty_successful_indexer(self):
        scripts = {
            "claude_memory_import.py": (
                "from pathlib import Path\n"
                "flag = Path(os.environ['HOME']) / 'imported'\n"
                "changed = not flag.exists()\n"
                "flag.touch()\n"
                "print(f'wiki pages added: {int(changed)}\\nwiki pages updated: 0\\nfacts added: 0')"
            ),
            "reindex_wiki.py": (
                "from pathlib import Path\n"
                "flag = Path(os.environ['HOME']) / 'index-attempted'\n"
                "if not flag.exists():\n"
                "    flag.touch()\n"
                "    print('no wiki directory — nothing to do')\n"
                "else: print('fixture: 1 page(s) -> indexed')"
            ),
        }
        output, _ = self.run_job("memory_claude_import.sh", scripts, False,
                                 expected_status=(1, 0), expected_pending=(True, False))
        self.assertIn("No indexed pages reported", output)

    def test_backup_isolates_server_interpreter(self):
        scripts = {"backup.py": "print('store: 1 nodes\\nfacts: 0\\ngit: committed 1 changed path(s)')"}
        for polluted in (False, True):
            with self.subTest(polluted=polluted):
                output, _ = self.run_job("memory_backup.sh", scripts, polluted)
                self.assertIn("Memory backup: 1 path(s) changed", output)

    def test_review_isolates_server_interpreter(self):
        scripts = {"weekly_review.py": "print('# Memory review\\n## Notes worth reading')"}
        for polluted in (False, True):
            with self.subTest(polluted=polluted):
                output, _ = self.run_job("memory_review.sh", scripts, polluted)
                self.assertIn("# Memory review", output)

    def test_inherited_pythonhome_does_not_redirect_interpreter(self):
        scripts = {"backup.py": "print('store: 1 nodes\\nfacts: 0')"}
        self.run_job("memory_backup.sh", scripts, False, pythonhome=True)

    def test_unsupported_interpreter_fails_before_import_or_install(self):
        old_python = os.environ.get("HERMES_RAG_TEST_OLD_PYTHON", "/usr/bin/python3")
        if not Path(old_python).exists():
            self.skipTest("older Python unavailable")
        version = subprocess.run([old_python, "-I", "-c", "import sys; print(sys.version_info[:2] < (3, 11))"],
                                 capture_output=True, text=True, timeout=10)
        if version.returncode != 0 or version.stdout.strip() != "True":
            self.skipTest("older Python unavailable")
        scripts = {"claude_memory_import.py": "raise AssertionError('should not run')",
                   "reindex_wiki.py": "raise AssertionError('should not run')"}
        output, log = self.run_job("memory_claude_import.sh", scripts, False,
                                   selected=old_python, expected_status=2)
        self.assertIn("Python 3.11", output)
        self.assertEqual(log, "")

    def test_missing_interpreter_fails_clearly(self):
        output, _ = self.run_job("memory_claude_import.sh", {}, False,
                                 selected="/nonexistent/python", expected_status=2)
        self.assertIn("interpreter", output)


if __name__ == "__main__":
    unittest.main()
