"""Which folders a namespace may index.

The contract these tests hold to:

- a namespace with no entry anywhere is unrestricted, as before, so nothing changes
  until the operator writes one; an entry with an empty list indexes nothing;
- a path is judged by its real path, after links and `..` are resolved, and whole
  path components are compared (/a/bc is not under /a/b);
- the operator's namespaces.json is a ceiling: a token file's own folders count only
  where they lie inside it, and a namespace with a token file always has a list;
- the store, the token and policy directories, HERMES_RAG_FOLDER_DENY, `/` and the
  home directory are never allowed folders, and a walk from above never enters them;
- every index path checks first: ingest_code, ingest_wiki and reindex_wiki, and the
  operator's script, before anything is read or deleted; a refused or empty request
  leaves the existing layer as it was, and the message does not list the folders;
- a symlink inside a folder never leads out of it: directory links are not entered,
  a file link that leaves the root is skipped, and the skipped file is not indexed;
- an unusable policy file refuses every index request until it is fixed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from hermes_memory_rag import fsafe, server
from support import (as_token, collection_docs, collection_ids, digest, write_json,
                     write_token_file)

SECRET = "the launch codes are 0000"


@pytest.fixture
def tree(rag):
    """allowed/ (two pages and a code file), other/ and a secret outside both."""
    base = rag.tmp / "tree"
    (base / "allowed" / "docs").mkdir(parents=True)
    (base / "other").mkdir()
    (base / "allowed" / "a.md").write_text("# A\n\n## Setup\nhow acme is set up\n")
    (base / "allowed" / "docs" / "b.md").write_text("# B\n\n## Run\nhow acme is run\n")
    (base / "allowed" / "app.py").write_text("def main():\n    return 1\n")
    (base / "other" / "o.md").write_text("# O\n\n## Other\nsomething else\n")
    (base / "other" / "o.py").write_text("def other():\n    return 2\n")
    (base / "secret.md").write_text(f"# S\n\n## Secret\n{SECRET}\n")
    return base


def policy(rag, mapping, mode=0o600):
    write_json(rag.namespaces, {ns: {"allowed_folders": [str(f) for f in folders]}
                                for ns, folders in mapping.items()}, mode)


def index_all(path, project="p"):
    """Try every index path on `path`; returns what each raised (or returned)."""
    results = {}
    for name, call in (("ingest_wiki", lambda: server.ingest_wiki(str(path), project)),
                       ("ingest_code", lambda: server.ingest_code(str(path), project)),
                       ("reindex_wiki", lambda: server.reindex_wiki_blocking(str(path), project))):
        try:
            results[name] = call()
        except PermissionError as exc:
            results[name] = exc
    return results


def refused(results):
    return all(isinstance(r, PermissionError) and "outside the folders allowed" in str(r)
               for r in results.values())


class TestUnrestricted:
    def test_no_entry_means_no_limit(self, rag, tree):
        assert not any(isinstance(r, PermissionError) for r in index_all(tree / "other").values())
        assert "rag_acme.p__wiki" in {c.name for c in rag.client.list_collections()}

    def test_an_entry_for_another_namespace_does_not_restrict_this_one(self, rag, tree):
        policy(rag, {"globex": []})
        assert not any(isinstance(r, PermissionError) for r in index_all(tree / "other").values())


class TestAllowedFolders:
    def test_a_path_outside_the_list_is_refused_by_every_index_path(self, rag, tree):
        policy(rag, {"acme": [tree / "allowed"]})
        assert refused(index_all(tree / "other"))
        assert not rag.client.list_collections()

    def test_the_folder_itself_and_anything_below_it_is_fine(self, rag, tree):
        policy(rag, {"acme": [tree / "allowed"]})
        for target in (tree / "allowed", tree / "allowed" / "docs"):
            assert not any(isinstance(r, PermissionError) for r in index_all(target).values())

    def test_dot_dot_is_resolved_before_judging(self, rag, tree):
        policy(rag, {"acme": [tree / "allowed"]})
        assert refused(index_all(tree / "allowed" / ".." / "other"))
        assert refused(index_all(f"{tree}/allowed/docs/../../other"))

    def test_whole_components_are_compared(self, rag, tree):
        sibling = tree / "allowed-and-more"
        sibling.mkdir()
        (sibling / "x.md").write_text("# X\n\n## X\nprefix sibling\n")
        policy(rag, {"acme": [tree / "allowed"]})
        assert refused(index_all(sibling))

    def test_a_symlinked_allowed_folder_is_resolved(self, rag, tree):
        link = rag.tmp / "link-to-allowed"
        link.symlink_to(tree / "allowed")
        policy(rag, {"acme": [link]})
        for target in (tree / "allowed", link):
            assert not any(isinstance(r, PermissionError) for r in index_all(target).values())
        assert refused(index_all(tree / "other"))

    def test_a_symlink_that_leaves_the_list_is_refused(self, rag, tree):
        escape = tree / "allowed" / "escape"
        escape.symlink_to(tree / "other")
        policy(rag, {"acme": [tree / "allowed"]})
        assert refused(index_all(escape))

    def test_an_empty_list_indexes_nothing(self, rag, tree):
        policy(rag, {"acme": []})
        assert refused(index_all(tree / "allowed"))

    def test_the_message_does_not_list_the_folders(self, rag, tree):
        policy(rag, {"acme": [tree / "allowed"]})
        with pytest.raises(PermissionError) as raised:
            server.ingest_wiki(str(tree / "other"), "p")
        assert str(tree) not in str(raised.value)
        assert str(raised.value) == "path is outside the folders allowed for this namespace"

    def test_a_missing_path_does_not_tell_a_caller_what_exists(self, rag, tree):
        policy(rag, {"acme": [tree / "allowed"]})
        for target in (tree / "other", tree / "nowhere"):
            with pytest.raises(PermissionError) as raised:
                server.ingest_wiki(str(target), "p")
            assert str(raised.value) == "path is outside the folders allowed for this namespace"

    def test_the_policy_is_read_again_on_every_call(self, rag, tree):
        policy(rag, {"acme": [tree / "allowed"]})
        assert refused(index_all(tree / "other"))
        policy(rag, {"acme": [tree / "allowed", tree / "other"]})
        assert not any(isinstance(r, PermissionError) for r in index_all(tree / "other").values())
        policy(rag, {"acme": []})
        assert refused(index_all(tree / "other"))

    def test_the_ingest_right_is_still_required(self, rag, tree, monkeypatch):
        policy(rag, {"acme": [tree / "allowed"]})
        write_json(rag.tokens_json, {digest("tok"): {"namespace": "acme"}})
        monkeypatch.setenv("HERMES_RAG_TRANSPORT", "streamable-http")
        with as_token("tok"), pytest.raises(PermissionError, match="may not run ingest"):
            server.ingest_wiki(str(tree / "allowed"), "p")

    def test_an_env_var_can_move_the_policy_file(self, rag, tree, monkeypatch):
        elsewhere = rag.tmp / "elsewhere"
        elsewhere.mkdir(mode=0o700)
        write_json(elsewhere / "ns.json", {"acme": {"allowed_folders": [str(tree / "allowed")]}})
        monkeypatch.setenv("HERMES_RAG_NAMESPACES_FILE", str(elsewhere / "ns.json"))
        assert refused(index_all(tree / "other"))


class TestCeiling:
    def test_the_operator_list_caps_what_a_token_file_may_add(self, rag, tree, make_tokens_dir):
        policy(rag, {"acme": [tree / "allowed"]})
        write_token_file(make_tokens_dir, "acme", {"t": {"ingest": True}},
                         folders=[tree / "allowed" / "docs", tree / "other"])
        assert not any(isinstance(r, PermissionError)
                       for r in index_all(tree / "allowed" / "docs").values())
        assert refused(index_all(tree / "other"))
        assert refused(index_all(tree / "allowed"))  # the token file narrowed it to docs/

    def test_a_token_file_without_an_operator_entry_is_limited_by_its_own_list(
            self, rag, tree, make_tokens_dir):
        write_token_file(make_tokens_dir, "acme", {"t": {"ingest": True}}, folders=[tree / "allowed"])
        assert not any(isinstance(r, PermissionError) for r in index_all(tree / "allowed").values())
        assert refused(index_all(tree / "other"))

    @pytest.mark.parametrize("folders", [None, []])
    def test_a_namespace_with_a_token_file_defaults_to_deny(self, rag, tree, make_tokens_dir, folders):
        write_token_file(make_tokens_dir, "acme", {"t": {}}, folders=folders)
        assert refused(index_all(tree / "allowed"))

    def test_a_broken_token_file_does_not_lift_the_limit(self, rag, tree, make_tokens_dir):
        write_json(rag.tokens_json, {digest("tok"): {"namespace": "acme", "ingest": True}})
        write_json(make_tokens_dir / "acme.json", {"version": 7})
        assert refused(index_all(tree / "allowed"))

    def test_a_reserved_namespace_stays_unrestricted_by_a_stray_file(self, rag, tree, make_tokens_dir):
        write_json(make_tokens_dir / "vps-host.json", {"junk": True})
        os.environ["MEMORY_RAG_LOCAL_NAMESPACE"] = "vps-host"
        try:
            assert not any(isinstance(r, PermissionError) for r in index_all(tree / "other").values())
        finally:
            os.environ["MEMORY_RAG_LOCAL_NAMESPACE"] = "acme"


class TestNeverAllowed:
    @pytest.fixture(autouse=True)
    def _home(self, tree):
        self.tree = tree

    def test_unsafe_entries_are_ignored_not_honoured(self, rag, tree, monkeypatch):
        deny = rag.tmp / "denied"
        (deny / "inner").mkdir(parents=True)
        (deny / "inner" / "d.md").write_text("# D\n\n## D\nin the deny list\n")
        monkeypatch.setenv("HERMES_RAG_FOLDER_DENY", f"{deny}:")
        policy(rag, {"acme": ["/", Path.home(), rag.store, rag.cfg, rag.tokens_dir,
                              deny, deny / "inner"]})
        for target in ("/", Path.home(), rag.store, rag.cfg, deny, deny / "inner"):
            assert refused(index_all(target)), target

    def test_a_folder_inside_a_denied_place_is_ignored_even_with_a_good_sibling(self, rag, tree):
        policy(rag, {"acme": [rag.store / "inside", tree / "allowed"]})
        assert refused(index_all(rag.store))
        assert not any(isinstance(r, PermissionError) for r in index_all(tree / "allowed").values())

    def test_an_allowed_parent_does_not_reach_a_denied_place(self, rag, tree):
        policy(rag, {"acme": [rag.tmp]})  # contains the store and the config directory
        assert refused(index_all(rag.store))
        assert refused(index_all(rag.cfg))

    def test_a_walk_from_above_does_not_enter_protected_places(self, rag, tree):
        (rag.store / "leak.md").write_text("# L\n\n## L\nstore leak\n")
        (rag.cfg / "leak.md").write_text("# L\n\n## L\nconfig leak\n")
        # unrestricted namespace, walking the directory that holds store and config
        server.ingest_wiki(str(rag.tmp), "p")
        docs = " ".join(collection_docs(rag.client, "rag_acme.p__wiki"))
        assert "how acme is set up" in docs and "leak" not in docs
        server.ingest_code(str(rag.tmp), "p")
        assert "leak" not in " ".join(collection_docs(rag.client, "rag_acme.p__code"))


class TestNothingIsDeletedFirst:
    def test_a_refused_request_leaves_the_layers_alone(self, rag, tree):
        for call in (lambda: server.ingest_wiki(str(tree / "allowed"), "p"),
                     lambda: server.ingest_code(str(tree / "allowed"), "p"),
                     lambda: server.reindex_wiki_blocking(str(tree / "allowed"), "p")):
            call()
        before = {c.name: collection_ids(rag.client, c.name) for c in rag.client.list_collections()}
        assert before["rag_acme.p__wiki"] and before["rag_acme.p__code"]
        policy(rag, {"acme": [tree / "allowed"]})
        assert refused(index_all(tree / "other"))
        assert {c.name: collection_ids(rag.client, c.name)
                for c in rag.client.list_collections()} == before

    def test_an_empty_folder_does_not_wipe_a_layer(self, rag, tree):
        server.ingest_wiki(str(tree / "allowed"), "p")
        server.ingest_code(str(tree / "allowed"), "p")
        wiki, code = (collection_ids(rag.client, f"rag_acme.p__{x}") for x in ("wiki", "code"))
        empty = rag.tmp / "empty"
        empty.mkdir()
        assert server.ingest_wiki(str(empty), "p").startswith("no markdown")
        assert server.ingest_code(str(empty), "p", rebuild=True).startswith("no indexable files")
        assert collection_ids(rag.client, "rag_acme.p__wiki") == wiki
        assert collection_ids(rag.client, "rag_acme.p__code") == code

    def test_a_folder_of_empty_files_does_not_wipe_a_layer(self, rag, tree):
        server.ingest_code(str(tree / "allowed"), "p")
        code = collection_ids(rag.client, "rag_acme.p__code")
        blank = rag.tmp / "blank"
        blank.mkdir()
        (blank / "x.py").write_text("   \n")
        assert server.ingest_code(str(blank), "p").startswith("no indexable files")
        assert collection_ids(rag.client, "rag_acme.p__code") == code

    def test_a_missing_folder_is_reported_and_harmless(self, rag, tree):
        server.ingest_wiki(str(tree / "allowed"), "p")
        wiki = collection_ids(rag.client, "rag_acme.p__wiki")
        assert server.ingest_wiki(str(rag.tmp / "nowhere"), "p").startswith("no markdown")
        assert collection_ids(rag.client, "rag_acme.p__wiki") == wiki


class TestLinksNeverLeadOut:
    @pytest.fixture
    def trapped(self, rag, tree):
        root = tree / "allowed"
        (root / "file-link.md").symlink_to(tree / "secret.md")
        (root / "dir-link").symlink_to(tree / "other")
        (root / "inner-link.md").symlink_to(root / "a.md")  # inside the root: allowed
        (root / "file-link.py").symlink_to(tree / "other" / "o.py")
        return root

    def test_ingest_wiki_skips_what_leaves_the_root(self, rag, trapped):
        out = server.ingest_wiki(str(trapped), "p")
        docs = " ".join(collection_docs(rag.client, "rag_acme.p__wiki"))
        assert SECRET not in docs and "something else" not in docs
        assert "how acme is set up" in docs and "skipped" in out

    def test_ingest_code_skips_what_leaves_the_root(self, rag, trapped, tree):
        server.ingest_code(str(trapped), "p")
        docs = " ".join(collection_docs(rag.client, "rag_acme.p__code"))
        assert "def other" not in docs and "def main" in docs

    def test_reindex_wiki_skips_what_leaves_the_root(self, rag, trapped):
        report = json.loads(server.reindex_wiki_blocking(str(trapped), "p"))
        docs = " ".join(collection_docs(rag.client, "rag_acme.p__wiki"))
        assert SECRET not in docs and "something else" not in docs
        reasons = {s["path"]: s["reason"] for s in report["skipped"]}
        assert reasons["file-link.md"] == "escapes folder"
        assert reasons["dir-link"] == "symlinked directory"
        assert "inner-link.md" not in reasons

    def test_a_link_given_as_the_root_is_judged_by_where_it_points(self, rag, tree):
        policy(rag, {"acme": [tree / "allowed"]})
        link = tree / "allowed" / "to-other"
        link.symlink_to(tree / "other")
        assert refused(index_all(link))


class TestSafeFileAccess:
    def test_a_file_swapped_for_a_link_after_the_walk_is_not_read(self, tree):
        root = str(tree / "allowed")
        files, _ = fsafe.walk_files(root, suffix=".md")
        target = next(f for f in files if f.rel == "a.md")
        os.remove(target.real)
        os.symlink(tree / "secret.md", target.real)
        with pytest.raises(OSError):
            fsafe.read_confined(target.real, root)

    def test_a_file_swapped_for_a_pipe_is_not_waited_on(self, tree):
        root = str(tree / "allowed")
        files, _ = fsafe.walk_files(root, suffix=".md")
        target = next(f for f in files if f.rel == "a.md")
        os.remove(target.real)
        os.mkfifo(target.real)
        with pytest.raises(OSError):
            fsafe.read_confined(target.real, root)

    def test_a_path_outside_the_root_is_refused_on_its_own(self, tree):
        with pytest.raises(OSError):
            fsafe.read_confined(str(tree / "secret.md"), str(tree / "allowed"))

    def test_line_endings_are_normalised_as_read_text_did(self):
        assert fsafe.decode_text(b"a\r\nb\rc\n\xff") == "a\nb\nc\n�"

    def test_the_walk_is_ordered_and_skips_hidden_choices_to_the_caller(self, tree):
        files, skipped = fsafe.walk_files(str(tree / "allowed"), suffix=".md")
        assert [f.rel for f in files] == ["a.md", os.path.join("docs", "b.md")]
        assert skipped == []


class TestUnusablePolicy:
    @pytest.mark.parametrize("make", [
        lambda p: p.write_text("{not json"),
        lambda p: p.write_text(json.dumps(["list"])),
        lambda p: p.write_text(json.dumps({"Bad NS": {"allowed_folders": []}})),
        lambda p: p.write_text(json.dumps({"acme": {"allowed_folders": ["relative/path"]}})),
        lambda p: p.write_text(json.dumps({"acme": {"allowed_folders": "/abs"}})),
        lambda p: p.write_text(json.dumps({"acme": {"allowed_folders": [], "extra": 1}})),
    ])
    def test_a_malformed_file_refuses_every_index_request(self, rag, tree, make):
        make(rag.namespaces)
        rag.namespaces.chmod(0o600)
        with pytest.raises(PermissionError, match="folder policy file is unusable"):
            server.ingest_wiki(str(tree / "allowed"), "p")
        assert not rag.client.list_collections()

    def test_a_file_open_to_others_refuses_and_fixing_it_recovers(self, rag, tree):
        policy(rag, {"acme": [tree / "allowed"]}, mode=0o644)
        with pytest.raises(PermissionError, match="unusable"):
            server.ingest_wiki(str(tree / "allowed"), "p")
        rag.namespaces.chmod(0o600)
        assert "nodes" in server.ingest_wiki(str(tree / "allowed"), "p")

    def test_a_symlinked_file_is_refused(self, rag, tree):
        policy(rag, {"acme": [tree / "allowed"]})
        real = rag.cfg / "real.json"
        rag.namespaces.rename(real)
        rag.namespaces.symlink_to(real)
        with pytest.raises(PermissionError, match="unusable"):
            server.ingest_wiki(str(tree / "allowed"), "p")

    def test_a_policy_in_an_open_directory_is_refused(self, rag, tree):
        policy(rag, {"acme": [tree / "allowed"]})
        rag.cfg.chmod(0o755)
        try:
            with pytest.raises(PermissionError, match="unusable"):
                server.ingest_wiki(str(tree / "allowed"), "p")
        finally:
            rag.cfg.chmod(0o700)


def _reindex_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "reindex_wiki.py"
    spec = importlib.util.spec_from_file_location("reindex_wiki_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestOperatorScript:
    def test_the_script_is_held_to_the_namespace_policy(self, rag, tree, capsys):
        policy(rag, {"acme": [tree / "allowed"]})
        wikis = tree  # wiki-root with two project directories
        rc = _reindex_script().main(["--wiki-root", str(wikis), "--project", "other",
                                     "--namespace", "acme"])
        out = capsys.readouterr().out
        assert rc == 1 and "FAILED" in out and "outside the folders allowed" in out
        assert not rag.client.list_collections()
        rc = _reindex_script().main(["--wiki-root", str(wikis), "--project", "allowed",
                                     "--namespace", "acme"])
        assert rc == 0 and "rag_acme.allowed__wiki" in {c.name for c in rag.client.list_collections()}

    def test_the_incremental_flags_go_through_reindex_wiki(self, rag, tree, capsys):
        script = _reindex_script()
        args = ["--wiki-root", str(tree), "--project", "allowed", "--namespace", "acme"]
        assert script.main(args + ["--incremental", "--prune"]) == 0
        first = capsys.readouterr().out
        assert "indexed 2 page(s)" in first
        ids = collection_ids(rag.client, "rag_acme.allowed__wiki")
        assert script.main(args + ["--incremental", "--prune"]) == 0
        assert "indexed 0 page(s), 2 unchanged" in capsys.readouterr().out
        assert collection_ids(rag.client, "rag_acme.allowed__wiki") == ids
        assert script.main(args + ["--incremental", "--dry-run"]) == 0

    def test_incremental_options_need_incremental(self, rag, tree):
        with pytest.raises(SystemExit):
            _reindex_script().main(["--wiki-root", str(tree), "--prune"])
