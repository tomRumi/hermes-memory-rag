"""reindex_wiki: keep a project's wiki layer in step with a folder of pages.

The contract these tests hold to:

- the first run indexes every page; a second run embeds nothing; an edited page
  replaces its own chunks (stale ones are gone, untouched ones keep their vectors);
  a deleted or renamed page is pruned and stops being recalled, unless prune is off;
- ids are derived from folder, page, position and text, so runs are reproducible and a
  run cut short by its budget converges, over repeated calls, on the state one run
  gives;
- an unmounted or wrong folder (no pages while the index has some) is refused, and
  emptying the index takes mode=full, prune and allow_empty together;
- links that leave the folder are skipped and anything already indexed from such a
  path is pruned; `exclude` drops top-level folders, hidden folders are always dropped;
- an embedder failing part-way leaves the pages done before it indexed and the failing
  page's old nodes intact; the embedding happens before anything old is deleted;
- only nodes this tool wrote for this folder are touched, by either mode; pages that
  ingest_wiki indexed into the same layer stay;
- dry_run writes nothing; a second run for the same project while one is running
  answers busy; the folder policy, the ingest right and write access all apply, and
  namespaces never see each other;
- the tool is asynchronous (the event loop is not blocked) and reachable over MCP.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time

import pytest

from hermes_memory_rag import server
from support import (as_token, call, collection_docs, collection_ids, digest, snapshot,
                     write_json, write_token_file)

# Every page below has a title chunk and one section chunk: two nodes each.
PAGES = {
    "index.md": "# Home\n\n## Overview\nthe overview page explains the platform\n",
    "ops/deploy.md": "# Deploy\n\n## Steps\nrun the blue green deploy in ten minutes\n",
    "ops/backup.md": "# Backup\n\n## Nightly\nbackups run nightly at three\n",
}


@pytest.fixture
def wiki(rag):
    root = rag.tmp / "llm-wiki"
    for rel, text in PAGES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def reindex(path, project="llm-wiki", **kw):
    return json.loads(server.reindex_wiki_blocking(str(path), project, **kw))


def wiki_ids(rag, project="llm-wiki", ns="acme"):
    return collection_ids(rag.client, server._collection_names(project, ns)["wiki"])


def wiki_docs(rag, project="llm-wiki", ns="acme"):
    return " | ".join(collection_docs(rag.client, server._collection_names(project, ns)["wiki"]))


def recall(query, project="llm-wiki"):
    return server.recall(query, project=project, layer="wiki", top_wiki=10)


def embedded(rag):
    """Chunks embedded since the last call (the health check is not a chunk)."""
    calls = [c for c in rag.embed.calls if c != "healthcheck"]
    rag.embed.calls.clear()
    return calls


class TestFirstAndRepeatedRuns:
    def test_the_first_run_indexes_every_page(self, rag, wiki):
        report = reindex(wiki)
        assert report["files"] == {"seen": 3, "unchanged": 0, "indexed": 3, "removed": 0, "skipped": 0}
        assert report["nodes"] == {"added": 6, "deleted": 0}
        assert not report["partial"] and report["remaining"] == 0 and not report["dry_run"]
        assert len(wiki_ids(rag)) == 6
        assert "blue green deploy" in recall("deploy steps")

    def test_a_second_run_embeds_nothing_and_changes_nothing(self, rag, wiki):
        reindex(wiki)
        ids, before = wiki_ids(rag), snapshot(rag.client)
        embedded(rag)
        report = reindex(wiki)
        assert report["files"]["unchanged"] == 3 and report["files"]["indexed"] == 0
        assert report["nodes"] == {"added": 0, "deleted": 0}
        assert embedded(rag) == [] and wiki_ids(rag) == ids
        after = snapshot(rag.client)
        assert after["collections"] == before["collections"]

    def test_ids_are_deterministic_and_do_not_depend_on_the_project(self, rag, wiki):
        reindex(wiki, "one")
        reindex(wiki, "two")
        assert wiki_ids(rag, "one") == wiki_ids(rag, "two") != set()

    def test_the_project_defaults_to_the_folder_name(self, rag, wiki):
        assert reindex(wiki, project="")["project"] == "llm-wiki"
        assert wiki_ids(rag)

    def test_nodes_carry_the_folder_and_the_hash(self, rag, wiki):
        reindex(wiki)
        col = rag.client.get_collection("rag_acme.llm-wiki__wiki")
        metas = col.get(include=["metadatas"])["metadatas"]
        assert {m["source"] for m in metas} == {"reindex_wiki"}
        assert len({m["root_id"] for m in metas}) == 1
        assert all(len(m["file_sha"]) == 64 and m["layer"] == "wiki" and m["status"] == "active"
                   for m in metas)
        assert {m["path"] for m in metas} == {"index.md", "ops/deploy.md", "ops/backup.md"}


class TestEditsDeletesRenames:
    def test_an_edited_page_replaces_its_chunks_and_keeps_the_rest(self, rag, wiki):
        page = wiki / "ops" / "deploy.md"
        page.write_text("# Deploy\n\n## Steps\nrun the deploy\n\n## Rollback\nroll back with the button\n")
        reindex(wiki)
        embedded(rag)
        page.write_text("# Deploy\n\n## Steps\nrun the canary deploy\n\n## Rollback\nroll back with the button\n")
        report = reindex(wiki)
        assert report["files"]["indexed"] == 1 and report["files"]["unchanged"] == 2
        assert report["nodes"] == {"added": 1, "deleted": 1}
        docs = wiki_docs(rag)
        assert "canary deploy" in docs and "run the deploy" not in docs
        assert docs.count("roll back with the button") == 1
        assert len(embedded(rag)) == 1  # only the changed section was embedded
        assert "canary" in recall("deploy canary")

    def test_surviving_chunks_record_the_new_hash(self, rag, wiki):
        page = wiki / "ops" / "deploy.md"
        page.write_text("# D\n\n## One\nfirst\n\n## Two\nsecond\n")
        reindex(wiki)
        page.write_text("# D\n\n## One\nfirst\n\n## Two\nchanged\n")
        reindex(wiki)
        col = rag.client.get_collection("rag_acme.llm-wiki__wiki")
        got = col.get(where={"path": "ops/deploy.md"}, include=["metadatas"])["metadatas"]
        assert len({m["file_sha"] for m in got}) == 1

    def test_a_deleted_page_is_pruned_and_no_longer_recalled(self, rag, wiki):
        reindex(wiki)
        (wiki / "ops" / "backup.md").unlink()
        report = reindex(wiki)
        assert report["files"]["removed"] == 1 and report["nodes"]["deleted"] == 2
        assert "backups run nightly" not in wiki_docs(rag)
        assert "backups run nightly" not in recall("nightly backups")

    def test_without_prune_a_deleted_page_stays(self, rag, wiki):
        reindex(wiki)
        (wiki / "ops" / "backup.md").unlink()
        report = reindex(wiki, prune=False)
        assert report["files"]["removed"] == 0 and "backups run nightly" in wiki_docs(rag)
        assert reindex(wiki)["files"]["removed"] == 1  # and a later pruning run removes it

    def test_a_rename_is_a_removal_and_an_addition(self, rag, wiki):
        reindex(wiki)
        (wiki / "ops" / "backup.md").rename(wiki / "ops" / "backups.md")
        report = reindex(wiki)
        assert report["files"]["indexed"] == 1 and report["files"]["removed"] == 1
        col = rag.client.get_collection("rag_acme.llm-wiki__wiki")
        assert {m["path"] for m in col.get(include=["metadatas"])["metadatas"]} == {
            "index.md", "ops/deploy.md", "ops/backups.md"}

    def test_recall_finds_new_content_after_a_run(self, rag, wiki):
        reindex(wiki)
        (wiki / "new.md").write_text("# New\n\n## Gadget\nthe flux capacitor needs servicing\n")
        assert "needs servicing" not in recall("flux capacitor servicing")
        reindex(wiki)
        assert "needs servicing" in recall("flux capacitor servicing")


class TestGuards:
    def test_an_empty_folder_with_an_index_is_refused(self, rag, wiki):
        reindex(wiki)
        ids = wiki_ids(rag)
        empty = rag.tmp / "empty"
        empty.mkdir()
        for page in list(wiki.rglob("*.md")):
            page.unlink()
        with pytest.raises(ValueError, match="refusing: no readable markdown"):
            reindex(wiki)
        assert wiki_ids(rag) == ids
        for kw in ({"mode": "full"}, {"mode": "full", "prune": True}, {"allow_empty": True},
                   {"mode": "incremental", "prune": True, "allow_empty": True}):
            with pytest.raises(ValueError, match="refusing"):
                reindex(wiki, **kw)
        assert wiki_ids(rag) == ids

    def test_an_unmounted_folder_is_the_same_case(self, rag, wiki):
        reindex(wiki)
        ids = wiki_ids(rag)
        for page in wiki.rglob("*.md"):
            page.unlink()
        (wiki / "ops").rmdir()
        with pytest.raises(ValueError, match="refusing"):
            reindex(wiki)
        assert wiki_ids(rag) == ids

    def test_emptying_on_purpose_needs_full_prune_and_allow_empty(self, rag, wiki):
        reindex(wiki)
        for page in list(wiki.rglob("*.md")):
            page.unlink()
        report = reindex(wiki, mode="full", prune=True, allow_empty=True)
        assert report["files"]["removed"] == 3 and wiki_ids(rag) == set()

    def test_an_empty_folder_with_no_index_is_simply_nothing_to_do(self, rag):
        empty = rag.tmp / "empty-wiki"
        empty.mkdir()
        report = reindex(empty)
        assert report["files"]["seen"] == 0 and report["nodes"] == {"added": 0, "deleted": 0}
        assert not rag.client.list_collections()

    def test_a_missing_folder_is_an_error(self, rag):
        with pytest.raises(ValueError, match="is not a folder"):
            reindex(rag.tmp / "nowhere")

    @pytest.mark.parametrize("kw,message", [
        ({"mode": "fast"}, "mode must be"),
        ({"budget_seconds": -1}, "budget_seconds"),
        ({"budget_seconds": 99999}, "budget_seconds"),
        ({"exclude": "a/b"}, "top-level"),
        ({"exclude": ".."}, "top-level"),
    ])
    def test_bad_arguments_are_refused_before_any_work(self, rag, wiki, kw, message):
        with pytest.raises(ValueError, match=message):
            reindex(wiki, **kw)
        assert not rag.client.list_collections()


class TestWhatIsWalked:
    def test_exclude_drops_top_level_folders_only(self, rag, wiki):
        for rel in ("raw/source.md", "docs/raw/keep.md", ".obsidian/cfg.md", "ops/.git/hook.md"):
            (wiki / rel).parent.mkdir(parents=True, exist_ok=True)
            (wiki / rel).write_text(f"# T\n\n## S\ncontent of {rel}\n")
        report = reindex(wiki, exclude="raw, nothing-here")
        paths = {m["path"] for m in rag.client.get_collection("rag_acme.llm-wiki__wiki")
                 .get(include=["metadatas"])["metadatas"]}
        assert "raw/source.md" not in paths and "docs/raw/keep.md" in paths
        assert not any(p.startswith(".") or "/.git/" in p for p in paths)
        assert report["files"]["seen"] == 4

    def test_excluding_after_indexing_prunes_it(self, rag, wiki):
        (wiki / "raw").mkdir()
        (wiki / "raw" / "source.md").write_text("# R\n\n## R\nraw source text\n")
        reindex(wiki)
        assert "raw source text" in wiki_docs(rag)
        reindex(wiki, exclude="raw")
        assert "raw source text" not in wiki_docs(rag)

    def test_non_markdown_files_are_ignored(self, rag, wiki):
        (wiki / "notes.txt").write_text("not a page")
        (wiki / "image.md.png").write_text("not a page either")
        assert reindex(wiki)["files"]["seen"] == 3

    def test_a_link_out_of_the_folder_is_skipped_and_its_nodes_are_pruned(self, rag, wiki):
        secret = rag.tmp / "secret.md"
        secret.write_text("# S\n\n## S\nthe vault combination is 1234\n")
        reindex(wiki)
        (wiki / "index.md").unlink()
        (wiki / "index.md").symlink_to(secret)
        report = reindex(wiki)
        assert report["skipped"] == [{"path": "index.md", "reason": "escapes folder"}]
        assert "vault combination" not in wiki_docs(rag) and "overview page" not in wiki_docs(rag)
        assert report["files"]["skipped"] == 1 and report["files"]["removed"] == 1

    def test_unreadable_pages_are_skipped_and_reported(self, rag, wiki, monkeypatch):
        real = server.fsafe.read_confined

        def flaky(path, root):
            if path.endswith("backup.md"):
                raise PermissionError("no access")
            return real(path, root)
        monkeypatch.setattr(server.fsafe, "read_confined", flaky)
        report = reindex(wiki)
        assert report["skipped"] == [{"path": "ops/backup.md", "reason": "unreadable"}]
        assert "backups run nightly" not in wiki_docs(rag) and len(wiki_ids(rag)) == 4

    def test_skips_are_listed_up_to_a_limit(self, rag, wiki):
        outside = rag.tmp / "outside.md"
        outside.write_text("# O\n\n## O\noutside\n")
        for n in range(30):
            (wiki / f"link{n:02}.md").symlink_to(outside)
        report = reindex(wiki)
        assert report["files"]["skipped"] == 30 and len(report["skipped"]) == 20


class TestBudgetAndResume:
    def test_repeating_a_cut_short_run_converges_on_the_one_shot_state(self, rag, wiki):
        for n in range(4):
            (wiki / f"extra{n}.md").write_text(f"# E{n}\n\n## S\nextra page number {n}\n")
        reindex(wiki, "oneshot")
        want = wiki_ids(rag, "oneshot")
        seen = []
        for _ in range(20):
            report = reindex(wiki, "stepped", budget_seconds=0)
            seen.append(report["remaining"])
            if not report["partial"]:
                break
        assert seen == [6, 5, 4, 3, 2, 1, 0][:len(seen)] and seen[-1] == 0
        assert wiki_ids(rag, "stepped") == want
        assert reindex(wiki, "stepped")["files"]["indexed"] == 0

    def test_every_call_makes_progress_even_with_no_budget(self, rag, wiki):
        report = reindex(wiki, budget_seconds=0)
        assert report["files"]["indexed"] == 1 and report["partial"] and report["remaining"] == 2

    def test_full_mode_ignores_the_budget(self, rag, wiki):
        report = reindex(wiki, mode="full", budget_seconds=0)
        assert report["files"]["indexed"] == 3 and not report["partial"]


class TestFailures:
    def test_an_embedder_failing_on_page_two_keeps_page_one_and_old_data(self, rag, wiki):
        reindex(wiki)
        old_ids = wiki_ids(rag)
        (wiki / "index.md").write_text("# Home\n\n## Overview\nthe first page changed\n")
        (wiki / "ops" / "deploy.md").write_text("# D\n\n## S\nPOISON the second page changed\n")
        rag.embed.fail_when = "POISON"
        with pytest.raises(RuntimeError, match="embedding failed on 'ops/deploy.md'.*1 page"):
            reindex(wiki)
        docs = wiki_docs(rag)
        assert "the first page changed" in docs and "overview page explains" not in docs
        assert "blue green deploy" in docs and "POISON" not in docs  # old nodes intact
        assert len(wiki_ids(rag)) == len(old_ids)
        rag.embed.fail_when = ""
        report = reindex(wiki)
        assert report["files"]["indexed"] == 1 and "POISON" in wiki_docs(rag)

    def test_a_failure_before_any_write_leaves_nothing_behind(self, rag, wiki):
        rag.embed.fail_when = "overview"
        with pytest.raises(RuntimeError, match="embedding failed"):
            reindex(wiki)
        assert wiki_ids(rag) == set()

    def test_a_failed_run_does_not_leave_the_project_busy(self, rag, wiki):
        rag.embed.fail_when = "overview"
        with pytest.raises(RuntimeError):
            reindex(wiki)
        rag.embed.fail_when = ""
        assert "busy" not in reindex(wiki)


class TestFullMode:
    def test_it_re_embeds_everything_and_removes_what_this_run_did_not_produce(self, rag, wiki):
        reindex(wiki)
        col = rag.client.get_collection("rag_acme.llm-wiki__wiki")
        root_id = col.get(include=["metadatas"])["metadatas"][0]["root_id"]
        leftover = {"source": "reindex_wiki", "root_id": root_id, "layer": "wiki", "status": "active"}
        col.upsert(ids=["stale-chunk", "ghost-page"], embeddings=[[0.1] * 64] * 2,
                   documents=["stale chunk of a live page", "chunk of a vanished page"],
                   metadatas=[{**leftover, "path": "index.md", "chunk_index": 9},
                              {**leftover, "path": "ghost.md", "chunk_index": 0}])
        embedded(rag)
        report = reindex(wiki, mode="full")
        assert len(embedded(rag)) == 6  # every chunk again
        assert report["files"]["indexed"] == 3 and report["files"]["removed"] == 1
        assert report["nodes"]["deleted"] == 2
        assert "stale-chunk" not in wiki_ids(rag) and "ghost-page" not in wiki_ids(rag)
        assert len(wiki_ids(rag)) == 6

    def test_nodes_of_other_origin_survive_either_mode(self, rag, wiki):
        reindex(wiki)
        col = rag.client.get_collection("rag_acme.llm-wiki__wiki")
        col.upsert(ids=["foreign-1", "foreign-2", "foreign-3"], embeddings=[[0.1] * 64] * 3,
                   documents=["ingest_wiki page", "other folder page", "no provenance"],
                   metadatas=[{"source": "ingest_wiki", "path": "x.md", "layer": "wiki", "root_id": "aa"},
                              {"source": "reindex_wiki", "path": "y.md", "layer": "wiki", "root_id": "bb"},
                              {"path": "z.md", "layer": "wiki"}])
        for mode in ("incremental", "full"):
            (wiki / "ops" / "backup.md").write_text(f"# B\n\n## N\nbackup text for {mode}\n")
            reindex(wiki, mode=mode, prune=True)
            assert {"foreign-1", "foreign-2", "foreign-3"} <= wiki_ids(rag)

    def test_it_coexists_with_ingest_wiki_in_the_same_project(self, rag, wiki):
        other = rag.tmp / "legacy-pages"
        other.mkdir()
        (other / "old.md").write_text("# Old\n\n## Page\nan older wiki page\n")
        server.ingest_wiki(str(other), "llm-wiki")
        old_ids = wiki_ids(rag)
        reindex(wiki)
        assert old_ids < wiki_ids(rag) and len(wiki_ids(rag)) == len(old_ids) + 6
        (wiki / "index.md").unlink()
        reindex(wiki)
        assert old_ids < wiki_ids(rag)
        assert "an older wiki page" in recall("older wiki page")

    def test_two_folders_in_one_project_do_not_touch_each_other(self, rag, wiki):
        second = rag.tmp / "second-wiki"
        second.mkdir()
        (second / "s.md").write_text("# S\n\n## S\nsecond folder page\n")
        reindex(wiki)
        reindex(second)
        (wiki / "index.md").unlink()
        reindex(wiki)
        assert "second folder page" in wiki_docs(rag)
        for page in list(second.glob("*.md")):
            page.unlink()
        with pytest.raises(ValueError, match="refusing"):  # its own guard, not wiki's content
            reindex(second)
        assert len(wiki_ids(rag)) == 6  # 2 pages left in the first folder + 1 in the second


class TestDryRun:
    def test_it_writes_nothing_and_reports_the_plan(self, rag, wiki):
        before = snapshot(rag.client)
        plan = reindex(wiki, dry_run=True)
        assert plan["dry_run"] and plan["files"]["indexed"] == 3 and plan["nodes"]["added"] == 6
        assert not rag.client.list_collections() and embedded(rag) == []
        assert snapshot(rag.client) == before

    def test_the_plan_matches_what_a_real_run_then_does(self, rag, wiki):
        reindex(wiki)
        (wiki / "ops" / "deploy.md").write_text("# D\n\n## S\nsomething new\n")
        (wiki / "ops" / "backup.md").unlink()
        before = snapshot(rag.client)
        plan = reindex(wiki, dry_run=True)
        assert snapshot(rag.client) == before
        real = reindex(wiki)
        assert (plan["files"], plan["nodes"]) == (real["files"], real["nodes"])


class TestLockAndEventLoop:
    def test_a_second_run_for_the_same_project_is_told_it_is_busy(self, rag, wiki):
        gate = threading.Event()
        rag.embed.gate = gate
        results = {}
        worker = threading.Thread(target=lambda: results.update(first=reindex(wiki)))
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while not server._reindex_busy and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server._reindex_busy
            assert reindex(wiki) == {"busy": True, "project": "llm-wiki"}
            # a different project is not blocked (a plan needs no embedder, so no gate)
            assert "busy" not in reindex(wiki, project="another", dry_run=True)
        finally:
            gate.set()
            worker.join(30)
        assert results["first"]["files"]["indexed"] == 3
        assert reindex(wiki)["files"]["unchanged"] == 3  # the lock is released

    def test_the_tool_is_asynchronous_and_leaves_the_loop_free(self, rag, wiki):
        assert inspect.iscoroutinefunction(server.reindex_wiki)
        gate = threading.Event()
        rag.embed.gate = gate

        async def scenario():
            task = asyncio.create_task(server.reindex_wiki(str(wiki), "llm-wiki"))
            for _ in range(200):  # the loop keeps turning while the worker is held
                await asyncio.sleep(0.01)
                if server._reindex_busy:
                    break
            assert server._reindex_busy and not task.done()
            ticks = 0
            for _ in range(5):
                await asyncio.sleep(0.01)
                ticks += 1
            gate.set()
            return ticks, json.loads(await task)

        ticks, report = asyncio.run(scenario())
        assert ticks == 5 and report["files"]["indexed"] == 3


class TestWhoMayCall:
    @pytest.fixture
    def tokens(self, rag, make_tokens_dir, wiki, monkeypatch):
        monkeypatch.setenv("HERMES_RAG_TRANSPORT", "streamable-http")
        write_token_file(make_tokens_dir, "acme", {
            "tok-admin": {"ingest": True},
            "tok-agent": {},
            "tok-reader": {"write": False},
        }, folders=[wiki])
        write_token_file(make_tokens_dir, "globex", {"tok-globex": {"ingest": True}},
                         folders=[wiki])
        return rag

    def test_without_the_ingest_right_it_is_refused(self, tokens, wiki):
        with as_token("tok-agent"), pytest.raises(PermissionError, match="may not run ingest"):
            server.reindex_wiki_blocking(str(wiki), "p")
        assert not tokens.client.list_collections()

    def test_a_read_only_token_is_refused_first(self, tokens, wiki):
        with as_token("tok-reader"), pytest.raises(PermissionError, match="read-only"):
            server.reindex_wiki_blocking(str(wiki), "p")
        assert not tokens.client.list_collections()

    def test_a_folder_outside_the_namespace_list_is_refused_and_nothing_is_deleted(
            self, tokens, wiki):
        with as_token("tok-admin"):
            server.reindex_wiki_blocking(str(wiki), "p")
        ids = wiki_ids(tokens, "p")
        outside = tokens.tmp / "outside"
        outside.mkdir()
        (outside / "o.md").write_text("# O\n\n## O\noutside page\n")
        with as_token("tok-admin"):
            for path in (outside, wiki / ".." / "outside"):
                with pytest.raises(PermissionError, match="outside the folders allowed"):
                    server.reindex_wiki_blocking(str(path), "p", mode="full", allow_empty=True)
        assert wiki_ids(tokens, "p") == ids

    def test_namespaces_do_not_see_each_other(self, tokens, wiki):
        with as_token("tok-admin"):
            server.reindex_wiki_blocking(str(wiki), "p")
        with as_token("tok-globex"):
            server.reindex_wiki_blocking(str(wiki), "p")
            (wiki / "ops" / "backup.md").unlink()
            server.reindex_wiki_blocking(str(wiki), "p")
        assert len(wiki_ids(tokens, "p", "acme")) == 6
        assert len(wiki_ids(tokens, "p", "globex")) == 4
        with as_token("tok-admin"):
            assert "backups run nightly" in server.recall("nightly backups", project="p", layer="wiki")

    def test_over_mcp_the_tool_is_listed_and_works(self, tokens, wiki, http_client):
        write_json(tokens.tokens_json, {digest("tok-plain"): {"namespace": "initech"}})
        err, out = call(http_client, "tok-admin", "reindex_wiki", wiki_dir=str(wiki), project="p")
        assert not err, out
        assert json.loads(out)["files"]["indexed"] == 3
        err, out = call(http_client, "tok-agent", "reindex_wiki", wiki_dir=str(wiki), project="p")
        assert err and "may not run ingest" in out
        err, out = call(http_client, "tok-admin", "reindex_wiki", wiki_dir=str(wiki), project="p",
                        mode="sideways")
        assert err and "mode must be" in out
