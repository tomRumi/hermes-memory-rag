"""Node status: replaced and withdrawn facts must stop being handed to the model,
and nothing may be deleted to make room.

The contract these tests hold to:

- every write stamps status/source/created;
- recall returns only live nodes;
- learn(supersedes=...) marks the old note replaced and keeps it;
- retire(...) withdraws a note and keeps it;
- an unknown or ambiguous reference changes nothing;
- past the staging cap the oldest notes are moved aside, not deleted;
- mirrored memory-file notes do not count toward the merge threshold;
- the cross-project layer is reachable, is labelled, and never outranks the
  project's own memory.
"""

from __future__ import annotations

import pytest

from hermes_memory_rag import server


def _mock_embed():
    """A real embedding object that needs no Ollama.

    It has to be a genuine llama-index embedding (the search path installs it into
    a global Settings slot, which is type-checked), and it has to return a fixed
    width, since a collection's width cannot change once written.
    """
    from llama_index.core.embeddings import MockEmbedding
    return MockEmbedding(embed_dim=16)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A store in a temp dir, with the embedder mocked and the ledger contained."""
    import chromadb
    from llama_index.core import Settings

    path = tmp_path / "chroma"
    path.mkdir()
    client = chromadb.PersistentClient(path=str(path))

    monkeypatch.setattr(server, "STORE_DIR", path)
    monkeypatch.setattr(server, "_client", lambda: client)
    monkeypatch.setattr(server, "_embed_model", _mock_embed)
    Settings.embed_model = _mock_embed()

    return client


def _memory_collection(client, project):
    return client.get_or_create_collection(server._collection_names(project)["memory"])


def col_get(client, project, node_id):
    """The metadata currently stored for one node."""
    col = _memory_collection(client, project)
    return (col.get(ids=[node_id], include=["metadatas"])["metadatas"] or [{}])[0]


def _status_of(client, project, text):
    col = _memory_collection(client, project)
    node_id = server._det_id("mem", text)
    return (col.get(ids=[node_id], include=["metadatas"])["metadatas"] or [{}])[0]


class TestStampOnWrite:
    def test_learn_stamps_status_source_and_created(self, store):
        text = "the store must stay on a local disk"
        server.learn(text, kind="rag:gotcha", project="stamptest")

        meta = _status_of(store, "stamptest", text)
        assert meta["status"] == server.STATUS_ACTIVE
        assert meta["source"] == "learn"
        assert meta["created"]


class TestSupersedeKeepsHistory:
    def test_replaced_note_stops_being_returned_but_is_kept(self, store):
        old = "the store lives at ~/old-store-location"
        new = "the store lives at ~/hermes-rag"
        old_id = server.learn(old, kind="rag:gotcha", project="suptest")

        result = server.learn(new, kind="rag:gotcha", project="suptest",
                              supersedes=old_id[:8])
        assert "replaced" in result

        old_meta = _status_of(store, "suptest", old)
        new_meta = _status_of(store, "suptest", new)
        assert old_meta["status"] == server.STATUS_SUPERSEDED
        assert old_meta["superseded_by"] == server._det_id("mem", new)
        # the old text is still there — nothing was deleted
        col = _memory_collection(store, "suptest")
        assert col.get(ids=[server._det_id("mem", old)])["documents"] == [old]
        assert new_meta["status"] == server.STATUS_ACTIVE

    def test_unknown_reference_changes_nothing_and_says_so(self, store):
        text = "a note with no successor"
        server.learn(text, kind="generic", project="suptest2")

        result = server.learn("another note", kind="generic", project="suptest2",
                              supersedes="deadbeef")
        assert "matched no single note" in result
        assert _status_of(store, "suptest2", text)["status"] == server.STATUS_ACTIVE


class TestRetire:
    def test_retired_note_is_withdrawn_not_deleted(self, store):
        text = "an assumption that turned out to be wrong"
        node_id = server.learn(text, kind="generic", project="rettest")

        assert "retired" in server.retire(node_id[:8], reason="not true", project="rettest")
        meta = _status_of(store, "rettest", text)
        assert meta["status"] == server.STATUS_RETIRED
        assert meta["retired_reason"] == "not true"
        col = _memory_collection(store, "rettest")
        assert col.get(ids=[node_id])["documents"] == [text]

    def test_unknown_reference_is_refused(self, store):
        server.learn("a real note", kind="generic", project="rettest2")
        result = server.retire("nope", project="rettest2")
        assert "nothing changed" in result

    def test_second_retire_is_a_no_op(self, store):
        node_id = server.learn("retire me twice", kind="generic", project="rettest3")
        server.retire(node_id[:8], project="rettest3")
        assert "already" in server.retire(node_id[:8], project="rettest3")


class TestRecallHidesNonLive:
    def test_replaced_fact_never_comes_back(self, store):
        # Deliberately not substrings of one another: a test that checks for
        # absence must not be satisfied-or-failed by string overlap.
        old = "the rsync deploy used the legacy port twenty-two"
        new = "the rsync deploy now uses port 2222 exclusively"
        old_id = server.learn(old, kind="config:fact", project="recalltest")
        server.learn(new, kind="config:fact", project="recalltest", supersedes=old_id[:8])

        out = server.recall("how does deploy use rsync", project="recalltest",
                            layer="memory", top_memory=5)
        assert new in out
        assert old not in out

    def test_retired_fact_never_comes_back(self, store):
        text = "the old proxy port was 4000"
        node_id = server.learn(text, kind="config:fact", project="recalltest2")
        server.retire(node_id[:8], project="recalltest2")

        out = server.recall("what port does the proxy use", project="recalltest2",
                            layer="memory", top_memory=5)
        assert text not in out


class TestOverflowArchivesInsteadOfDeleting:
    def test_oldest_notes_are_moved_aside_and_kept(self, store, monkeypatch):
        monkeypatch.setattr(server, "MEMORY_CAP", 3)
        monkeypatch.setattr(server, "CONSOLIDATE_THRESHOLD", 99)  # keep the noise out

        texts = [f"note number {i}" for i in range(5)]
        for text in texts:
            server.learn(text, kind="generic", project="captest_overflow")

        col = _memory_collection(store, "captest_overflow")
        assert col.count() == 5, "nothing may be deleted to make room"

        statuses = {t: _status_of(store, "captest_overflow", t)["status"] for t in texts}
        archived = [t for t, s in statuses.items() if s == server.STATUS_ARCHIVED]
        assert len(archived) == 2, f"expected the two oldest archived, got {statuses}"
        assert statuses[texts[0]] == server.STATUS_ARCHIVED
        assert statuses[texts[-1]] == server.STATUS_ACTIVE


class TestStagingCount:
    def test_a_learning_kind_starting_with_memory_still_counts(self, store):
        """Regression: mirroring used to be detected from the kind string, so a
        note filed as "memory:decision" was stored, never merged and never
        reported. Detection is a field now, so any kind is safe."""
        server.learn("a decision about memory layout", kind="memory:decision",
                     project="counttest0")
        col = _memory_collection(store, "counttest0")
        assert server._staging_count(col) == 1

    def test_mirrored_memory_notes_do_not_count(self, store):
        for i in range(3):
            server.learn(f"a real learning {i}", kind="rag:gotcha", project="counttest")
        # A mirror writer marks itself through the source field.
        for i in range(4):
            server.learn(f"an entry mirrored from the memory file {i}",
                         kind="generic", project="counttest")
            node_id = server._det_id("mem", f"an entry mirrored from the memory file {i}")
            meta = col_get(store, "counttest", node_id)
            _memory_collection(store, "counttest").update(
                ids=[node_id], metadatas=[{**meta, "source": server.MIRROR_SOURCE}])

        col = _memory_collection(store, "counttest")
        assert col.count() == 7
        assert server._staging_count(col) == 3

    def test_non_live_notes_do_not_count(self, store):
        server.learn("a live note", kind="rag:gotcha", project="counttest2")
        gone = server.learn("a note about to be retired", kind="rag:gotcha", project="counttest2")
        server.retire(gone[:8], project="counttest2")

        col = _memory_collection(store, "counttest2")
        assert server._staging_count(col) == 1
        assert _status_of(store, "counttest2", "a live note")["status"] == server.STATUS_ACTIVE


class TestProjectAttribution:
    """A session's working directory is not necessarily the project being worked
    on, so a file of path prefixes decides; the directory name is the last resort."""

    def test_explicit_name_wins(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "PROJECTS_FILE", tmp_path / "none.yaml")
        assert server._resolve_project("named", cwd="/tmp/whatever") == "named"

    def test_path_prefix_maps_to_project(self, tmp_path, monkeypatch):
        mapping = tmp_path / "projects.yaml"
        mapping.write_text(
            "- path: /code/site-a\n  project: site-a\n"
            "- path: /code/site-b\n  project: site-b\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(server, "PROJECTS_FILE", mapping)
        assert server._resolve_project("", cwd="/code/site-b/plugins") == "site-b"
        assert server._resolve_project("", cwd="/code/site-a/wp-content") == "site-a"

    def test_longest_prefix_wins_for_nested_projects(self, tmp_path, monkeypatch):
        mapping = tmp_path / "projects.yaml"
        mapping.write_text(
            "- path: /code\n  project: parent\n"
            "- path: /code/inner\n  project: inner\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(server, "PROJECTS_FILE", mapping)
        assert server._resolve_project("", cwd="/code/inner/src") == "inner"
        assert server._resolve_project("", cwd="/code/other") == "parent"

    def test_unmapped_directory_falls_back_to_its_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "PROJECTS_FILE", tmp_path / "missing.yaml")
        assert server._resolve_project("", cwd="/somewhere/else/proj") == "proj"

    def test_a_broken_file_is_not_fatal(self, tmp_path, monkeypatch):
        bad = tmp_path / "projects.yaml"
        bad.write_text("this is not: a valid entry\n:::\n", encoding="utf-8")
        monkeypatch.setattr(server, "PROJECTS_FILE", bad)
        assert server._resolve_project("", cwd="/somewhere/proj") == "proj"


class TestGlobalLayer:
    def test_global_fact_reaches_another_project_and_is_labelled(self, store):
        text = "the user wants costs shown in dollars only"
        server.learn(text, kind="memory:fact", project=server.GLOBAL_PROJECT)

        out = server.recall("how should costs be shown", project="someotherproject",
                            layer="memory", top_memory=3)
        assert "[global]" in out
        assert text in out

    def test_project_memory_comes_before_global(self, store):
        server.learn("the project uses port 4000 locally", kind="config:fact",
                     project="ordertest")
        server.learn("the user prefers plain English everywhere", kind="memory:fact",
                     project=server.GLOBAL_PROJECT)

        out = server.recall("what port does the project use and what language",
                            project="ordertest", layer="memory", top_memory=3)
        assert out.index("port 4000") < out.index("[global]")