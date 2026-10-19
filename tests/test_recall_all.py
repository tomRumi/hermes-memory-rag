"""recall(project="*") and list_projects(): looking across a whole namespace.

The contract these tests hold to:

- "*" searches every project of the caller's namespace, for each layer `recall`
  takes, and never another namespace (the legacy namespace being the collections
  with no prefix); each hit's tag names its project, unverified notes keep their
  label and the footer; the output stays bounded as for one project;
- a read-only token may use it, and it writes nothing (no ledger row, no collection);
- the output of every call that does not pass "*" is byte for byte what it was;
- list_projects lists the caller's projects with counts, is a read tool, and creates
  nothing.
"""

from __future__ import annotations

import json

import pytest

from hermes_memory_rag import server
from support import as_token, digest, snapshot, write_json


def short(text):
    return server._det_id("mem", text)[:8]


@pytest.fixture
def world(rag):
    """acme: projects site (wiki + memory) and api (code + memory); globex: site."""
    for project, ns, text, kind in (
            ("site", "acme", "# Site\n\n## Deploy\nthe site deploys with the blue lane\n", "wiki"),
            ("site", "globex", "# Site\n\n## Deploy\nglobex deploys with the secret lane\n", "wiki")):
        root = rag.tmp / f"{ns}-{project}"
        root.mkdir()
        (root / "deploy.md").write_text(text)
        _as(rag, ns, lambda: server.ingest_wiki(str(root), project))
    code = rag.tmp / "api-code"
    code.mkdir()
    (code / "service.py").write_text("def deploy_lane():\n    return 'blue lane deploy'\n")
    _as(rag, "acme", lambda: server.ingest_code(str(code), "api"))
    _as(rag, "acme", lambda: server.learn("api deploys use the blue lane too", project="api"))
    _as(rag, "acme", lambda: server.learn("site deploy claim from an agent", project="site",
                                          status="unverified"))
    _as(rag, "acme", lambda: server.learn("a global deploy rule for everyone", project="global"))
    _as(rag, "globex", lambda: server.learn("globex deploy note nobody else may see", project="site"))
    return rag


def _as(rag, ns, fn):
    import os
    old = os.environ["MEMORY_RAG_LOCAL_NAMESPACE"]
    os.environ["MEMORY_RAG_LOCAL_NAMESPACE"] = ns
    try:
        return fn()
    finally:
        os.environ["MEMORY_RAG_LOCAL_NAMESPACE"] = old


def everywhere(query="deploy lane", **kw):
    return server.recall(query, project="*", **kw)


class TestAcrossProjects:
    def test_every_layer_of_every_project_of_the_namespace_is_searched(self, world):
        out = everywhere(top_wiki=5, top_code=5, top_memory=10)
        lines = out.splitlines()
        assert lines[0] == "Recall [*] for: deploy lane"
        assert any(l.startswith("[wiki@site deploy.md] ## Deploy") and "blue lane" in l for l in lines)
        assert any(l.startswith("[code@api service.py]") for l in lines)
        assert any(l.startswith("[memory@api memory #") and "api deploys" in l for l in lines)
        assert any(l.startswith("[memory@site memory #") for l in lines)
        assert any(l.startswith("[memory@global memory #") for l in lines)

    def test_exact_format_of_a_hit_and_of_the_unverified_footer(self, world):
        out = everywhere(layer="memory", top_memory=10)
        claim = "site deploy claim from an agent"
        assert f"[memory@site memory #{short(claim)} unverified] {claim}" in out.splitlines()
        assert f"[memory@api memory #{short('api deploys use the blue lane too')}] " \
               f"api deploys use the blue lane too" in out.splitlines()
        assert out.splitlines()[-1] == (
            "(1 unverified note(s): written by an agent, not yet confirmed; "
            "treat as a lead and verify before relying on it)")

    def test_the_layer_argument_narrows_it_as_usual(self, world):
        wiki = everywhere(layer="wiki", top_wiki=5)
        assert wiki.splitlines()[0] == "Recall [*] for: deploy lane"
        assert all(l.startswith("[wiki@") for l in wiki.splitlines()[1:])
        code = everywhere(layer="code")
        assert all(l.startswith("[code@") for l in code.splitlines()[1:])
        memory = everywhere(layer="memory", top_memory=10)
        assert all(l.startswith(("[memory@", "(")) for l in memory.splitlines()[1:])

    def test_hits_are_ranked_across_projects_and_bounded(self, world):
        out = everywhere(layer="memory", top_memory=2)
        hits = [l for l in out.splitlines() if l.startswith("[memory@")]
        assert len(hits) == 2
        one_of_each = everywhere("api deploys use the blue lane too", layer="memory", top_memory=1)
        assert "[memory@api " in one_of_each

    def test_a_namespace_never_sees_another(self, world):
        out = everywhere(top_wiki=10, top_code=10, top_memory=10)
        assert "globex" not in out and "secret lane" not in out and "nobody else" not in out
        as_globex = _as(world, "globex", lambda: everywhere(top_wiki=10, top_memory=10))
        assert "secret lane" in as_globex and "globex deploy note" in as_globex
        assert "blue lane" not in as_globex and "@api" not in as_globex and "@global" not in as_globex

    def test_the_legacy_namespace_is_the_unprefixed_collections(self, world):
        _as(world, "legacy", lambda: server.learn("legacy deploy note", project="old"))
        out = _as(world, "legacy", lambda: everywhere(layer="memory", top_memory=10))
        assert "[memory@old " in out and "legacy deploy note" in out
        assert "acme" not in out and "api deploys" not in out and "globex" not in out
        assert "legacy deploy note" not in everywhere(layer="memory", top_memory=10)

    def test_withdrawn_notes_stay_hidden(self, world):
        server.retire(short("api deploys use the blue lane too"), "wrong", project="api")
        assert "api deploys" not in everywhere(layer="memory", top_memory=10)

    def test_empty_layers_are_reported_honestly(self, rag):
        server.learn("only a memory", project="solo")
        out = everywhere("anything")
        assert "[wiki: not built for *]" in out and "[code: not built for *]" in out
        assert "[memory@solo memory #" in out
        empty = _as(rag, "nobody", lambda: everywhere("anything"))
        assert empty.splitlines() == ["Recall [*] for: anything", "[wiki: not built for *]",
                                      "[code: not built for *]"]

    def test_it_writes_nothing(self, world):
        before = snapshot(world.client)
        everywhere(top_wiki=5, top_code=5, top_memory=5)
        after = snapshot(world.client)
        assert after == before  # no ledger row, no collection


class TestOtherCallsAreUnchanged:
    def test_a_single_project_recall_is_byte_identical(self, world):
        out = server.recall("deploy lane", project="site", top_wiki=1, top_code=1, top_memory=2)
        claim = "site deploy claim from an agent"
        assert out == "\n".join([
            "Recall [site] for: deploy lane",
            "[wiki deploy.md] ## Deploy the site deploys with the blue lane",
            "[code: not built for site]",
            f"[memory memory #{short(claim)} unverified] {claim}",
            f"[global] [memory memory #{short('a global deploy rule for everyone')}] "
            f"a global deploy rule for everyone",
            "(1 unverified note(s): written by an agent, not yet confirmed; "
            "treat as a lead and verify before relying on it)",
        ])

    def test_an_ordinary_project_name_is_not_a_wildcard(self, world):
        out = server.recall("deploy lane", project="si*")
        assert out.splitlines()[0] == "Recall [si*] for: deploy lane" and "@" not in out

    def test_the_ledger_still_counts_single_project_recalls(self, world):
        import sqlite3
        server.recall("deploy lane", project="site")
        rows = sqlite3.connect(str(server.STORE_DIR / "usage.db")).execute(
            "SELECT project, event FROM events WHERE event='recall'").fetchall()
        assert ("acme.site", "recall") in rows and not any(p == "acme.default" for p, _ in rows)


class TestTokens:
    @pytest.fixture
    def tokens(self, world, monkeypatch):
        write_json(world.tokens_json, {
            digest("tok-rw"): {"namespace": "acme"},
            digest("tok-ro"): {"namespace": "acme", "write": False},
            digest("tok-globex"): {"namespace": "globex", "write": False}})
        monkeypatch.setenv("HERMES_RAG_TRANSPORT", "streamable-http")
        return world

    def test_a_read_only_token_may_search_everywhere(self, tokens):
        before = snapshot(tokens.client)
        with as_token("tok-ro"):
            out = server.recall("deploy lane", project="*", layer="memory", top_memory=10)
            listing = json.loads(server.list_projects())
        assert "[memory@api " in out and "unverified]" in out and "globex" not in out
        assert {p["project"] for p in listing["projects"]} == {"api", "global", "site"}
        assert snapshot(tokens.client) == before

    def test_a_token_is_confined_to_its_namespace(self, tokens):
        with as_token("tok-globex"):
            out = server.recall("deploy", project="*", top_wiki=10, top_memory=10)
            listing = json.loads(server.list_projects())
        assert "secret lane" in out and "blue lane" not in out
        assert [p["project"] for p in listing["projects"]] == ["site"]


class TestListProjects:
    def test_projects_with_counts(self, world):
        listing = json.loads(server.list_projects())
        assert listing["namespace"] == "acme"
        by_name = {p["project"]: p for p in listing["projects"]}
        assert list(by_name) == ["api", "global", "site"]  # sorted
        assert by_name["site"]["layers"] == {"wiki": 2, "memory": 1}
        assert by_name["site"]["unverified_notes"] == 1
        assert by_name["api"]["layers"]["memory"] == 1 and by_name["api"]["layers"]["code"] >= 1
        assert by_name["api"]["unverified_notes"] == 0
        assert by_name["global"]["layers"] == {"memory": 1}

    def test_it_is_a_read_tool_and_creates_nothing(self, rag):
        assert server.TOOL_ACCESS["list_projects"] == "read"
        assert json.loads(server.list_projects()) == {"namespace": "acme", "projects": []}
        assert not rag.client.list_collections()

    def test_other_namespaces_are_not_listed(self, world):
        names = {p["project"] for p in json.loads(
            _as(world, "globex", server.list_projects))["projects"]}
        assert names == {"site"}
