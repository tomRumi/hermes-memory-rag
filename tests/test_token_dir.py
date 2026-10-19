"""Provisioning tokens through a directory of per-namespace files (tokens.d).

The contract these tests hold to:

- a digest listed in `<namespace>.json` (format v1) in the token directory is valid
  over HTTP; deleting the file revokes it, even inside an open session; rewriting it
  by the documented temp-file-and-rename is noticed; temporary files, dotfiles and
  other suffixes are ignored;
- the directory and every file must belong to the server user and be closed to
  others, and files may not be links; one bad file costs only itself;
- the namespace must equal the file name and not be reserved: `legacy` always, and
  `vps-host` (or whatever HERMES_RAG_RESERVED_NAMESPACES says) can only come from the
  operator's tokens file;
- a digest claimed under two namespaces is refused everywhere; on the same namespace
  the tokens file wins; a broken tokens file refuses everything;
- rights are checked as in the tokens file: ingest and verify need write, strict
  types, unknown keys refused; ingest is granted only with folders listed;
- an install with only the directory (even empty) starts; one with nothing does not;
- GET /healthz needs no token and says nothing about the store.
"""

from __future__ import annotations

import json
import logging
import os

import pytest

from hermes_memory_rag import server
from support import (as_token, call, call_in_session, digest, open_session, rpc, write_json,
                     write_token_file)

AGENT, ADMIN, OTHER = "tok-agent", "tok-admin", "tok-other"


@pytest.fixture
def d(rag, make_tokens_dir, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "streamable-http")
    return make_tokens_dir


def tokens_now():
    return server._verifier._current()


def valid(http, token):
    """The server accepts the token (a request with no session is then a 400, not a 401)."""
    return rpc(http, token, "tools/list").status_code != 401


class TestFilesAreHonoured:
    def test_a_token_from_the_directory_works(self, d, http_client):
        write_token_file(d, "acme", {AGENT: {}})
        assert tokens_now()[digest(AGENT)] == server.Caller("acme", False)
        assert valid(http_client, AGENT)
        assert not valid(http_client, "something-else")

    def test_rights_come_from_the_file(self, d):
        write_token_file(d, "acme", {
            AGENT: {"learn_status": "unverified"},
            ADMIN: {"ingest": True, "verify": True}}, folders=["/srv/wiki"])
        assert tokens_now()[digest(AGENT)] == server.Caller("acme", False, True, False, "unverified")
        assert tokens_now()[digest(ADMIN)] == server.Caller("acme", True, True, True)

    def test_deleting_the_file_revokes_inside_an_open_session(self, d, http_client):
        path = write_token_file(d, "acme", {AGENT: {}})
        session = open_session(http_client, AGENT)
        assert call_in_session(http_client, AGENT, session, "stats", project="p")[0] is False
        path.unlink()
        assert rpc(http_client, AGENT, "tools/call", {"name": "stats", "arguments": {"project": "p"}},
                   session=session, id_=3).status_code == 401
        assert not valid(http_client, AGENT)

    def test_a_new_file_is_noticed_without_a_restart(self, d, http_client):
        assert not valid(http_client, AGENT)
        write_token_file(d, "acme", {AGENT: {}})
        assert valid(http_client, AGENT)

    def test_an_atomic_rewrite_is_noticed_even_with_the_same_size(self, d):
        write_token_file(d, "acme", {AGENT: {"write": True}})
        assert tokens_now()[digest(AGENT)].write is True
        before = os.stat(d / "acme.json")
        write_token_file(d, "acme", {AGENT: {"write": False}})  # one character longer or equal
        os.utime(d / "acme.json", ns=(before.st_atime_ns, before.st_mtime_ns))
        assert tokens_now()[digest(AGENT)].write is False

    def test_rights_can_change_inside_an_open_session(self, d, http_client):
        write_token_file(d, "acme", {AGENT: {}})
        session = open_session(http_client, AGENT)
        assert call_in_session(http_client, AGENT, session, "learn", text="one", project="p")[0] is False
        write_token_file(d, "acme", {AGENT: {"write": False}})
        err, out = call_in_session(http_client, AGENT, session, "learn", text="two", project="p")
        assert err and "read-only" in out

    def test_other_names_in_the_directory_are_ignored(self, d):
        write_token_file(d, "acme", {AGENT: {}})
        body = json.dumps({"version": 1, "namespace": "globex",
                           "tokens": {digest(OTHER): {}}})
        for name in (".globex.json", ".globex.json.tmp", "globex.json.tmp", "globex.txt",
                     "globex.json~", "globex"):
            (d / name).write_text(body)
            (d / name).chmod(0o600)
        (d / "sub.json").mkdir()
        assert set(tokens_now()) == {digest(AGENT)}

    def test_the_directory_can_be_moved_by_env(self, rag, monkeypatch):
        elsewhere = rag.tmp / "plugin-tokens"
        elsewhere.mkdir(mode=0o700)
        monkeypatch.setenv("HERMES_RAG_TOKENS_DIR", str(elsewhere))
        write_token_file(elsewhere, "acme", {AGENT: {}})
        assert digest(AGENT) in tokens_now()

    def test_a_missing_directory_means_no_extra_tokens(self, rag):
        write_json(rag.tokens_json, {digest(OTHER): {"namespace": "globex"}})
        assert set(tokens_now()) == {digest(OTHER)}


class TestFilesAreJudgedStrictly:
    def test_a_file_open_to_others_is_ignored(self, d):
        write_token_file(d, "acme", {AGENT: {}})
        write_token_file(d, "globex", {OTHER: {}})
        (d / "globex.json").chmod(0o640)
        assert set(tokens_now()) == {digest(AGENT)}
        (d / "globex.json").chmod(0o600)
        assert set(tokens_now()) == {digest(AGENT), digest(OTHER)}

    def test_a_symlinked_file_is_ignored(self, rag, d):
        real = write_token_file(rag.cfg, "globex", {OTHER: {}}, atomic=False)
        (d / "globex.json").symlink_to(real)
        assert tokens_now() == {}

    @pytest.mark.parametrize("mode", [0o750, 0o755, 0o770])
    def test_a_directory_open_to_others_is_ignored(self, d, mode):
        write_token_file(d, "acme", {AGENT: {}})
        d.chmod(mode)
        try:
            assert tokens_now() == {}
        finally:
            d.chmod(0o700)
        assert digest(AGENT) in tokens_now()

    def test_a_symlinked_directory_is_ignored(self, rag, monkeypatch):
        real = rag.tmp / "real-tokens"
        real.mkdir(mode=0o700)
        write_token_file(real, "acme", {AGENT: {}})
        rag.tokens_dir.symlink_to(real)
        assert tokens_now() == {}

    def test_one_bad_file_costs_only_itself(self, d, caplog):
        write_token_file(d, "acme", {AGENT: {}})
        (d / "globex.json").write_text("{not json")
        (d / "globex.json").chmod(0o600)
        write_token_file(d, "initech", {OTHER: {}})
        with caplog.at_level(logging.ERROR, logger="hermes_memory_rag"):
            assert set(tokens_now()) == {digest(AGENT), digest(OTHER)}
        assert "ignoring that file" in caplog.text

    def test_the_namespace_must_equal_the_file_name(self, d):
        write_json(d / "acme.json", {"version": 1, "namespace": "globex",
                                     "tokens": {digest(AGENT): {}}})
        assert tokens_now() == {}

    @pytest.mark.parametrize("name", ["Bad NS", "a.b", "-x", "x" * 41])
    def test_the_namespace_must_be_valid(self, d, name):
        write_json(d / f"{name}.json", {"version": 1, "namespace": name,
                                        "tokens": {digest(AGENT): {}}})
        assert tokens_now() == {}

    @pytest.mark.parametrize("body", [
        {"version": 2, "namespace": "acme", "tokens": {}},
        {"version": True, "namespace": "acme", "tokens": {}},
        {"namespace": "acme", "tokens": {}},
        {"version": 1, "namespace": "acme", "tokens": []},
        {"version": 1, "namespace": "acme", "tokens": {}, "surprise": 1},
        {"version": 1, "namespace": "acme", "tokens": {"short": {}}},
        {"version": 1, "namespace": "acme", "tokens": {"A" * 64: {}}},
        {"version": 1, "namespace": "acme", "tokens": {"0" * 64: "no"}},
        {"version": 1, "namespace": "acme", "tokens": {"0" * 64: {"ingset": True}}},
        {"version": 1, "namespace": "acme", "tokens": {"0" * 64: {"ingest": "true"}}},
        {"version": 1, "namespace": "acme", "tokens": {"0" * 64: {"verify": 1}}},
        {"version": 1, "namespace": "acme", "tokens": {"0" * 64: {"write": "false"}}},
        {"version": 1, "namespace": "acme", "tokens": {"0" * 64: {"learn_status": "active"}}},
        {"version": 1, "namespace": "acme", "tokens": {}, "allowed_folders": "/x"},
        {"version": 1, "namespace": "acme", "tokens": {}, "allowed_folders": ["rel"]},
        {"version": 1, "namespace": "acme", "tokens": {}, "allowed_folders": [3]},
    ])
    def test_a_malformed_file_vouches_for_nobody(self, d, body):
        write_json(d / "acme.json", body)
        assert tokens_now() == {}

    def test_caps_on_files_and_tokens(self, d, monkeypatch, caplog):
        monkeypatch.setattr(server, "MAX_TOKEN_FILES", 2)
        monkeypatch.setattr(server, "MAX_TOKENS_PER_FILE", 2)
        write_token_file(d, "aa", {"t1": {}, "t2": {}})
        write_token_file(d, "bb", {"t3": {}, "t4": {}, "t5": {}})  # too many tokens
        write_token_file(d, "cc", {"t6": {}})                       # beyond the file cap
        with caplog.at_level(logging.ERROR, logger="hermes_memory_rag"):
            assert set(tokens_now()) == {digest("t1"), digest("t2")}
        assert "only the first 2 are read" in caplog.text


class TestReservedNamespaces:
    @pytest.mark.parametrize("name", ["vps-host", "legacy"])
    def test_reserved_names_cannot_be_minted_from_the_directory(self, d, http_client, name):
        write_token_file(d, name, {AGENT: {}})
        assert tokens_now() == {}
        assert not valid(http_client, AGENT)

    def test_the_operators_file_still_vouches_for_them(self, d, rag):
        write_json(rag.tokens_json, {digest(ADMIN): {"namespace": "vps-host", "ingest": True}})
        write_token_file(d, "vps-host", {AGENT: {}})
        assert set(tokens_now()) == {digest(ADMIN)}

    def test_the_reserved_list_is_configurable_but_legacy_is_always_reserved(self, d, monkeypatch):
        monkeypatch.setenv("HERMES_RAG_RESERVED_NAMESPACES", "factory-host")
        write_token_file(d, "vps-host", {AGENT: {}})
        write_token_file(d, "factory-host", {ADMIN: {}})
        write_token_file(d, "legacy", {OTHER: {}})
        assert set(tokens_now()) == {digest(AGENT)}
        monkeypatch.setenv("HERMES_RAG_RESERVED_NAMESPACES", "")
        assert set(tokens_now()) == {digest(AGENT), digest(ADMIN)}


class TestDuplicates:
    def test_a_digest_in_two_files_is_refused_everywhere(self, d, caplog):
        write_token_file(d, "acme", {AGENT: {}, ADMIN: {}})
        write_token_file(d, "globex", {AGENT: {}})
        with caplog.at_level(logging.ERROR, logger="hermes_memory_rag"):
            assert set(tokens_now()) == {digest(ADMIN)}
        assert "several namespaces" in caplog.text

    def test_a_digest_in_the_tokens_file_and_another_namespace_is_refused_everywhere(self, d, rag):
        write_json(rag.tokens_json, {digest(AGENT): {"namespace": "globex"},
                                     digest(OTHER): {"namespace": "globex"}})
        write_token_file(d, "acme", {AGENT: {}})
        assert set(tokens_now()) == {digest(OTHER)}

    def test_on_the_same_namespace_the_tokens_file_wins(self, d, rag):
        write_json(rag.tokens_json, {digest(AGENT): {"namespace": "acme", "write": False}})
        write_token_file(d, "acme", {AGENT: {"write": True, "verify": True}})
        assert tokens_now()[digest(AGENT)] == server.Caller("acme", False, False)

    def test_a_broken_tokens_file_refuses_the_directory_too(self, d, rag):
        write_token_file(d, "acme", {AGENT: {}})
        write_json(rag.tokens_json, {"nonsense": 1})
        assert tokens_now() == {}
        write_json(rag.tokens_json, {})
        assert digest(AGENT) in tokens_now()


class TestRights:
    @pytest.mark.parametrize("right", ["ingest", "verify"])
    def test_a_right_without_write_refuses_that_token_only(self, d, caplog, right):
        write_token_file(d, "acme", {AGENT: {}, ADMIN: {right: True, "write": False}},
                         folders=["/srv/wiki"])
        with caplog.at_level(logging.ERROR, logger="hermes_memory_rag"):
            assert set(tokens_now()) == {digest(AGENT)}
        assert f"{right} without write" in caplog.text

    def test_ingest_is_not_granted_without_folders(self, d, caplog):
        write_token_file(d, "acme", {ADMIN: {"ingest": True, "verify": True}})
        with caplog.at_level(logging.ERROR, logger="hermes_memory_rag"):
            assert tokens_now()[digest(ADMIN)] == server.Caller("acme", False, True, True)
        assert "no allowed_folders" in caplog.text
        write_token_file(d, "acme", {ADMIN: {"ingest": True}}, folders=[])
        assert tokens_now()[digest(ADMIN)].ingest is False

    def test_an_agent_token_cannot_index_or_verify(self, d, rag):
        write_token_file(d, "acme", {AGENT: {"learn_status": "unverified"}}, folders=[rag.tmp])
        with as_token(AGENT):
            with pytest.raises(PermissionError, match="may not run ingest"):
                server.ingest_wiki(str(rag.tmp), "p")
            with pytest.raises(PermissionError, match="may not run ingest"):
                server.reindex_wiki_blocking(str(rag.tmp), "p")
            server.learn("an agent claim", project="p", status="active")
            with pytest.raises(PermissionError, match="may not run verify"):
                server.verify("abcd", project="p")
        col = rag.client.get_collection("rag_acme.p__memory")
        assert col.get(include=["metadatas"])["metadatas"][0]["status"] == "unverified"

    def test_an_admin_token_indexes_inside_its_folders_and_promotes(self, d, rag):
        wiki = rag.tmp / "wiki"
        wiki.mkdir()
        (wiki / "a.md").write_text("# A\n\n## Setup\nhow it is set up\n")
        write_token_file(d, "acme", {ADMIN: {"ingest": True, "verify": True},
                                     AGENT: {"learn_status": "unverified"}}, folders=[wiki])
        with as_token(ADMIN):
            assert "nodes" in server.ingest_wiki(str(wiki), "p")
            with pytest.raises(PermissionError, match="outside the folders"):
                server.ingest_wiki(str(rag.tmp), "p")
        with as_token(AGENT):
            node = server.learn("agents write this", project="p")
        with as_token(ADMIN):
            assert server.verify(node[:8], project="p").startswith("verified")


class TestStartup:
    @pytest.fixture
    def serving(self, rag, monkeypatch):
        monkeypatch.setenv("HERMES_RAG_TRANSPORT", "streamable-http")
        ran = []
        monkeypatch.setattr(server.mcp, "run", lambda transport: ran.append(transport))
        return ran

    def test_an_empty_directory_alone_is_enough(self, rag, make_tokens_dir, serving):
        server.main()
        assert serving == ["streamable-http"]

    def test_a_tokens_file_alone_is_enough(self, rag, serving):
        write_json(rag.tokens_json, {digest(AGENT): {"namespace": "acme"}})
        server.main()
        assert serving == ["streamable-http"]

    def test_nothing_at_all_refuses(self, rag, serving):
        with pytest.raises(SystemExit, match="refusing to serve"):
            server.main()
        assert serving == []

    def test_an_empty_tokens_file_without_a_directory_refuses(self, rag, serving):
        write_json(rag.tokens_json, {})
        with pytest.raises(SystemExit, match="no tokens"):
            server.main()

    def test_an_unfit_tokens_file_refuses_even_with_a_directory(self, rag, make_tokens_dir, serving):
        write_json(rag.tokens_json, {digest(AGENT): {"namespace": "acme"}}, mode=0o644)
        with pytest.raises(SystemExit, match="mode 600"):
            server.main()

    def test_an_unfit_directory_refuses(self, rag, make_tokens_dir, serving):
        write_json(rag.tokens_json, {digest(AGENT): {"namespace": "acme"}})
        make_tokens_dir.chmod(0o755)
        with pytest.raises(SystemExit, match="mode 700"):
            server.main()

    def test_the_provisioned_namespace_starts_serving_the_moment_a_file_appears(
            self, rag, make_tokens_dir, http_client, monkeypatch):
        monkeypatch.setenv("HERMES_RAG_TRANSPORT", "streamable-http")
        assert not valid(http_client, AGENT)
        write_token_file(make_tokens_dir, "acme", {AGENT: {}})
        assert valid(http_client, AGENT)


class TestHealth:
    def test_healthz_needs_no_token_and_says_nothing_else(self, rag, http_client):
        response = http_client.get("/healthz")
        assert response.status_code == 200 and response.json() == {"ok": True}

    def test_the_mcp_endpoint_still_needs_a_token(self, rag, http_client):
        assert rpc(http_client, None, "tools/list").status_code == 401

    def test_healthz_is_get_only(self, rag, http_client):
        assert http_client.post("/healthz").status_code == 405
