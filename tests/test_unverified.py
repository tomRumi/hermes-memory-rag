"""Unverified notes: an agent's claim is kept and returned, but labelled, and only a
holder of the verify right can turn it into an established fact.

The contract these tests hold to:

- `learn` without a status is active, exactly as before; `status="unverified"` stores
  an unverified note; anything else is refused;
- recall returns an unverified note inside its bracket after the id, adds one footer
  line saying how many there were, and leaves active notes and wiki hits unchanged;
- a token whose entry says `learn_status: unverified` cannot write anything else;
- `verify` needs the verify right (and write), records who and when, and changes
  nothing for a note that is not unverified;
- learning text that is already an unverified note promotes it only for a caller that
  may verify and says "active" outright; an active note is never demoted;
- retired and superseded unverified notes are hidden like any other;
- past the staging cap the oldest live notes, unverified ones included, are archived;
- stats counts them; consolidation never merges them; the weekly review lists the
  old ones; a store written before statuses existed still answers.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest

from hermes_memory_rag import server
from support import as_token, digest, write_json

AGENT, MAINTAINER, PLAIN, READER, FORCED_VERIFIER = (
    "tok-agent", "tok-maintainer", "tok-plain", "tok-reader", "tok-forced-verifier")
NOTE = "the deploy lane for acme is blue and takes ten minutes"
OTHER = "invoices are generated nightly at three"


@pytest.fixture
def tokens(rag, monkeypatch):
    write_json(rag.tokens_json, {
        digest(AGENT): {"namespace": "acme", "learn_status": "unverified"},
        digest(MAINTAINER): {"namespace": "acme", "verify": True},
        digest(PLAIN): {"namespace": "acme"},
        digest(READER): {"namespace": "acme", "write": False},
        digest(FORCED_VERIFIER): {"namespace": "acme", "verify": True,
                                  "learn_status": "unverified"},
    })
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "streamable-http")
    return rag


def _memory(rag, project="p", ns="acme"):
    return rag.client.get_or_create_collection(server._collection_names(project, ns)["memory"])


def _meta(rag, text, project="p", ns="acme"):
    col = _memory(rag, project, ns)
    return (col.get(ids=[server._det_id("mem", text)], include=["metadatas"])["metadatas"] or [{}])[0]


def _learn(token, text, **kw):
    with as_token(token):
        return server.learn(text, project=kw.pop("project", "p"), **kw)


def _recall(token, query, **kw):
    with as_token(token):
        return server.recall(query, project="p", layer="memory", **kw)


def _short(text):
    return server._det_id("mem", text)[:8]


class TestLearnStatus:
    def test_default_stays_active(self, rag):
        server.learn(NOTE, project="p")
        assert _meta(rag, NOTE)["status"] == server.STATUS_ACTIVE
        out = server.recall("deploy lane", project="p", layer="memory")
        assert NOTE in out and "unverified" not in out

    def test_unverified_is_stored_and_labelled_in_recall(self, rag):
        server.learn(NOTE, project="p", status="unverified", origin="agent:qa")
        meta = _meta(rag, NOTE)
        assert meta["status"] == server.STATUS_UNVERIFIED and meta["origin"] == "agent:qa"
        out = server.recall("deploy lane", project="p", layer="memory")
        assert f"[memory memory #{_short(NOTE)} unverified] {NOTE}" in out
        assert out.splitlines()[-1] == (
            "(1 unverified note(s): written by an agent, not yet confirmed; "
            "treat as a lead and verify before relying on it)")

    def test_active_notes_render_exactly_as_before(self, rag):
        server.learn(NOTE, project="p", status="unverified")
        server.learn(OTHER, project="p", status="active")
        out = server.recall("invoices nightly", project="p", layer="memory", top_memory=5)
        assert f"[memory memory #{_short(OTHER)}] {OTHER}" in out
        assert f"#{_short(NOTE)} unverified]" in out
        assert "(1 unverified note(s)" in out  # only the unverified one is counted

    def test_footer_counts_every_unverified_hit(self, rag):
        server.learn(NOTE, project="p", status="unverified")
        server.learn(OTHER, project="p", status="unverified")
        out = server.recall("deploy invoices", project="p", layer="memory", top_memory=5)
        assert "(2 unverified note(s)" in out

    def test_no_footer_without_unverified_notes(self, rag):
        server.learn(NOTE, project="p")
        assert "unverified" not in server.recall("deploy", project="p")

    def test_the_global_layer_is_labelled_too(self, rag):
        server.learn(NOTE, project="global", status="unverified")
        out = server.recall("deploy lane", project="p")
        assert f"[global] [memory memory #{_short(NOTE)} unverified]" in out
        assert "(1 unverified note(s)" in out

    @pytest.mark.parametrize("bad", ["verified", "UNVERIFIED", "retired", "archived", " "])
    def test_other_statuses_are_refused(self, rag, bad):
        with pytest.raises(ValueError, match="status must be"):
            server.learn(NOTE, project="p", status=bad)
        assert not rag.client.list_collections()

    def test_origin_is_trimmed_and_optional(self, rag):
        server.learn(NOTE, project="p", origin="  agent:" + "x" * 200 + "\n")
        assert len(_meta(rag, NOTE)["origin"]) == 80
        server.learn(OTHER, project="p")
        assert "origin" not in _meta(rag, OTHER)

    def test_unverified_counts_toward_the_staging_count(self, rag):
        server.learn(NOTE, project="p", status="unverified")
        assert server._staging_count(_memory(rag)) == 1


class TestForcedStatus:
    def test_a_forced_token_cannot_write_active_notes(self, tokens):
        _learn(AGENT, NOTE, status="active")
        assert _meta(tokens, NOTE)["status"] == "unverified"
        _learn(AGENT, OTHER)  # and the default is forced too
        assert _meta(tokens, OTHER)["status"] == "unverified"

    def test_a_plain_token_is_unchanged(self, tokens):
        _learn(PLAIN, NOTE)
        assert _meta(tokens, NOTE)["status"] == "active"

    def test_the_entry_is_validated(self, tokens):
        write_json(tokens.tokens_json, {digest(AGENT): {"namespace": "acme",
                                                        "learn_status": "unverifed"}})
        with pytest.raises(ValueError, match="malformed"):
            server._read_tokens(tokens.tokens_json)
        write_json(tokens.tokens_json, {digest(AGENT): {"namespace": "acme",
                                                        "learn_status": "active"}})
        with pytest.raises(ValueError, match="malformed"):
            server._read_tokens(tokens.tokens_json)

    def test_defaults_keep_old_callers_equal(self, tokens):
        assert server._read_tokens(tokens.tokens_json)[digest(PLAIN)] == server.Caller("acme", False)


class TestStatusIsHidden:
    def test_a_retired_unverified_note_is_hidden(self, rag):
        server.learn(NOTE, project="p", status="unverified")
        server.retire(_short(NOTE), "wrong", project="p")
        assert NOTE not in server.recall("deploy lane", project="p", layer="memory")

    def test_a_superseded_unverified_note_is_hidden(self, rag):
        server.learn(NOTE, project="p", status="unverified")
        server.learn(OTHER, project="p", supersedes=_short(NOTE))
        assert _meta(rag, NOTE)["status"] == "superseded"
        out = server.recall("deploy lane", project="p", layer="memory", top_memory=5)
        assert NOTE not in out and OTHER in out


class TestVerify:
    def test_needs_the_flag(self, tokens):
        _learn(AGENT, NOTE)
        for token in (AGENT, PLAIN):
            with as_token(token), pytest.raises(PermissionError, match="may not run verify"):
                server.verify(_short(NOTE), project="p")
        assert _meta(tokens, NOTE)["status"] == "unverified"

    def test_a_read_only_token_is_refused_before_anything_changes(self, tokens):
        _learn(AGENT, NOTE)
        with as_token(READER), pytest.raises(PermissionError, match="read-only"):
            server.verify(_short(NOTE), project="p")
        assert _meta(tokens, NOTE)["status"] == "unverified"

    def test_verify_without_write_is_a_contradiction(self, tokens, caplog):
        write_json(tokens.tokens_json, {digest(READER): {"namespace": "acme", "write": False,
                                                         "verify": True},
                                        digest(PLAIN): {"namespace": "acme"}})
        with caplog.at_level(logging.ERROR, logger="hermes_memory_rag"):
            loaded = server._read_tokens(tokens.tokens_json)
        assert digest(READER) not in loaded and digest(PLAIN) in loaded
        assert "verify without write" in caplog.text

    def test_promotes_and_records_who_and_when(self, tokens):
        _learn(AGENT, NOTE)
        with as_token(MAINTAINER):
            assert server.verify(_short(NOTE), project="p", note=" checked on staging ") == \
                f"verified {_short(NOTE)}"
        meta = _meta(tokens, NOTE)
        assert meta["status"] == "active" and meta["verified_at"]
        assert meta["verified_by"] == f"acme:{digest(MAINTAINER)[:12]}"
        assert meta["verified_note"] == "checked on staging"
        out = _recall(PLAIN, "deploy lane")
        assert f"#{_short(NOTE)}]" in out and "unverified" not in out

    def test_the_local_owner_may_verify(self, rag):
        server.learn(NOTE, project="p", status="unverified")
        assert server.verify(_short(NOTE), project="p") == f"verified {_short(NOTE)}"
        assert _meta(rag, NOTE)["verified_by"] == "acme:local"

    def test_other_states_change_nothing(self, tokens):
        _learn(PLAIN, NOTE)
        with as_token(MAINTAINER):
            assert "is active, not unverified" in server.verify(_short(NOTE), project="p")
            assert "no single memory note" in server.verify("zzzzzzzz", project="p")
        _learn(AGENT, OTHER)
        with as_token(PLAIN):
            server.retire(_short(OTHER), "no", project="p")
        with as_token(MAINTAINER):
            assert "is retired, not unverified" in server.verify(_short(OTHER), project="p")

    def test_an_unknown_layer_creates_nothing(self, tokens):
        before = {c.name for c in tokens.client.list_collections()}
        with as_token(MAINTAINER):
            assert "nothing changed" in server.verify("abcd1234", project="nowhere")
        assert {c.name for c in tokens.client.list_collections()} == before

    def test_another_namespace_cannot_verify(self, tokens):
        _learn(AGENT, NOTE)
        write_json(tokens.tokens_json, {
            digest(AGENT): {"namespace": "acme", "learn_status": "unverified"},
            digest("tok-other"): {"namespace": "globex", "verify": True}})
        with as_token("tok-other"):
            assert "nothing changed" in server.verify(_short(NOTE), project="p")
        assert _meta(tokens, NOTE)["status"] == "unverified"


class TestPromotionByRelearning:
    def test_an_active_request_from_a_verifier_promotes(self, tokens):
        _learn(AGENT, NOTE)
        assert _learn(MAINTAINER, NOTE, status="active").endswith("promoted from unverified")
        assert _meta(tokens, NOTE)["status"] == "active"
        assert _meta(tokens, NOTE)["verified_by"].startswith("acme:")

    @pytest.mark.parametrize("who,status", [
        (PLAIN, "active"),          # no verify right
        (MAINTAINER, ""),           # did not ask for active outright
        (MAINTAINER, "unverified"),
        (AGENT, "active"),          # forced to unverified
        (FORCED_VERIFIER, "active"),
    ])
    def test_everyone_else_leaves_it_alone(self, tokens, who, status):
        _learn(AGENT, NOTE)
        assert _learn(who, NOTE, status=status) == server._det_id("mem", NOTE)
        assert _meta(tokens, NOTE)["status"] == "unverified"

    def test_an_active_note_is_never_demoted(self, tokens):
        _learn(PLAIN, NOTE)
        for who in (AGENT, MAINTAINER, PLAIN):
            _learn(who, NOTE, status="unverified")
        assert _meta(tokens, NOTE)["status"] == "active"

    def test_the_local_owner_may_promote(self, rag):
        server.learn(NOTE, project="p", status="unverified")
        assert server.learn(NOTE, project="p", status="active").endswith("promoted from unverified")


class TestOverflowAndStats:
    def test_overflow_archives_the_oldest_live_notes_unverified_ones_included(self, rag, monkeypatch):
        monkeypatch.setattr(server, "MEMORY_CAP", 3)
        stamps = iter(f"2026-01-01T00:00:{n:02d}Z" for n in range(60))
        monkeypatch.setattr(server, "_now", lambda: next(stamps))
        texts = [f"unverified claim number {n}" for n in range(5)]
        for text in texts:
            server.learn(text, project="p", status="unverified")
        statuses = [_meta(rag, t)["status"] for t in texts]
        assert statuses == ["archived", "archived", "unverified", "unverified", "unverified"]
        assert server._staging_count(_memory(rag)) == 3

    @pytest.fixture
    def clock(self, monkeypatch):
        stamps = iter(f"2026-01-01T00:00:{n:02d}Z" for n in range(90))
        monkeypatch.setattr(server, "_now", lambda: next(stamps))

    def _statuses(self, rag, texts):
        return [_meta(rag, t)["status"] for t in texts]

    def test_agents_cannot_push_verified_notes_into_the_archive(self, rag, monkeypatch, clock):
        monkeypatch.setattr(server, "MEMORY_CAP", 3)
        active = [f"verified fact number {n}" for n in range(3)]
        claims = [f"agent claim number {n}" for n in range(4)]
        for text in active:
            server.learn(text, project="p")
        for text in claims:
            server.learn(text, project="p", status="unverified")
        assert self._statuses(rag, active) == ["active"] * 3
        assert self._statuses(rag, claims) == ["archived"] * 4
        assert server._staging_count(_memory(rag)) == 3

    def test_unverified_notes_go_first_then_the_oldest_active_for_the_excess(
            self, rag, monkeypatch, clock):
        monkeypatch.setattr(server, "MEMORY_CAP", 3)
        active = [f"verified fact number {n}" for n in range(4)]
        server.learn(active[0], project="p")
        server.learn("agent claim", project="p", status="unverified")
        for text in active[1:]:
            server.learn(text, project="p")
        # 4 active + 1 unverified over a cap of 3: the claim, then the oldest fact
        assert _meta(rag, "agent claim")["status"] == "archived"
        assert self._statuses(rag, active) == ["archived", "active", "active", "active"]

    def test_the_oldest_unverified_note_is_archived_before_a_newer_one(
            self, rag, monkeypatch, clock):
        monkeypatch.setattr(server, "MEMORY_CAP", 4)
        server.learn("fact one", project="p")
        server.learn("fact two", project="p")
        claims = [f"agent claim {n}" for n in range(4)]
        for text in claims:
            server.learn(text, project="p", status="unverified")
        assert self._statuses(rag, claims) == ["archived", "archived", "unverified", "unverified"]
        assert _meta(rag, "fact one")["status"] == "active"

    def test_stats_counts_unverified_notes(self, rag):
        server.learn(NOTE, project="p", status="unverified")
        server.learn(OTHER, project="p")
        report = json.loads(server.stats(project="p"))
        assert report["unverified_notes"] == 1 and report["staged_notes"] == 2

    def test_stats_reports_zero_without_a_memory_layer(self, rag):
        assert json.loads(server.stats(project="empty"))["unverified_notes"] == 0
        assert not rag.client.list_collections()


class TestOlderStores:
    def test_a_note_written_before_statuses_still_answers(self, rag):
        col = _memory(rag)
        col.upsert(ids=[server._det_id("mem", NOTE)], embeddings=[[0.1] * 64], documents=[NOTE],
                   metadatas=[{"path": "memory", "layer": "memory", "kind": "learning"}])
        out = server.recall("deploy lane", project="p", layer="memory")
        assert NOTE in out and "unverified" not in out
        assert "is active, not unverified" in server.verify(_short(NOTE), project="p")


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestScripts:
    def test_consolidation_never_picks_up_unverified_notes(self, rag, monkeypatch):
        monkeypatch.setenv("HERMES_RAG_STORE", str(rag.store))
        monkeypatch.setenv("MEMORY_RAG_LOCAL_NAMESPACE", "legacy")
        server.learn(NOTE, project="p", status="unverified")
        server.learn(OTHER, project="p")
        server.learn("a third, archived-looking thing", project="p")
        col = rag.client.get_collection("rag_p__memory")
        third = server._det_id("mem", "a third, archived-looking thing")
        col.update(ids=[third], metadatas=[{**col.get(ids=[third], include=["metadatas"])
                                            ["metadatas"][0], "status": "archived"}])
        nodes = _script("consolidate")._staging_nodes("p")
        assert [n["text"] for n in nodes.values()] == [OTHER]

    def test_the_weekly_review_lists_old_unverified_notes(self, rag, monkeypatch):
        monkeypatch.setenv("MEMORY_RAG_LOCAL_NAMESPACE", "legacy")
        server.learn(NOTE, project="p", status="unverified")
        server.learn(OTHER, project="p", status="unverified")
        server.learn("an active note", project="p")
        col = rag.client.get_collection("rag_p__memory")
        old = server._det_id("mem", NOTE)
        col.update(ids=[old], metadatas=[{**col.get(ids=[old], include=["metadatas"])
                                          ["metadatas"][0], "created": "2020-01-01T00:00:00Z"}])
        review = _script("weekly_review")
        found = review.stale_unverified(rag.client, "rag_p__memory", days=7)
        assert [(i, t) for i, t, _ in found] == [(old, NOTE)]
        assert found[0][2] > 1000
        # and they are not counted as staged for a merge
        assert review.inventory(rag.client, "p", {"memory": "rag_p__memory"})["staged"] == 1
