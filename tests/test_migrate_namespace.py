"""scripts/migrate_namespace.py: copying a namespace's collections to another one.

The contract these tests hold to:

- every collection of the source namespace is copied with the same ids, documents,
  metadatas and embeddings, nothing re-embedded, and the source is left untouched;
- after copying legacy -> vps-host, a vps-host token recalls exactly what the
  legacy token did, and stats reports the same counts (the ledger is copied too);
- a `namespace` field in metadata naming the source is rewritten, nothing else is;
- an existing target is refused unless --force; --dry-run writes nothing;
- a store some other process holds open is refused unless --i-stopped-the-service.

The store is built through the server's own tools (learn, retire, ingest_*), with a
deterministic bag-of-words embedder instead of Ollama so ranking is meaningful.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_memory_rag import server

ROOT = Path(__file__).resolve().parents[1]
TOKENS = {"legacy": "tok-mig-legacy", "vps-host": "tok-mig-vps", "acme": "tok-mig-acme"}


def _load_script():
    spec = importlib.util.spec_from_file_location("migrate_namespace",
                                                  ROOT / "scripts" / "migrate_namespace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mig = _load_script()


def _digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def _hash_embedding(text: str, dim: int = 32) -> list[float]:
    vec = [0.0] * dim
    for word in text.lower().split():
        h = int(hashlib.md5(word.strip(".,:;!?#").encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 8) % 2 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _hash_embed_model():
    from llama_index.core.embeddings import BaseEmbedding

    class HashEmbedding(BaseEmbedding):
        def _get_text_embedding(self, text):
            return _hash_embedding(text)

        def _get_query_embedding(self, query):
            return _hash_embedding(query)

        async def _aget_query_embedding(self, query):
            return _hash_embedding(query)

    return HashEmbedding(model_name="hash")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A store built by the server as the legacy local owner, and a tokens file."""
    import chromadb
    from llama_index.core import Settings

    store = tmp_path / "store"
    store.mkdir()
    client = chromadb.PersistentClient(path=str(store))
    embed = _hash_embed_model()
    monkeypatch.setattr(server, "STORE_DIR", store)
    monkeypatch.setattr(server, "_client", lambda: client)
    monkeypatch.setattr(server, "_embed_model", lambda: embed)
    monkeypatch.setattr(Settings, "_embed_model", embed)
    monkeypatch.setattr(server, "CONSOLIDATE_THRESHOLD", 999)
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "stdio")
    monkeypatch.delenv("MEMORY_RAG_LOCAL_NAMESPACE", raising=False)

    cfg = tmp_path / "cfg"
    cfg.mkdir(mode=0o700)
    tokens = cfg / "tokens.json"
    tokens.write_text(json.dumps({
        _digest(TOKENS["legacy"]): {"namespace": "legacy"},
        _digest(TOKENS["vps-host"]): {"namespace": "vps-host"},
        _digest(TOKENS["acme"]): {"namespace": "acme"},
    }))
    tokens.chmod(0o600)
    monkeypatch.setattr(server, "TOKENS_FILE", tokens)

    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "arch.md").write_text(
        "# Architecture\n\n## Pagination\nThe todo api pages results with an opaque cursor.\n\n"
        "## Storage\nTodos live in postgres, one table per tenant.\n\n"
        "## Auth\nRequests carry a bearer token checked by middleware.\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "api.py").write_text("def list_todos(cursor):\n    return page_after(cursor)\n")
    (repo / "db.py").write_text("def connect():\n    return postgres_pool()\n")

    server.ingest_wiki(str(wiki), "todo-api")
    server.ingest_code(str(repo), "todo-api")
    for text in ("cursor pagination must stay opaque to clients",
                 "postgres migrations run before deploy",
                 "the deploy lane is blue on fridays"):
        server.learn(text, project="todo-api")
    old = server.learn("tokens are checked in the router", project="todo-api")
    server.learn("tokens are checked by middleware, not the router", project="todo-api",
                 supersedes=old[:8])
    wrong = server.learn("pagination uses page numbers", project="todo-api")
    server.retire(wrong[:8], reason="wrong", project="todo-api")
    server.learn("always answer in the user's language", project="global")
    server.learn("web landing uses a static build", project="web-landing")
    # A note that names its namespace in metadata, as some importer might write it.
    col = client.get_collection("rag_web-landing__memory")
    col.add(ids=["ns-note"], documents=["imported note about the static build"],
            embeddings=[_hash_embedding("imported note about the static build")],
            metadatas=[{"path": "memory", "layer": "memory", "kind": "learning",
                        "status": "active", "source": "import", "namespace": "legacy",
                        "created": "2026-10-01T00:00:00Z"}])
    # Something in another namespace, which must not move.
    monkeypatch.setenv("MEMORY_RAG_LOCAL_NAMESPACE", "acme")
    server.learn("acme keeps its own notes", project="todo-api")
    monkeypatch.delenv("MEMORY_RAG_LOCAL_NAMESPACE")
    return client, store


def as_token(name, fn, *args, **kwargs):
    """Call a server tool as the bearer token `name`, as the HTTP layer would."""
    from mcp.server.auth.middleware.auth_context import auth_context_var
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from mcp.server.auth.provider import AccessToken

    digest = _digest(TOKENS[name])
    reset = auth_context_var.set(AuthenticatedUser(
        AccessToken(token=digest, client_id=f"{name}:{digest[:12]}", scopes=[])))
    try:
        return fn(*args, **kwargs)
    finally:
        auth_context_var.reset(reset)


QUERIES = [("todo-api", "how does pagination with a cursor work"),
           ("todo-api", "where do todos live postgres"),
           ("todo-api", "who checks the bearer tokens"),
           ("todo-api", "deploy lane"),
           ("web-landing", "static build"),
           ("other", "answer language")]


def _recalls(token):
    return [as_token(token, server.recall, q, project=p, top_memory=3).split("\n", 1)[1]
            for p, q in QUERIES]


def _stats(token, project):
    out = json.loads(as_token(token, server.stats, project=project))
    out.pop("project")
    return out


def _dump(client, name):
    got = client.get_collection(name).get(include=["documents", "metadatas", "embeddings"])
    return {i: (d, m, list(e)) for i, d, m, e in
            zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"])}


def _run(store, *args):
    out = io.StringIO()
    rc = mig.migrate(store, "legacy", "vps-host", out=out, **dict(args))
    return rc, out.getvalue()


def test_names_are_parsed_per_namespace():
    assert mig.split_name("rag_todo-api__memory", "legacy") == ("todo-api", "memory")
    assert mig.split_name("rag_acme.todo-api__memory", "legacy") is None
    assert mig.split_name("rag_acme.todo-api__memory", "acme") == ("todo-api", "memory")
    assert mig.split_name("rag_acme.todo-api__other", "acme") is None
    assert mig.target_name("todo-api", "wiki", "vps-host") == "rag_vps-host.todo-api__wiki"
    assert mig.target_name("todo-api", "wiki", "legacy") == "rag_todo-api__wiki"


def test_migrated_namespace_recalls_what_legacy_did(env):
    client, store = env
    before = _recalls("legacy")
    assert any("opaque cursor" in r for r in before)  # the store really answers
    assert all("page numbers" not in r for r in before)  # retired stays hidden
    legacy_stats = _stats("legacy", "todo-api")
    legacy_names = sorted(c.name for c in client.list_collections() if "." not in c.name)
    legacy_dump = {n: _dump(client, n) for n in legacy_names}
    assert not [c for c in client.list_collections() if c.name.startswith("rag_vps-host.")]

    rc, out = _run(store, ("service_stopped", False))
    assert rc == 0, out
    assert "verified" in out and "VERIFY FAILED" not in out

    # stats first: every recall adds a ledger row, which stats counts.
    assert _stats("vps-host", "todo-api") == legacy_stats
    assert _recalls("vps-host") == before
    # The source is untouched and still answers the legacy token.
    assert {n: _dump(client, n) for n in legacy_names} == legacy_dump
    assert _recalls("legacy") == before
    # And still nothing leaks to another namespace.
    assert all("opaque cursor" not in r for r in _recalls("acme"))


def test_copy_is_exact_and_only_the_namespace_field_changes(env):
    client, store = env
    names = sorted(c.name for c in client.list_collections() if "." not in c.name)
    assert _run(store)[0] == 0
    for name in names:
        new = name.replace("rag_", "rag_vps-host.", 1)
        src, dst = _dump(client, name), _dump(client, new)
        assert src.keys() == dst.keys()
        for i, (doc, meta, emb) in src.items():
            d2, m2, e2 = dst[i]
            assert doc == d2 and emb == e2
            expected = dict(meta)
            if expected.get("namespace") == "legacy":
                expected["namespace"] = "vps-host"
            assert m2 == expected
        assert (client.get_collection(name).configuration_json["hnsw"]
                == client.get_collection(new).configuration_json["hnsw"])
    assert _dump(client, "rag_vps-host.web-landing__memory")["ns-note"][1]["namespace"] == "vps-host"
    assert _dump(client, "rag_web-landing__memory")["ns-note"][1]["namespace"] == "legacy"
    # Other namespaces are not part of the plan.
    assert "rag_vps-host.acme.todo-api__memory" not in {c.name for c in client.list_collections()}


def test_dry_run_writes_nothing(env):
    client, store = env
    names = {c.name for c in client.list_collections()}
    ledger = (store / "usage.db").read_bytes()
    rc, out = _run(store, ("dry_run", True))
    assert rc == 0
    assert "rag_todo-api__memory (" in out and "-> rag_vps-host.todo-api__memory" in out
    assert "rag_acme.todo-api__memory" not in out
    assert {c.name for c in client.list_collections()} == names
    assert (store / "usage.db").read_bytes() == ledger


def test_existing_target_is_refused_without_force(env):
    client, store = env
    assert _run(store)[0] == 0
    with pytest.raises(mig.Refused, match="--force"):
        _run(store)
    before = _recalls("vps-host")
    rc, out = _run(store, ("force", True))
    assert rc == 0 and "VERIFY FAILED" not in out
    assert _recalls("vps-host") == before
    legacy, vps = _stats("legacy", "todo-api"), _stats("vps-host", "todo-api")
    assert vps["layers"] == legacy["layers"] and vps["staged_notes"] == legacy["staged_notes"]
    assert vps["learnings_deposited"] == legacy["learnings_deposited"]  # not duplicated


def test_bad_arguments_are_refused(env, tmp_path):
    _, store = env
    with pytest.raises(mig.Refused, match="not a valid namespace"):
        mig.migrate(store, "legacy", "Bad.NS", out=io.StringIO())
    with pytest.raises(mig.Refused, match="same"):
        mig.migrate(store, "legacy", "legacy", out=io.StringIO())
    with pytest.raises(mig.Refused, match="no Chroma store"):
        mig.migrate(tmp_path / "nowhere", "legacy", "vps-host", out=io.StringIO())
    with pytest.raises(mig.Refused, match="no collections"):
        mig.migrate(store, "globex", "vps-host", out=io.StringIO())


def test_a_store_held_open_by_another_process_is_refused(env):
    client, store = env
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; f = open(sys.argv[1], 'rb'); print('ready', flush=True); time.sleep(60)",
         str(store / "chroma.sqlite3")], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        with pytest.raises(mig.Refused, match="held open by pid"):
            _run(store)
        with pytest.raises(mig.Refused, match="held open"):
            _run(store, ("dry_run", True))
        assert not [c for c in client.list_collections() if c.name.startswith("rag_vps-host.")]
        assert mig.main(["--store", str(store), "--to", "vps-host"]) == 1
        rc, out = _run(store, ("service_stopped", True))
        assert rc == 0, out
    finally:
        holder.kill()
        holder.wait()
