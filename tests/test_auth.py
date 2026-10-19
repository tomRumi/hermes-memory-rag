"""Bearer-token access: who may call, and what each caller can reach.

The contract these tests hold to:

- over HTTP, a request with no token, a wrong token, or a token the file does not
  (or can no longer) vouch for is refused with 401, and a tokens file that is
  missing, a symlink, readable by others, in a directory others can write, or
  malformed vouches for nobody;
- what a token may do is read from the file on every call, so revoking it or
  changing its ingest right takes effect inside a session that is already open;
- a token maps to one namespace, and everything it reads or writes is confined to
  it, including the ledger; another namespace cannot recall or retire its notes,
  whatever project name it passes;
- ingest_* is refused unless the token's entry allows it;
- a token whose entry says "write": false may call only the read tools; every tool
  that changes the store refuses it before touching anything. "write" defaults to
  true, and "ingest" together with "write": false refuses that entry;
- the "legacy" namespace is the store as it was before namespaces, so data
  written without a prefix stays reachable through it and through nothing else;
- the local owner (stdio, in-process scripts) writes into MEMORY_RAG_LOCAL_NAMESPACE,
  "legacy" when unset, always with ingest and write allowed.

The tests drive the real streamable-HTTP app, so the middleware, the session
binding and the tools are exercised together. Embeddings are a mock: no Ollama.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest
from starlette.testclient import TestClient

from hermes_memory_rag import server

URL = "http://127.0.0.1:8765"
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
TOKENS = {"acme": "tok-acme-secret", "globex": "tok-globex-secret",
          "legacy": "tok-legacy-secret", "acme-ingest": "tok-acme-ingest-secret",
          "acme-ro": "tok-acme-readonly-secret", "legacy-ro": "tok-legacy-readonly-secret",
          "vps-host": "tok-vps-host-secret"}


def _digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def _write_tokens(path, entries, mode=0o600):
    path.write_text(json.dumps(entries))
    path.chmod(mode)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A Chroma store in a temp dir, a mock embedder, served as if over HTTP."""
    import chromadb
    from llama_index.core import Settings
    from llama_index.core.embeddings import MockEmbedding

    path = tmp_path / "chroma"
    path.mkdir()
    client = chromadb.PersistentClient(path=str(path))
    monkeypatch.setattr(server, "STORE_DIR", path)
    monkeypatch.setattr(server, "_client", lambda: client)
    monkeypatch.setattr(server, "_embed_model", lambda: MockEmbedding(embed_dim=16))
    Settings.embed_model = MockEmbedding(embed_dim=16)
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "streamable-http")
    return client


@pytest.fixture
def tokens_file(tmp_path, monkeypatch):
    (tmp_path / "cfg").mkdir(mode=0o700)
    path = tmp_path / "cfg" / "tokens.json"
    _write_tokens(path, {
        _digest(TOKENS["acme"]): {"namespace": "acme"},
        _digest(TOKENS["globex"]): {"namespace": "globex"},
        _digest(TOKENS["legacy"]): {"namespace": "legacy", "ingest": True},
        _digest(TOKENS["acme-ingest"]): {"namespace": "acme", "ingest": True},
        _digest(TOKENS["acme-ro"]): {"namespace": "acme", "write": False},
        _digest(TOKENS["legacy-ro"]): {"namespace": "legacy", "write": False},
        _digest(TOKENS["vps-host"]): {"namespace": "vps-host", "write": False},
    })
    monkeypatch.setattr(server, "TOKENS_FILE", path)
    return path


@pytest.fixture(scope="module")
def http():
    """One server app for the module: its session manager can only run once."""
    with TestClient(server.mcp.streamable_http_app(), base_url=URL) as client:
        yield client


def _rpc(http, token, method, params=None, session=None, id_=1):
    headers = dict(HEADERS)
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if session:
        headers["mcp-session-id"] = session
    body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if id_ is not None:
        body["id"] = id_
    return http.post("/mcp", headers=headers, json=body)


def _payload(response):
    """The JSON-RPC message in a response, whether sent as JSON or as one SSE event."""
    if response.headers["content-type"].startswith("text/event-stream"):
        data = [ln[6:] for ln in response.text.splitlines() if ln.startswith("data: ")]
        return json.loads(data[-1])
    return response.json()


def _open(http, token):
    """Initialise a session as `token`; returns its id."""
    response = _rpc(http, token, "initialize", {
        "protocolVersion": "2025-03-26", "capabilities": {},
        "clientInfo": {"name": "t", "version": "0"}})
    assert response.status_code == 200, response.text
    session = response.headers["mcp-session-id"]
    _rpc(http, token, "notifications/initialized", session=session, id_=None)
    return session


def call(http, token, tool, **arguments):
    """Call a tool as `token`; returns (is_error, text)."""
    session = _open(http, token)
    response = _rpc(http, token, "tools/call", {"name": tool, "arguments": arguments},
                    session=session, id_=2)
    assert response.status_code == 200, response.text
    result = _payload(response)["result"]
    return result["isError"], "".join(c["text"] for c in result["content"])


# ── who may call ──────────────────────────────────────────────────────────


def test_no_token_is_refused(http, store, tokens_file):
    assert _rpc(http, None, "tools/list").status_code == 401


def test_wrong_token_is_refused(http, store, tokens_file):
    assert _rpc(http, "not-a-token", "tools/list").status_code == 401


def test_other_scheme_is_refused(http, store, tokens_file):
    headers = {**HEADERS, "Authorization": f"Basic {TOKENS['acme']}"}
    response = http.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert response.status_code == 401


def test_a_valid_token_is_accepted(http, store, tokens_file):
    assert _rpc(http, TOKENS["acme"], "tools/list").status_code in (200, 400)  # 400: no session yet
    assert _open(http, TOKENS["acme"])


def test_token_is_checked_by_hash_not_stored_in_clear(tokens_file):
    assert TOKENS["acme"] not in tokens_file.read_text()


def test_missing_tokens_file_vouches_for_nobody(http, store, tokens_file):
    tokens_file.unlink()
    assert _rpc(http, TOKENS["acme"], "tools/list").status_code == 401


def test_tokens_file_open_to_others_vouches_for_nobody(http, store, tokens_file):
    tokens_file.chmod(0o644)
    assert _rpc(http, TOKENS["acme"], "tools/list").status_code == 401


def test_malformed_tokens_file_vouches_for_nobody(http, store, tokens_file):
    _write_tokens(tokens_file, {_digest(TOKENS["acme"]): {"namespace": "Not Valid!"}})
    assert _rpc(http, TOKENS["acme"], "tools/list").status_code == 401


def test_revoking_a_token_takes_effect_without_restart(http, store, tokens_file):
    assert _open(http, TOKENS["globex"])
    _write_tokens(tokens_file, {_digest(TOKENS["acme"]): {"namespace": "acme"}})
    os.utime(tokens_file, ns=(1, 1))  # a new mtime, whatever the clock resolution
    assert _rpc(http, TOKENS["globex"], "tools/list").status_code == 401
    assert _open(http, TOKENS["acme"])


def test_a_session_cannot_be_borrowed_by_another_token(http, store, tokens_file):
    session = _open(http, TOKENS["acme"])
    response = _rpc(http, TOKENS["globex"], "tools/list", session=session)
    assert response.status_code >= 400


def test_tokens_file_that_is_a_symlink_vouches_for_nobody(http, store, tokens_file):
    real = tokens_file.with_name("real.json")
    tokens_file.rename(real)
    tokens_file.symlink_to(real)
    assert _rpc(http, TOKENS["acme"], "tools/list").status_code == 401
    with pytest.raises(ValueError, match="cannot read"):
        server._read_tokens(tokens_file)


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o660, 0o666])
def test_tokens_file_with_group_or_other_access_is_refused(tokens_file, mode):
    tokens_file.chmod(mode)
    with pytest.raises(ValueError, match="mode 600"):
        server._read_tokens(tokens_file)


@pytest.mark.parametrize("mode", [0o750, 0o705, 0o755, 0o777])
def test_tokens_file_in_an_open_directory_is_refused(http, store, tokens_file, mode):
    tokens_file.parent.chmod(mode)
    try:
        with pytest.raises(ValueError, match="mode 700"):
            server._read_tokens(tokens_file)
        assert _rpc(http, TOKENS["acme"], "tools/list").status_code == 401
    finally:
        tokens_file.parent.chmod(0o700)


def test_tokens_file_in_a_symlinked_directory_is_refused(http, store, tokens_file):
    link = tokens_file.parent.with_name("cfg-link")
    link.symlink_to(tokens_file.parent)
    with pytest.raises(ValueError, match="real directory"):
        server._read_tokens(link / "tokens.json")


def test_opening_up_the_directory_later_is_noticed(http, store, tokens_file):
    assert _open(http, TOKENS["acme"])
    tokens_file.parent.chmod(0o755)
    try:
        assert _rpc(http, TOKENS["acme"], "tools/list").status_code == 401
    finally:
        tokens_file.parent.chmod(0o700)
    assert _open(http, TOKENS["acme"])


def test_replacing_the_file_with_the_same_size_and_mtime_is_noticed(http, store, tokens_file):
    """The reload key is more than the mtime: inode and ctime change on a swap."""
    _write_tokens(tokens_file, {_digest(TOKENS["acme"]): {"namespace": "acme"}})
    assert _open(http, TOKENS["acme"])
    before = tokens_file.stat()
    other = tokens_file.with_name("other.json")
    _write_tokens(other, {_digest(TOKENS["globex"]): {"namespace": "acme"}})
    assert other.stat().st_size == before.st_size
    os.replace(other, tokens_file)
    os.utime(tokens_file, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert _rpc(http, TOKENS["acme"], "tools/list").status_code == 401
    assert _open(http, TOKENS["globex"])


def test_server_will_not_start_without_usable_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "streamable-http")
    monkeypatch.setattr(server, "TOKENS_FILE", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="refusing to serve"):
        server.main()
    empty = tmp_path / "empty.json"
    _write_tokens(empty, {})
    monkeypatch.setattr(server, "TOKENS_FILE", empty)
    with pytest.raises(SystemExit, match="no tokens"):
        server.main()


@pytest.mark.parametrize("entry", [
    {"namespace": "bad/ns"}, {"namespace": ""}, {"namespace": "a.b"}, {"namespace": 7}, {}, "acme"])
def test_malformed_entries_reject_the_whole_file(tmp_path, entry):
    path = tmp_path / "t.json"
    _write_tokens(path, {_digest("x"): {"namespace": "ok"}, _digest("y"): entry})
    with pytest.raises(ValueError, match="malformed"):
        server._read_tokens(path)


def test_tool_without_credentials_over_http_is_denied(store, tokens_file):
    """Reached some way that skips the middleware: the tools still refuse."""
    with pytest.raises(PermissionError, match="authentication required"):
        server.recall("anything", project="p")


def test_stdio_caller_is_the_local_owner(store, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "stdio")
    monkeypatch.delenv("MEMORY_RAG_LOCAL_NAMESPACE", raising=False)
    assert server._caller() == server.Caller("legacy", True)


# ── what each caller can reach ────────────────────────────────────────────

NOTE = "acme prefers the blue deployment lane"


def test_namespaces_do_not_see_each_other(http, store, tokens_file):
    err, out = call(http, TOKENS["acme"], "learn", text=NOTE, project="shared")
    assert not err, out
    err, out = call(http, TOKENS["acme"], "recall", query="deployment lane", project="shared")
    assert NOTE in out
    err, out = call(http, TOKENS["globex"], "recall", query="deployment lane", project="shared")
    assert not err and NOTE not in out


def test_another_namespace_cannot_retire_a_note(http, store, tokens_file):
    _, node_id = call(http, TOKENS["acme"], "learn", text=NOTE, project="shared")
    err, out = call(http, TOKENS["globex"], "retire", node_id=node_id[:8], project="shared")
    assert "nothing changed" in out
    err, out = call(http, TOKENS["acme"], "recall", query="deployment lane", project="shared")
    assert NOTE in out  # still live
    err, out = call(http, TOKENS["acme"], "retire", node_id=node_id[:8], project="shared")
    assert out.startswith("retired")


def test_same_namespace_tokens_share_notes(http, store, tokens_file):
    call(http, TOKENS["acme"], "learn", text=NOTE, project="shared")
    _, out = call(http, TOKENS["acme-ingest"], "recall", query="deployment lane", project="shared")
    assert NOTE in out


@pytest.mark.parametrize("project", [
    "acme.shared", "acme/shared", "../acme/shared", "acme-shared", "rag_acme.shared", "shared"])
def test_a_project_name_cannot_reach_into_another_namespace(http, store, tokens_file, project):
    call(http, TOKENS["acme"], "learn", text=NOTE, project="shared")
    err, out = call(http, TOKENS["globex"], "recall", query="deployment lane", project=project)
    assert not err and NOTE not in out


def test_collections_are_prefixed_by_namespace(http, store, tokens_file):
    call(http, TOKENS["acme"], "learn", text=NOTE, project="Shared Proj")
    call(http, TOKENS["globex"], "learn", text="other", project="Shared Proj")
    names = {c.name for c in store.list_collections()}
    assert {"rag_acme.shared-proj__memory", "rag_globex.shared-proj__memory"} <= names
    assert "rag_shared-proj__memory" not in names


def test_global_layer_is_per_namespace(http, store, tokens_file):
    call(http, TOKENS["acme"], "learn", text=NOTE, project="global")
    _, out = call(http, TOKENS["globex"], "recall", query="deployment lane", project="anything")
    assert NOTE not in out
    _, out = call(http, TOKENS["acme"], "recall", query="deployment lane", project="anything")
    assert NOTE in out and "[global]" in out


def test_ledger_and_stats_are_per_namespace(http, store, tokens_file):
    call(http, TOKENS["acme"], "learn", text=NOTE, project="shared")
    acme = json.loads(call(http, TOKENS["acme"], "stats", project="shared")[1])
    globex = json.loads(call(http, TOKENS["globex"], "stats", project="shared")[1])
    assert acme["learnings_deposited"] == 1 and acme["staged_notes"] == 1
    assert globex["learnings_deposited"] == 0 and globex["staged_notes"] == 0
    import sqlite3
    rows = sqlite3.connect(str(server.STORE_DIR / "usage.db")).execute(
        "SELECT DISTINCT project FROM events").fetchall()
    assert ("acme.shared",) in rows and ("shared",) not in rows


# ── ingest ────────────────────────────────────────────────────────────────


@pytest.fixture
def wiki(tmp_path):
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "a.md").write_text("# A\n\n## Setup\nhow it is set up\n")
    return root


@pytest.mark.parametrize("tool,arg", [("ingest_wiki", "wiki_dir"), ("ingest_code", "root")])
def test_ingest_is_refused_for_an_agent_token(http, store, tokens_file, wiki, tool, arg):
    err, out = call(http, TOKENS["acme"], tool, project="p", **{arg: str(wiki)})
    assert err and "may not run ingest" in out
    assert not [c for c in store.list_collections() if "acme.p" in c.name]


def test_ingest_follows_the_tokens_flag(http, store, tokens_file, wiki):
    err, out = call(http, TOKENS["acme-ingest"], "ingest_wiki", wiki_dir=str(wiki), project="p")
    assert not err, out
    assert "rag_acme.p__wiki" in {c.name for c in store.list_collections()}
    err, out = call(http, TOKENS["legacy"], "ingest_wiki", wiki_dir=str(wiki), project="p")
    assert not err, out
    assert "rag_p__wiki" in {c.name for c in store.list_collections()}


# ── legacy ────────────────────────────────────────────────────────────────


def _old_note(store, project, text):
    """A note exactly as the server wrote it before namespaces existed."""
    col = store.get_or_create_collection(f"rag_{project}__memory")
    col.upsert(ids=[server._det_id("mem", text)], embeddings=[[0.1] * 16], documents=[text],
               metadatas=[{"path": "memory", "layer": "memory", "kind": "learning",
                           "status": "active", "source": "learn", "created": "2026-10-10T00:00:00Z"}])
    return server._det_id("mem", text)


def test_legacy_namespace_sees_existing_data(http, store, tokens_file):
    node = _old_note(store, "todo-api", "the todo api uses cursor pagination")
    err, out = call(http, TOKENS["legacy"], "recall", query="pagination", project="todo-api")
    assert not err and "cursor pagination" in out and f"#{node[:8]}" in out
    stats = json.loads(call(http, TOKENS["legacy"], "stats", project="todo-api")[1])
    assert stats["layers"]["memory"] == 1 and stats["staged_notes"] == 1


def test_legacy_data_is_not_reachable_from_other_namespaces(http, store, tokens_file):
    _old_note(store, "todo-api", "the todo api uses cursor pagination")
    for project in ("todo-api", "legacy.todo-api", "legacy/todo-api"):
        err, out = call(http, TOKENS["acme"], "recall", query="pagination", project=project)
        assert not err and "cursor pagination" not in out
        _, out = call(http, TOKENS["acme"], "retire", node_id="0", project=project)
        assert "nothing changed" in out


def test_legacy_writes_keep_the_old_layout(http, store, tokens_file):
    call(http, TOKENS["legacy"], "learn", text=NOTE, project="todo-api")
    assert "rag_todo-api__memory" in {c.name for c in store.list_collections()}


# ── rights are read at call time, not pinned at session start ────────────


def _in_session(http, token, session, tool, **arguments):
    response = _rpc(http, token, "tools/call", {"name": tool, "arguments": arguments},
                    session=session, id_=3)
    if response.status_code != 200:
        return True, f"http {response.status_code}"
    result = _payload(response)["result"]
    return result["isError"], "".join(c["text"] for c in result["content"])


def test_revoking_a_token_denies_inside_an_open_session(http, store, tokens_file):
    session = _open(http, TOKENS["acme"])
    assert not _in_session(http, TOKENS["acme"], session, "stats", project="p")[0]
    _write_tokens(tokens_file, {_digest(TOKENS["globex"]): {"namespace": "globex"}})
    err, _ = _in_session(http, TOKENS["acme"], session, "stats", project="p")
    assert err


def test_flipping_ingest_takes_effect_inside_an_open_session(http, store, tokens_file, wiki):
    session = _open(http, TOKENS["acme-ingest"])
    err, out = _in_session(http, TOKENS["acme-ingest"], session, "ingest_wiki",
                           wiki_dir=str(wiki), project="p")
    assert not err, out
    _write_tokens(tokens_file, {_digest(TOKENS["acme-ingest"]): {"namespace": "acme", "ingest": False}})
    err, out = _in_session(http, TOKENS["acme-ingest"], session, "ingest_wiki",
                           wiki_dir=str(wiki), project="p")
    assert err


def test_caller_is_looked_up_again_even_if_the_session_pinned_the_token(store, tokens_file):
    """What a long-lived session sees: the token captured when it opened."""
    from mcp.server.auth.middleware.auth_context import auth_context_var
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from mcp.server.auth.provider import AccessToken

    digest = _digest(TOKENS["acme-ingest"])
    pinned = AccessToken(token=digest, client_id=f"acme:{digest[:12]}", scopes=[])
    reset = auth_context_var.set(AuthenticatedUser(pinned))
    try:
        assert server._caller() == server.Caller("acme", True)
        _write_tokens(tokens_file, {digest: {"namespace": "acme", "ingest": False}})
        assert server._caller() == server.Caller("acme", False)
        _write_tokens(tokens_file, {_digest(TOKENS["globex"]): {"namespace": "globex"}})
        with pytest.raises(PermissionError, match="no longer valid"):
            server._caller()
    finally:
        auth_context_var.reset(reset)


def test_consolidate_script_hint_is_only_for_legacy(http, store, tokens_file, monkeypatch):
    monkeypatch.setattr(server, "CONSOLIDATE_THRESHOLD", 1)
    _, out = call(http, TOKENS["acme"], "learn", text=NOTE, project="p")
    assert "CONSOLIDATION DUE" not in out and "consolidate.py" not in out
    _, out = call(http, TOKENS["legacy"], "learn", text=NOTE, project="p")
    assert "consolidate.py p" in out


# ── read-only tokens ──────────────────────────────────────────────────────

# Arguments that make each tool do real work if it were allowed to run.
def _tool_arguments(wiki):
    return {
        "recall": {"query": "deployment lane", "project": "shared"},
        "stats": {"project": "shared"},
        "learn": {"text": "a read-only token wrote this", "project": "shared"},
        "retire": {"node_id": server._det_id("mem", NOTE)[:8], "project": "shared"},
        "ingest_code": {"root": str(wiki), "project": "shared"},
        "ingest_wiki": {"wiki_dir": str(wiki), "project": "shared"},
    }


def _snapshot(store):
    """Everything a tool could change: every collection's rows, and the ledger."""
    import sqlite3
    cols = {}
    for c in store.list_collections():
        got = store.get_collection(c.name).get(include=["documents", "metadatas"])
        cols[c.name] = sorted(zip(got["ids"], got["documents"],
                                  [json.dumps(m, sort_keys=True) for m in got["metadatas"]]))
    ledger = server.STORE_DIR / "usage.db"
    rows = (sqlite3.connect(str(ledger)).execute("SELECT * FROM events").fetchall()
            if ledger.exists() else [])
    return cols, rows


def _listed_tools(http, token):
    session = _open(http, token)
    response = _rpc(http, token, "tools/list", session=session, id_=2)
    return {t["name"] for t in _payload(response)["result"]["tools"]}


def test_every_tool_is_classified_read_or_write(http, store, tokens_file):
    """A new tool must be classified, or the read-only tests below would miss it."""
    assert _listed_tools(http, TOKENS["acme"]) == set(server.TOOL_ACCESS)
    assert set(server.TOOL_ACCESS.values()) <= {"read", "write"}


@pytest.mark.parametrize("tool", sorted(server.TOOL_ACCESS))
def test_read_only_token_against_every_tool(http, store, tokens_file, wiki, tool):
    call(http, TOKENS["acme"], "learn", text=NOTE, project="shared")
    before = _snapshot(store)
    err, out = call(http, TOKENS["acme-ro"], tool, **_tool_arguments(wiki)[tool])
    if server.TOOL_ACCESS[tool] == "read":
        assert not err, out
        cols, _ = _snapshot(store)
        assert cols == before[0]  # a read changes no node and creates no collection
    else:
        assert err and "read-only" in out, out
        assert _snapshot(store) == before  # refused before anything was touched


def test_read_only_token_reads_its_namespace(http, store, tokens_file):
    call(http, TOKENS["acme"], "learn", text=NOTE, project="shared")
    err, out = call(http, TOKENS["acme-ro"], "recall", query="deployment lane", project="shared")
    assert not err and NOTE in out
    stats = json.loads(call(http, TOKENS["acme-ro"], "stats", project="shared")[1])
    assert stats["layers"]["memory"] == 1 and stats["staged_notes"] == 1


def test_read_only_retire_leaves_the_note_live(http, store, tokens_file):
    _, node_id = call(http, TOKENS["acme"], "learn", text=NOTE, project="shared")
    err, out = call(http, TOKENS["acme-ro"], "retire", node_id=node_id[:8], project="shared")
    assert err and "read-only" in out
    _, out = call(http, TOKENS["acme"], "recall", query="deployment lane", project="shared")
    assert NOTE in out


def test_read_only_stats_creates_no_collection(http, store, tokens_file):
    for tool in ("stats", "recall"):
        args = {"project": "never-written"}
        if tool == "recall":
            args["query"] = "anything"
        err, out = call(http, TOKENS["acme-ro"], tool, **args)
        assert not err, out
    assert not [c for c in store.list_collections() if "never-written" in c.name]


def test_write_defaults_to_true(http, store, tokens_file, tmp_path):
    assert server._read_tokens(tokens_file)[_digest(TOKENS["acme"])].write is True
    err, out = call(http, TOKENS["acme"], "learn", text=NOTE, project="shared")
    assert not err, out
    _write_tokens(tokens_file, {_digest(TOKENS["acme"]): {"namespace": "acme", "write": True}})
    err, out = call(http, TOKENS["acme"], "learn", text="explicitly writable", project="shared")
    assert not err, out


def test_ingest_without_write_refuses_that_entry(http, store, tokens_file, caplog):
    _write_tokens(tokens_file, {
        _digest(TOKENS["acme"]): {"namespace": "acme"},
        _digest(TOKENS["acme-ingest"]): {"namespace": "acme", "ingest": True, "write": False},
    })
    with caplog.at_level("ERROR", logger="hermes_memory_rag"):
        tokens = server._read_tokens(tokens_file)
    assert _digest(TOKENS["acme-ingest"]) not in tokens
    assert "ingest without write" in caplog.text
    assert _rpc(http, TOKENS["acme-ingest"], "tools/list").status_code == 401
    assert _open(http, TOKENS["acme"])  # the rest of the file still counts


def test_only_contradictory_entries_leave_nothing_to_serve(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "streamable-http")
    path = tmp_path / "t.json"
    _write_tokens(path, {_digest("x"): {"namespace": "acme", "ingest": True, "write": False}})
    monkeypatch.setattr(server, "TOKENS_FILE", path)
    with pytest.raises(SystemExit, match="no tokens"):
        server.main()


@pytest.mark.parametrize("write", ["false", "no", 0, 1, None, []])
def test_non_boolean_write_rejects_the_whole_file(tmp_path, write):
    """"false" as a string must never be read as writable, nor guessed at."""
    path = tmp_path / "t.json"
    _write_tokens(path, {_digest("x"): {"namespace": "ok"},
                         _digest("y"): {"namespace": "ok", "write": write}})
    with pytest.raises(ValueError, match="malformed"):
        server._read_tokens(path)


def test_read_only_token_stays_in_its_namespace(http, store, tokens_file):
    call(http, TOKENS["globex"], "learn", text=NOTE, project="shared")
    _old_note(store, "shared", "legacy knows the old pagination")
    for project in ("shared", "globex.shared", "../globex/shared", "legacy.shared"):
        err, out = call(http, TOKENS["acme-ro"], "recall", query="deployment lane", project=project)
        assert not err and NOTE not in out and "pagination" not in out
    err, out = call(http, TOKENS["legacy-ro"], "recall", query="pagination", project="shared")
    assert not err and "old pagination" in out and NOTE not in out
    err, out = call(http, TOKENS["legacy-ro"], "learn", text="sneak", project="shared")
    assert err and "read-only" in out


def test_turning_write_off_takes_effect_inside_an_open_session(http, store, tokens_file):
    session = _open(http, TOKENS["acme"])
    err, out = _in_session(http, TOKENS["acme"], session, "learn", text=NOTE, project="p")
    assert not err, out
    _write_tokens(tokens_file, {_digest(TOKENS["acme"]): {"namespace": "acme", "write": False}})
    err, out = _in_session(http, TOKENS["acme"], session, "learn", text="later", project="p")
    assert err and "read-only" in out
    assert not _in_session(http, TOKENS["acme"], session, "recall", query="x", project="p")[0]


def test_read_only_is_refused_even_without_the_http_layer(store, tokens_file):
    """The check lives in the tools, not only in the transport."""
    from mcp.server.auth.middleware.auth_context import auth_context_var
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from mcp.server.auth.provider import AccessToken

    digest = _digest(TOKENS["acme-ro"])
    reset = auth_context_var.set(AuthenticatedUser(
        AccessToken(token=digest, client_id=f"acme:{digest[:12]}", scopes=[])))
    try:
        assert server._caller() == server.Caller("acme", False, False)
        with pytest.raises(PermissionError, match="read-only"):
            server.learn("x", project="p")
        with pytest.raises(PermissionError, match="read-only"):
            server.retire("x", project="p")
    finally:
        auth_context_var.reset(reset)
    assert not store.list_collections()


def test_stdio_caller_can_write(store, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "stdio")
    assert server._caller(write=True).write is True
    assert server.learn(NOTE, project="p")


# ── the local owner's namespace ───────────────────────────────────────────


def test_local_owner_defaults_to_legacy_and_can_be_pointed_elsewhere(store, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "stdio")
    monkeypatch.delenv("MEMORY_RAG_LOCAL_NAMESPACE", raising=False)
    assert server._caller(write=True) == server.Caller("legacy", True, True)
    monkeypatch.setenv("MEMORY_RAG_LOCAL_NAMESPACE", "vps-host")
    assert server._caller(write=True) == server.Caller("vps-host", True, True)


@pytest.mark.parametrize("value", ["Not Valid", "a.b", "../x", "x" * 41])
def test_local_owner_with_a_bad_namespace_is_refused(store, monkeypatch, value):
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "stdio")
    monkeypatch.setenv("MEMORY_RAG_LOCAL_NAMESPACE", value)
    with pytest.raises(PermissionError, match="not a valid namespace"):
        server.learn(NOTE, project="p")
    assert not store.list_collections()


def test_local_ingest_lands_in_the_chosen_namespace(http, store, tokens_file, wiki, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "stdio")
    monkeypatch.setenv("MEMORY_RAG_LOCAL_NAMESPACE", "vps-host")
    assert "nodes" in server.ingest_wiki(str(wiki), "p")
    names = {c.name for c in store.list_collections()}
    assert "rag_vps-host.p__wiki" in names and "rag_p__wiki" not in names
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "streamable-http")
    err, out = call(http, TOKENS["vps-host"], "recall", query="set up", project="p", layer="wiki")
    assert not err and "how it is set up" in out
    err, out = call(http, TOKENS["legacy"], "recall", query="set up", project="p", layer="wiki")
    assert not err and "how it is set up" not in out


def _reindex_script():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "reindex_wiki.py"
    spec = importlib.util.spec_from_file_location("reindex_wiki", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reindex_wiki_namespace_flag(http, store, tokens_file, wiki, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "stdio")
    monkeypatch.delenv("MEMORY_RAG_LOCAL_NAMESPACE", raising=False)
    monkeypatch.setenv("MEMORY_RAG_LOCAL_NAMESPACE", "")  # restored after the test
    rc = _reindex_script().main(["--wiki-root", str(wiki.parent), "--project", wiki.name,
                                 "--namespace", "vps-host"])
    assert rc == 0
    names = {c.name for c in store.list_collections()}
    assert f"rag_vps-host.{wiki.name}__wiki" in names and f"rag_{wiki.name}__wiki" not in names
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "streamable-http")
    err, out = call(http, TOKENS["vps-host"], "recall", query="set up", project=wiki.name)
    assert not err and "how it is set up" in out


def test_reindex_wiki_rejects_a_bad_namespace(store, wiki, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "stdio")
    monkeypatch.setenv("MEMORY_RAG_LOCAL_NAMESPACE", "")
    rc = _reindex_script().main(["--wiki-root", str(wiki.parent), "--project", wiki.name,
                                 "--namespace", "Bad NS"])
    assert rc == 2 and not store.list_collections()
