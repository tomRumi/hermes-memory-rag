"""Helpers shared by the tests of the plugin-support features (unverified notes,
folder policy, token directory, reindex_wiki).

Nothing here touches a real store, real tokens or a real port: callers get temp
directories, a bag-of-words embedder that needs no Ollama, and a way to run a tool
as a given bearer token without going through HTTP.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import Field

from hermes_memory_rag import server

DIM = 64


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def write_json(path: Path, obj, mode: int = 0o600) -> None:
    """Write `obj` as JSON the way an operator would: then set the mode."""
    path.write_text(json.dumps(obj))
    path.chmod(mode)


def write_token_file(directory: Path, namespace: str, tokens: dict, folders=None,
                     *, atomic: bool = True, **extra) -> Path:
    """Write `<directory>/<namespace>.json` (format v1) by the documented protocol:
    a dotfile temp, then rename. `tokens` maps plaintext token -> its flags."""
    body = {"version": 1, "namespace": namespace,
            "tokens": {digest(t): flags for t, flags in tokens.items()}}
    if folders is not None:
        body["allowed_folders"] = [str(f) for f in folders]
    body.update(extra)
    target = directory / f"{namespace}.json"
    if atomic:
        tmp = directory / f".{namespace}.json.tmp"
        write_json(tmp, body)
        os.replace(tmp, target)
    else:
        write_json(target, body)
    return target


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class CountingEmbedding(BaseEmbedding):
    """A deterministic bag-of-words embedder that records what it was asked to embed.

    `calls` lists every text embedded for storage (not queries). If `fail_when` is
    set, embedding a text that contains it raises. If `gate` is set, embedding waits
    on the event first, which lets a test hold a re-index mid-run.
    """

    embed_dim: int = DIM
    calls: list = Field(default_factory=list)
    fail_when: str = ""
    gate: object = None

    @classmethod
    def class_name(cls) -> str:
        return "CountingEmbedding"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in words(text):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        if isinstance(self.gate, threading.Event):
            self.gate.wait(timeout=30)
        if self.fail_when and self.fail_when in text:
            raise RuntimeError("embedder refused this text")
        self.calls.append(text)
        return self._vector(text)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


@contextlib.contextmanager
def as_token(token: str):
    """Run the body as the caller `token` is: the auth context the HTTP layer would
    set, without HTTP. The token must be known to the server at the time."""
    from mcp.server.auth.middleware.auth_context import auth_context_var
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from mcp.server.auth.provider import AccessToken

    d = digest(token)
    caller = server._verifier._current().get(d)
    client_id = f"{caller.namespace if caller else 'unknown'}:{d[:12]}"
    reset = auth_context_var.set(
        AuthenticatedUser(AccessToken(token=d, client_id=client_id, scopes=[])))
    try:
        yield
    finally:
        auth_context_var.reset(reset)


def collection_ids(client, name: str) -> set[str]:
    try:
        return set(client.get_collection(name).get(include=[])["ids"])
    except Exception:  # noqa: BLE001 - no such collection
        return set()


def collection_docs(client, name: str) -> list[str]:
    try:
        return list(client.get_collection(name).get(include=["documents"])["documents"])
    except Exception:  # noqa: BLE001
        return []


def snapshot(client) -> dict:
    """Every collection's rows and the ledger: everything a tool could change."""
    import sqlite3
    cols = {}
    for c in client.list_collections():
        got = client.get_collection(c.name).get(include=["documents", "metadatas"])
        cols[c.name] = sorted(zip(got["ids"], got["documents"],
                                  [json.dumps(m, sort_keys=True) for m in got["metadatas"]]))
    ledger = server.STORE_DIR / "usage.db"
    rows = (sqlite3.connect(str(ledger)).execute("SELECT * FROM events").fetchall()
            if ledger.exists() else [])
    return {"collections": cols, "ledger": rows}


# ── talking to the real HTTP app ──────────────────────────────────────────

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def rpc(http, token, method, params=None, session=None, id_=1):
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
    if response.headers["content-type"].startswith("text/event-stream"):
        data = [ln[6:] for ln in response.text.splitlines() if ln.startswith("data: ")]
        return json.loads(data[-1])
    return response.json()


def open_session(http, token):
    response = rpc(http, token, "initialize", {
        "protocolVersion": "2025-03-26", "capabilities": {},
        "clientInfo": {"name": "t", "version": "0"}})
    assert response.status_code == 200, response.text
    session = response.headers["mcp-session-id"]
    rpc(http, token, "notifications/initialized", session=session, id_=None)
    return session


def call_in_session(http, token, session, tool, **arguments):
    response = rpc(http, token, "tools/call", {"name": tool, "arguments": arguments},
                   session=session, id_=2)
    assert response.status_code == 200, response.text
    result = _payload(response)["result"]
    return result["isError"], "".join(c["text"] for c in result["content"])


def call(http, token, tool, **arguments):
    """Call a tool over HTTP as `token`: (is_error, text)."""
    return call_in_session(http, token, open_session(http, token), tool, **arguments)
