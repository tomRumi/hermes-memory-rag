"""Fixtures for the plugin-support tests (see support.py). The older test modules
bring their own fixtures and do not use these."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hermes_memory_rag import server
from support import CountingEmbedding

URL = "http://127.0.0.1:8765"

# ── no test may reach the real configuration ──────────────────────────────
# The server reads the tokens file, the token directory and namespaces.json (and
# scripts read the same place) from ~/.config/memory-rag unless told otherwise, so a
# test that forgot to point them at a temp dir would be judged by whatever the
# machine's operator has configured. Two layers: an autouse fixture points all three
# at temp paths for every test, and an audit hook fails any test that still opens,
# lists or changes something under the real directory.
REAL_CONFIG_DIR = Path.home() / ".config" / "memory-rag"
GUARDED_EVENTS = {"open", "os.listdir", "os.scandir", "os.chmod", "os.remove", "os.rename",
                  "os.mkdir", "os.rmdir"}
touched_real_config: list[tuple[str, str]] = []
_guard_armed = False


def _audit(event: str, args: tuple) -> None:
    if not _guard_armed or event not in GUARDED_EVENTS or not args:
        return
    target = args[0]
    if isinstance(target, (str, bytes, os.PathLike)):
        path = os.path.abspath(os.fsdecode(target))
        for root in {str(REAL_CONFIG_DIR), os.path.realpath(REAL_CONFIG_DIR)}:
            if path == root or path.startswith(root + os.sep):
                touched_real_config.append((event, path))


sys.addaudithook(_audit)  # cannot be removed; it does nothing unless armed


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point the tokens file, token directory and namespaces file at temp paths, and
    fail the test if it still touched the real configuration directory."""
    global _guard_armed

    nowhere = tmp_path / "isolated-config"  # deliberately not created
    monkeypatch.setattr(server, "TOKENS_FILE", nowhere / "tokens.json")
    monkeypatch.setenv("HERMES_RAG_TOKENS_DIR", str(nowhere / "tokens.d"))
    monkeypatch.setenv("HERMES_RAG_NAMESPACES_FILE", str(nowhere / "namespaces.json"))
    touched_real_config.clear()
    _guard_armed = True
    try:
        yield
        found = list(touched_real_config)
    finally:
        _guard_armed = False
    assert not found, f"the test touched the real configuration directory: {found}"
_ENV = ("HERMES_RAG_TOKENS_DIR", "HERMES_RAG_NAMESPACES_FILE", "HERMES_RAG_RESERVED_NAMESPACES",
        "HERMES_RAG_FOLDER_DENY", "MEMORY_RAG_LOCAL_NAMESPACE")


@pytest.fixture
def rag(tmp_path, monkeypatch):
    """A store, a config directory and an embedder, all in tmp_path.

    The caller is the local owner (stdio) of namespace `acme`; tests that need a
    bearer token write token files and use `support.as_token`. Nothing outside
    tmp_path is read or written.
    """
    import chromadb
    from llama_index.core import Settings

    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "chroma"
    path.mkdir()
    client = chromadb.PersistentClient(path=str(path))
    embed = CountingEmbedding()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    cfg.chmod(0o700)
    monkeypatch.setattr(server, "STORE_DIR", path)
    monkeypatch.setattr(server, "_client", lambda: client)
    monkeypatch.setattr(server, "_embed_model", lambda: embed)
    monkeypatch.setattr(server, "TOKENS_FILE", cfg / "tokens.json")
    monkeypatch.setattr(server, "PROJECTS_FILE", tmp_path / "no-projects.yaml")
    monkeypatch.setenv("HERMES_RAG_TRANSPORT", "stdio")
    monkeypatch.setenv("MEMORY_RAG_LOCAL_NAMESPACE", "acme")
    Settings.embed_model = embed
    return types.SimpleNamespace(
        client=client, embed=embed, tmp=tmp_path, cfg=cfg, store=path,
        tokens_json=cfg / "tokens.json", tokens_dir=cfg / "tokens.d",
        namespaces=cfg / "namespaces.json")


@pytest.fixture
def make_tokens_dir(rag):
    rag.tokens_dir.mkdir(mode=0o700)
    rag.tokens_dir.chmod(0o700)
    return rag.tokens_dir


@pytest.fixture(scope="module")
def http_client():
    """The real streamable-HTTP app. A FastMCP object runs one session manager only
    once, so each module that needs HTTP gets a fresh one and hands it back."""
    server.mcp._session_manager = None
    with TestClient(server.mcp.streamable_http_app(), base_url=URL) as client:
        yield client
    server.mcp._session_manager = None
