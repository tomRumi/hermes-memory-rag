"""Hermes plugin entry point.

What exists today: the memory server (``hermes_memory_rag.server``, run as the MCP
server ``hermes-memory-rag``) and the scripts in ``scripts/`` for merging staged
learnings into the wiki and for exporting, restoring and comparing the store.
Both are usable without this plugin file.

What is not built yet, and will register here as it lands:

- the installer, which sets up the store, pulls the embedding model and writes the
  server into each profile's configuration;
- the part that retrieves relevant memory before each message and puts it in front
  of the model, and that mirrors the agent's own memory writes into the store so
  that removing an entry stops being destructive;
- the desktop window for status, export, import and the backup toggle;
- the scheduled jobs (backup, reviewing staged learnings, importing Claude Code's
  memory files).

Importing this module must stay cheap: nothing heavy (no chromadb, no llama-index)
should be imported at plugin-load time.
"""

from __future__ import annotations

from typing import Any


def register(ctx: Any) -> None:
    """Register plugin contributions.

    Deliberately empty for now — see the module docstring. An honest empty
    registration is better than declaring capabilities that do not exist.
    """
    return None