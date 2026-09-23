"""TDD: chunk char cap must prevent nomic context overflow (2048-token limit).

Measured facts about nomic-embed-text:
- nomic-embed-text accepts at most 2048 tokens; ~2.5 chars/token for dense
  JSON means a 5,316-char JSON chunk exceeds the context → HTTP 500.
- LlamaIndex's SentenceSplitter counts tokens with a GPT-style tokenizer:
  its 1500-token cap ≈ 7,500 chars — a cap that does NOT protect nomic.
- The new /api/embed endpoint silently TRUNCATES over-long inputs (cos(t1,t2)
  = 1.0 for texts differing only after the limit) — worse than failing.

Fix contract for `ingest_code` (server) and `index_workspace` (v1 indexer):
1. A character cap per chunk (default 4500 chars ≈ 1800 nomic tokens),
   applied AFTER node parsing: oversized nodes are split further.
2. Machine-generated files are excluded from ingestion (cache JSON, lock
   files, minified assets) — they pollute retrieval and bust context.
"""
from __future__ import annotations

import hashlib

import pytest

from hermes_memory_rag import server


class TestCharCap:
    def test_split_oversized_node_respects_cap(self):
        # 10,000 chars of dense JSON → one SentenceSplitter chunk (fits llama's
        # token budget) but must be split by the char cap into ≤4500-char pieces
        text = '{"key": "value-with-some-length"}' * 300  # 9,900 chars, one line
        pieces = server._cap_chunks([text], max_chars=4500)
        assert all(len(p) <= 4500 for p in pieces)
        assert sum(len(p) for p in pieces) >= len(text) * 0.99  # no content lost

    def test_short_chunk_untouched(self):
        assert server._cap_chunks(["short"], max_chars=4500) == ["short"]

    def test_cap_splits_on_reasonable_boundaries(self):
        # splitting shouldn't cut a word if avoidable: piece ends at whitespace
        text = "word " * 2000  # 10,000 chars
        pieces = server._cap_chunks([text], max_chars=4500)
        assert all(len(p) <= 4500 for p in pieces)
        assert all(p.endswith((" ", "\n")) or len(p) < 4500 for p in pieces)


class TestMachineFileExclusion:
    def test_cache_json_excluded(self):
        skip = server._machine_file_skip()
        assert skip("db/providers_models_cache.json") is True
        assert skip("db/app_state.db") is True

    def test_normal_code_not_excluded(self):
        skip = server._machine_file_skip()
        assert skip("src/app/config.py") is False
        assert skip("config/litellm_config.yaml") is False
        assert skip("README.md") is False


class TestIngestEndToEndWithCap:
    def test_ingest_repo_with_monster_json_succeeds(self, tmp_path, monkeypatch):
        # A repo containing a 300KB single-line JSON must ingest without
        # hitting the embedder's context limit.
        repo = tmp_path / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "app.py").write_text("def main():\n    pass\n" * 100)
        (repo / "db").mkdir()
        (repo / "db" / "cache.json").write_text('{"a": "' + "x" * 400000 + '"}')

        monkeypatch.setattr(server, "_client", lambda: _fresh_client(tmp_path))
        # fake the embedder to avoid Ollama in tests, but make it REJECT
        # inputs over the real limit so the cap is actually exercised
        monkeypatch.setattr(server, "_embed_model", _StrictFakeEmbed)

        result = server.ingest_code(str(repo), "captest", rebuild=True)
        assert "nodes" in result

    def test_strict_embedder_rejects_over_limit(self):
        emb = _StrictFakeEmbed()
        with pytest.raises(Exception):
            emb.get_text_embedding("word " * 3000)


class _StrictFakeEmbed:
    """Mimics nomic's 2048-token limit: rejects >5200-char dense-ish input."""

    limit_chars = 5200

    def get_text_embedding(self, text: str) -> list[float]:
        if len(text) > self.limit_chars:
            raise RuntimeError(
                "the input length exceeds the context length")
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[:8]]

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.get_text_embedding(t) for t in texts]


def _fresh_client(tmp_path):
    import chromadb
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))
