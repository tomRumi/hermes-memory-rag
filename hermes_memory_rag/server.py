"""hermes-memory-rag — layered project memory as an MCP server.

Three collections per project in one VM-local Chroma store (`~/hermes-rag/`):

- `rag_<slug>__code`   raw code chunks (detail layer)
- `rag_<slug>__wiki`   code-wiki markdown, heading-aware (map layer)
- `rag_<slug>__memory` episodic session learnings (staging area, capped)

Tools exposed to every Hermes session (any profile, any project):

- `recall(query, project?, layer?, top?)` — bounded layered retrieval.
- `learn(text, kind, project?)` — deposit ONE learning (never logging).
- `stats(project?)` — layer counts + wiki-earn suggestion signals.

Session-end procedure (encoded in the `project-memory` skill): deposit
learnings via `learn`; check `stats` for the wiki-earn suggestion.

Design invariants:
- embed-before-store; nothing reaches Chroma without a vector
- deterministic IDs → idempotent upserts
- loud failures: a partial or empty index never looks like success
- retrieval is bounded (char caps) and honest ("not built" vs "no results")
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import chromadb
from mcp.server.fastmcp import FastMCP

STORE_DIR = Path(
    os.environ.get("HERMES_RAG_STORE", str(Path.home() / "hermes-rag"))
)
EMBED_MODEL = os.environ.get("HERMES_RAG_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.environ.get("HERMES_RAG_OLLAMA_URL", "http://localhost:11434")
MEMORY_CAP = int(os.environ.get("HERMES_RAG_MEMORY_CAP", "50"))
CONSOLIDATE_THRESHOLD = int(os.environ.get("HERMES_RAG_CONSOLIDATE_THRESHOLD", "10"))
MAX_CHARS_PER_HIT = int(os.environ.get("HERMES_RAG_MAX_CHARS", "600"))
# nomic-embed-text: 2048-token hard limit; ~2.5 chars/token for dense JSON.
# 4500 chars ≈ 1800 tokens — safe margin under the context ceiling.
CHUNK_CHAR_CAP = int(os.environ.get("HERMES_RAG_CHUNK_CHAR_CAP", "4500"))

# Machine-generated files: worthless for retrieval, context-busters.
_MACHINE_SUFFIXES = (".lock", ".min.js", ".min.css", ".map", ".svg")
_MACHINE_BASENAMES = ("package-lock.json", "yarn.lock", "poetry.lock",
                      "pdm.lock", "uv.lock", "composer.lock")
_MACHINE_DIRS = ("chroma", "chroma-agent-workflows", ".ollama")
# Dense-JSON directories to skip entirely, comma-separated. Generated JSON
# indexes and caches are worthless for retrieval and bust the embedding context
# (~2.5 chars/token means even a 4500-char chunk can exceed 2048 tokens).
_MACHINE_JSON_DIRS = tuple(
    d.strip() for d in os.environ.get("HERMES_RAG_SKIP_JSON_DIRS", "").split(",") if d.strip()
)


def _machine_file_skip():
    """Return a predicate: True if a rel path is machine-generated output."""
    def _skip(rel: str) -> bool:
        p = Path(rel)
        if p.suffix in _MACHINE_SUFFIXES:
            return True
        if p.name in _MACHINE_BASENAMES:
            return True
        if any(part in _MACHINE_DIRS for part in p.parts):
            return True
        # Generated JSON under a configured dense-JSON directory: ~2.5
        # chars/token busts the context even at the 4500-char cap.
        if p.suffix == ".json" and any(d in rel for d in _MACHINE_JSON_DIRS):
            return True
        # cache/db JSON anywhere under db/, data/, cache/ style dirs
        lowered = rel.lower()
        for marker in ("cache", "db/", "data/"):
            if marker in lowered and p.suffix == ".json":
                return True
        # state/binary db files anywhere
        if p.suffix in (".db", ".sqlite", ".sqlite3"):
            return True
        return False
    return _skip


def _cap_chunks(texts: list[str], max_chars: int = CHUNK_CHAR_CAP) -> list[str]:
    """Split any text longer than max_chars into ≤max_chars pieces.

    Prefers whitespace boundaries (never cuts mid-word unless a single
    'word' itself exceeds the cap). Content is preserved in order.
    """
    out: list[str] = []
    for text in texts:
        while len(text) > max_chars:
            cut = text.rfind(" ", max_chars - 200, max_chars)
            if cut <= 0:
                cut = max_chars  # unbreakable blob: hard cut
            out.append(text[:cut])
            text = text[cut:].lstrip(" ")
        if text:
            out.append(text)
    return out

mcp = FastMCP("hermes-memory-rag")

# ── shared plumbing ───────────────────────────────────────────────────────


def _slugify(project: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", project.strip().lower()).strip("-")
    return slug or "default"


def _collection_names(project: str) -> dict[str, str]:
    slug = _slugify(project)
    return {
        "code": f"rag_{slug}__code",
        "wiki": f"rag_{slug}__wiki",
        "memory": f"rag_{slug}__memory",
    }


def _client() -> chromadb.PersistentClient:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(STORE_DIR))


def _count(client: chromadb.PersistentClient, name: str) -> int:
    try:
        return client.get_or_create_collection(name).count()
    except Exception:
        return 0


def _embed_model():
    from llama_index.embeddings.ollama import OllamaEmbedding
    return OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_URL)


def _embedder_healthcheck(embed_model) -> None:
    """Fail loudly — a silent dead embedder poisons every layer."""
    embed_model.get_text_embedding("healthcheck")


def _det_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\x00".join((prefix, *parts)).encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _render_hits(hits, tag: str, cap: int = MAX_CHARS_PER_HIT) -> list[str]:
    out = []
    for node in hits:
        text = (node.get_content() or "").strip().replace("\n", " ")
        if len(text) > cap:
            text = text[:cap] + "…"
        meta = node.metadata or {}
        src = meta.get("path", "?")
        section = meta.get("section", "")
        label = f"{src}::{section}" if section else src
        out.append(f"[{tag} {label}] {text}")
    return out


def _search_collection(client: chromadb.PersistentClient, name: str,
                       query: str, top_k: int, embed_model):
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    if getattr(Settings, "_embed_model", None) is None:
        Settings.embed_model = embed_model
    col = client.get_or_create_collection(name)
    index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=col))
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)


# ── usage ledger (sqlite, same dir as the store) ─────────────────────────


def _ledger() -> sqlite3.Connection:
    conn = sqlite3.connect(str(STORE_DIR / "usage.db"))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS events (
               ts REAL, project TEXT, event TEXT, detail TEXT)"""
    )
    conn.commit()
    return conn


def _log_event(project: str, event: str, detail: str = "") -> None:
    conn = _ledger()
    with conn:
        conn.execute(
            "INSERT INTO events (ts, project, event, detail) VALUES (?,?,?,?)",
            (time.time(), _slugify(project), event, detail[:500]),
        )
    conn.close()


# ── tools ─────────────────────────────────────────────────────────────────


@mcp.tool()
def recall(query: str, project: str = "", layer: str = "auto",
           top_wiki: int = 2, top_code: int = 3, top_memory: int = 1) -> str:
    """Layered context recall for a project: wiki (map) → code (detail) →
    memory (prior session learnings). Bounded output; honest per-layer status.
    `project` defaults to the cwd basename; `layer` = auto|wiki|code|memory."""
    project = project or os.path.basename(os.getcwd())
    names = _collection_names(project)
    client = _client()
    embed = _embed_model()

    lines = [f"Recall [{project}] for: {query}"]
    _log_event(project, "recall", query[:200])

    want = layer if layer != "auto" else None

    def _wanted(which: str) -> bool:
        return want == which or (want is None)

    for which, top in (("wiki", top_wiki), ("code", top_code), ("memory", top_memory)):
        if not _wanted(which):
            continue
        name = names[which]
        if _count(client, name) == 0:
            if which != "memory":  # empty staging area is normal
                lines.append(f"[{which}: not built for {project}]")
            continue
        try:
            hits = _search_collection(client, name, query, top, embed)
            rendered = _render_hits(hits, which)
            lines += rendered or [f"[{which}: no results]"]
        except Exception as exc:  # noqa: BLE001 - retrieval never crashes a session
            lines.append(f"[{which} unavailable: {exc}]")
    return "\n".join(lines)


@mcp.tool()
def learn(text: str, kind: str = "learning", project: str = "") -> str:
    """Deposit ONE learning into the project's episodic memory (staging area).
    kind may carry a module prefix for consolidation routing, e.g.
    'rag:gotcha' or 'config:decision'. Learnings only — routine actions are
    logging, not learning, and belong in session transcripts."""
    project = project or os.path.basename(os.getcwd())
    text = text.strip()
    if not text:
        raise ValueError("refusing to store an empty learning")
    if len(text) > 500:
        text = text[:497] + "…"

    names = _collection_names(project)
    client = _client()
    col = client.get_or_create_collection(names["memory"])
    node_id = _det_id("mem", text)
    if col.get(ids=[node_id])["ids"]:
        _log_event(project, "learn_dup")
        return node_id  # idempotent

    embed = _embed_model()
    _embedder_healthcheck(embed)
    vector = embed.get_text_embedding(text)
    col.upsert(
        ids=[node_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{
            "path": "memory",
            "layer": "memory",
            "kind": kind,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }],
    )
    # Backstop cap: prune oldest beyond the staging cap.
    if col.count() > MEMORY_CAP:
        all_meta = col.get(include=["metadatas"])
        pairs = sorted(
            zip(all_meta["ids"], all_meta["metadatas"]),
            key=lambda p: str(p[1].get("created", "")),
        )
        overflow = col.count() - MEMORY_CAP
        col.delete(ids=[pid for pid, _ in pairs[:overflow]])
    _log_event(project, "learn", kind)

    result = node_id
    if col.count() >= CONSOLIDATE_THRESHOLD:
        result += (
            f" || CONSOLIDATION DUE: staging at {col.count()} >= "
            f"{CONSOLIDATE_THRESHOLD} — run scripts/consolidate.py {project} "
            f"--apply (dry-run first)"
        )
    return result


@mcp.tool()
def stats(project: str = "") -> str:
    """Layer counts + wiki-earn suggestion for a project. The suggestion rule:
    (≥3 sessions AND ≥5 learnings) OR (≥3 architecture-phrased recalls) on a
    project with no wiki → suggest generating the wiki."""
    project = project or os.path.basename(os.getcwd())
    names = _collection_names(project)
    client = _client()
    conn = _ledger()
    slug = _slugify(project)

    sessions = conn.execute(
        "SELECT COUNT(DISTINCT CAST(ts/86400 AS INT)) FROM events "
        "WHERE project=? AND event IN ('learn','recall')", (slug,)
    ).fetchone()[0]
    learnings = conn.execute(
        "SELECT COUNT(*) FROM events WHERE project=? AND event='learn'", (slug,)
    ).fetchone()[0]
    arch_recalls = conn.execute(
        "SELECT COUNT(*) FROM events WHERE project=? AND event='recall' "
        "AND (detail LIKE '%where %' OR detail LIKE '%how does%' "
        "OR detail LIKE '%what handles%')", (slug,)
    ).fetchone()[0]
    conn.close()

    counts = {w: _count(client, names[w]) for w in names}
    has_wiki = counts["wiki"] > 0
    earned = (not has_wiki) and ((sessions >= 3 and learnings >= 5) or arch_recalls >= 3)

    report = {
        "project": project,
        "layers": counts,
        "distinct_active_days": sessions,
        "learnings_deposited": learnings,
        "architecture_phased_recalls": arch_recalls,
        "wiki_suggestion": (
            "GENERATE — this repo has earned a wiki" if earned
            else ("wiki present" if has_wiki else "not yet earned")
        ),
        "consolidation_due": counts["memory"] >= CONSOLIDATE_THRESHOLD,
    }
    return json.dumps(report, indent=2)


@mcp.tool()
def ingest_code(root: str, project: str = "", rebuild: bool = True) -> str:
    """Index a repo's source files into the project's code layer.
    Skips vendored/binary/hidden dirs; deterministic IDs make unchanged
    chunks idempotent; rebuild=True drops stale chunks of edited files."""
    project = project or os.path.basename(root.rstrip("/"))
    names = _collection_names(project)
    client = _client()
    embed = _embed_model()
    _embedder_healthcheck(embed)

    root_path = Path(root).expanduser().resolve()
    skip_parts = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules",
        "dist", "build", "target", ".idea", ".vscode", "logs", ".hermes",
    }
    skip_suffixes = {
        ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".class", ".jar",
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar",
        ".gz", ".whl", ".db", ".db-wal", ".db-shm", ".lock", ".log", ".min.js",
    }
    files: list[Path] = []
    for p in root_path.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root_path))
        rel_parts = p.relative_to(root_path).parts
        if any(part in skip_parts for part in rel_parts):
            continue
        if p.suffix.lower() in skip_suffixes:
            continue
        if _machine_file_skip()(rel):
            continue
        files.append(p)
    files.sort()

    if rebuild:
        try:
            client.delete_collection(names["code"])
        except Exception:
            pass

    from llama_index.core import Document
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.schema import TransformComponent
    from llama_index.vector_stores.chroma import ChromaVectorStore

    docs = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        docs.append(Document(text=text[:200_000],
                             metadata={"path": str(p.relative_to(root_path)),
                                       "layer": "code"}))
    if not docs:
        return f"no indexable files under {root_path}"

    parser = SimpleNodeParser.from_defaults(chunk_size=1500, chunk_overlap=150)
    nodes = parser.get_nodes_from_documents(docs)
    # nomic-embed-text hard limit is 2048 tokens (~2.5 chars/token for dense
    # JSON): a 5,316-char JSON chunk busts the context. SentenceSplitter's
    # 1500-token budget ≈ 7,500 chars — NOT a safe cap. Enforce a char cap
    # after parsing; oversized nodes are split further (prefer whitespace
    # boundaries).
    capped_texts: list[list[str]] = []
    for node in nodes:
        capped_texts.append(
            _cap_chunks([node.get_content()], max_chars=CHUNK_CHAR_CAP))
    flat: list[tuple[Any, str]] = []
    for node, pieces in zip(nodes, capped_texts):
        for piece in pieces:
            flat.append((node, piece))
    col = client.get_or_create_collection(names["code"])

    class _Embed(TransformComponent):
        def __call__(self, nodes_in, **kwargs):
            vectors = embed.get_text_embedding_batch(
                [n.get_content() for n in nodes_in])
            for node, vec in zip(nodes_in, vectors):
                node.embedding = vec
            return nodes_in

    pipeline = IngestionPipeline(
        transformations=[_Embed()],
        vector_store=ChromaVectorStore(chroma_collection=col),
    )
    final_nodes = []
    # rebuild node list from flat pieces (each piece = one node copy).
    # Assign the deterministic id directly per TextNode copy — never via the
    # parser node object, which is shared by ALL pieces of one chunk and
    # would give every piece the same (last-written) id.
    from llama_index.core.schema import TextNode
    for i, (node, piece) in enumerate(flat):
        tn = TextNode(text=piece, metadata=dict(node.metadata))
        tn.id_ = _det_id("code", str(node.metadata["path"]),
                         str(i), piece)
        final_nodes.append(tn)
    pipeline.run(nodes=final_nodes)
    stored = col.count()
    if stored < len(final_nodes):
        raise RuntimeError(
            f"ingest incomplete: {stored} stored, {len(final_nodes)} expected")
    _log_event(project, "ingest_code", f"{stored} nodes")
    return f"code layer [{project}]: {stored} nodes from {len(docs)} files"


@mcp.tool()
def ingest_wiki(wiki_dir: str, project: str = "") -> str:
    """(Re)index a code-wiki directory (markdown) into the project's wiki
    layer. Heading-aware chunking keeps `##` sections atomic."""
    project = project or os.path.basename(wiki_dir.rstrip("/"))
    names = _collection_names(project)
    client = _client()
    embed = _embed_model()
    _embedder_healthcheck(embed)

    root = Path(wiki_dir).expanduser().resolve()
    files = sorted(p for p in root.rglob("*.md") if p.is_file())
    if not files:
        return f"no markdown under {root}"

    try:
        client.delete_collection(names["wiki"])
    except Exception:
        pass

    from llama_index.core import Document
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.schema import TransformComponent
    from llama_index.vector_stores.chroma import ChromaVectorStore

    def _heading_chunks(text: str) -> list[str]:
        parts = text.split("\n## ")
        out = []
        for i, part in enumerate(parts):
            if i > 0:
                head, _, rest = part.partition("\n")
                body = rest.strip()
                title = head.strip()
            else:
                body, title = part.strip(), ""
            if not body:
                continue
            chunks = [f"## {title}\n{body}"] if title else [body]
            for chunk in chunks:
                while len(chunk) > 4000:
                    out.append(chunk[:4000])
                    chunk = "## " + title + "\n" + chunk[4000:]
                out.append(chunk)
        return out

    docs = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        for idx, chunk in enumerate(_heading_chunks(text)):
            docs.append(Document(
                text=chunk,
                metadata={"path": str(p.relative_to(root)), "layer": "wiki",
                          "section": "", "chunk_index": idx},
            ))

    class _Embed(TransformComponent):
        def __call__(self, nodes_in, **kwargs):
            vectors = embed.get_text_embedding_batch(
                [n.get_content() for n in nodes_in])
            for node, vec in zip(nodes_in, vectors):
                node.embedding = vec
            return nodes_in

    col = client.get_or_create_collection(names["wiki"])
    pipeline = IngestionPipeline(
        transformations=[_Embed()],
        vector_store=ChromaVectorStore(chroma_collection=col),
    )
    pipeline.run(nodes=docs)
    stored = col.count()
    if stored < len(docs):
        raise RuntimeError(f"wiki ingest incomplete: {stored}/{len(docs)}")
    _log_event(project, "ingest_wiki", f"{stored} nodes")
    return f"wiki layer [{project}]: {stored} nodes"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
