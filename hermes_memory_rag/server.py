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

# ── node status ───────────────────────────────────────────────────────────
# Every node carries a status, and recall returns only live ones. A fact that has
# been replaced, withdrawn or moved aside therefore stops being handed to the
# model while staying in the store, so a wrong judgement is reversible and the
# history is auditable. Nodes written before this existed carry no status and are
# treated as live, so an un-migrated store keeps working.
STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"  # replaced by another node, named in superseded_by
STATUS_RETIRED = "retired"        # withdrawn; nothing replaced it
STATUS_ARCHIVED = "archived"      # moved out of the way (staging overflow)
NON_LIVE_STATUSES = (STATUS_SUPERSEDED, STATUS_RETIRED, STATUS_ARCHIVED)

# The cross-project memory layer. Reserved name: a project may not be called this.
GLOBAL_PROJECT = "global"

# Notes that mirror the agent's own memory file into the store. They are kept so
# that removing an entry from that file stops being destructive, but they are not
# learnings: they must not count toward the merge threshold, and consolidation
# must not route them into the wiki.
#
# Marked by a field, not by the kind string. An earlier version keyed off kinds
# starting with "memory", which silently swallowed legitimate learnings filed as
# "memory:decision" — they were stored, never merged, and nothing said so.
MIRROR_SOURCE = "memory_file"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stamp(metadata: dict, *, source: str) -> dict:
    """Add the fields every node carries: status, source (and created where absent)."""
    metadata.setdefault("status", STATUS_ACTIVE)
    metadata.setdefault("source", source)
    metadata.setdefault("created", _now())
    return metadata


def _is_live(hit_or_meta) -> bool:
    """False for a node that must not be handed to the model.

    Missing status counts as live: an un-migrated store still answers.
    """
    meta = getattr(hit_or_meta, "metadata", None)
    if meta is None:
        meta = hit_or_meta if isinstance(hit_or_meta, dict) else {}
    return (meta.get("status") or STATUS_ACTIVE) not in NON_LIVE_STATUSES


def _resolve_node_id(col, prefix: str) -> str | None:
    """Resolve a node id from an abbreviated form (as recall prints it).

    Exact id wins; otherwise a unique prefix match. Returns None when the prefix
    matches nothing, or when it is ambiguous — an ambiguous reference must never
    silently mark the wrong fact.
    """
    ids = (col.get(include=[]) or {}).get("ids") or []
    if prefix in ids:
        return prefix
    matches = [i for i in ids if i.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


def _staging_count(col) -> int:
    """Notes that count toward the merge threshold.

    A node copied in from the agent's own memory file is excluded: it is a copy of
    something that already lives in the context window, so counting it would report
    a merge as due when no session had actually learned anything. Those nodes mark
    themselves with a source of ``memory_file``.
    """
    got = col.get(include=["metadatas"]) or {}
    total = 0
    for meta in got.get("metadatas") or []:
        meta = meta or {}
        if meta.get("source") == MIRROR_SOURCE:
            continue
        if meta.get("status") in NON_LIVE_STATUSES:
            continue
        total += 1
    return total


# ── which project is this? ────────────────────────────────────────────────
# Falling back to the name of the working directory is wrong often enough to
# matter: several projects can share a directory name, and a session's working
# directory is wherever the shell happens to be, not necessarily the project being
# discussed. A file of path prefixes removes the guess. Without one, the old
# behaviour stands, so nothing breaks for an install that has never configured it.

PROJECTS_FILE = Path(os.environ.get(
    "HERMES_RAG_PROJECTS_FILE", str(Path.home() / ".hermes" / "projects.yaml")))


def _load_project_map(path: Path | None = None) -> list[tuple[str, str]]:
    """(path prefix, project) pairs from the projects file, longest prefix first.

    The file is a plain list of ``- path: /where/it/is`` / ``project: name``
    entries. A missing or unreadable file yields no mapping rather than an error:
    project attribution must never be the reason a session cannot work.
    """
    path = path or PROJECTS_FILE
    try:
        text = path.expanduser().read_text(encoding="utf-8")
    except OSError:
        return []

    pairs: list[tuple[str, str]] = []
    prefix = project = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            prefix = project = None
            line = line[2:].strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip().strip("'\"")
        if key in ("path", "prefix", "dir", "root"):
            prefix = value
        elif key in ("project", "name", "slug"):
            project = value
        if prefix and project:
            pairs.append((str(Path(prefix).expanduser()), project))
            prefix = project = None

    # Longest matching prefix wins, so a nested directory can name its own project.
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return pairs


def _resolve_project(explicit: str = "", cwd: str | None = None) -> str:
    """The project a call belongs to.

    An explicit name always wins. Otherwise the working directory is matched
    against the projects file, and only if nothing matches is the directory name
    used — the old, guessy behaviour.
    """
    if explicit.strip():
        return explicit.strip()
    here = Path(cwd or os.getcwd())
    for prefix, project in _load_project_map():
        try:
            here.relative_to(prefix)
        except ValueError:
            continue
        return project
    return here.name


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


def _render_hits(hits, tag: str, cap: int = MAX_CHARS_PER_HIT, show_id: bool = False) -> list[str]:
    out = []
    for node in hits:
        text = (node.get_content() or "").strip().replace("\n", " ")
        if len(text) > cap:
            text = text[:cap] + "…"
        meta = node.metadata or {}
        src = meta.get("path", "?")
        section = meta.get("section", "")
        label = f"{src}::{section}" if section else src
        if show_id and (node_id := getattr(node, "node_id", "")):
            # A short form of the stored id, so a later call can name this exact
            # fact when replacing it (learn(supersedes=...)) or withdrawing it
            # (retire(...)). Only shown where that matters — the memory layer.
            label = f"{label} #{node_id[:8]}"
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
    `project` defaults to whatever the projects file says the working directory
    belongs to, falling back to its name; `layer` = auto|wiki|code|memory."""
    project = _resolve_project(project)
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
            # Over-fetch, then drop anything superseded, retired or archived: a
            # fact that was replaced must not be handed to the model, and dropping
            # it must not quietly shrink what the caller asked for.
            hits = [h for h in _search_collection(client, name, query, top * 3, embed)
                    if _is_live(h)][:top]
            rendered = _render_hits(hits, which, show_id=(which == "memory"))
            lines += rendered or [f"[{which}: no results]"]
        except Exception as exc:  # noqa: BLE001 - retrieval never crashes a session
            lines.append(f"[{which} unavailable: {exc}]")

    # Cross-project memory, after the project's own layers so a project hit always
    # comes first, and capped so it cannot crowd out project context. A project is
    # never silently read as the global one: this is a separate, labelled pass.
    if _wanted("memory") and _slugify(project) != GLOBAL_PROJECT:
        gname = _collection_names(GLOBAL_PROJECT)["memory"]
        if _count(client, gname):
            try:
                ghits = [h for h in _search_collection(client, gname, query, top_memory * 3, embed)
                         if _is_live(h)][:top_memory]
                lines += [f"[global] {line}" for line in
                          _render_hits(ghits, "memory", show_id=True)]
            except Exception as exc:  # noqa: BLE001
                lines.append(f"[global memory unavailable: {exc}]")
    return "\n".join(lines)


@mcp.tool()
def learn(text: str, kind: str = "learning", project: str = "",
          supersedes: str = "") -> str:
    """Deposit ONE learning into the project's episodic memory (staging area).
    kind may carry a module prefix for consolidation routing, e.g.
    'rag:gotcha' or 'config:decision'. Learnings only — routine actions are
    logging, not learning, and belong in session transcripts.

    `supersedes` names an existing note (the short id recall prints, or a unique
    prefix of it) that this one replaces. The old note is marked as replaced and
    stops being returned, but is kept, so the change is reversible and you can
    still see what the earlier fact was. Use it instead of rewriting history."""
    project = _resolve_project(project)
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
    metadata = _stamp({"path": "memory", "layer": "memory", "kind": kind}, source="learn")
    col.upsert(
        ids=[node_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[metadata],
    )

    note = ""
    if supersedes:
        target = _resolve_node_id(col, supersedes.strip())
        if target is None:
            note = (f" || supersedes={supersedes!r} matched no single note — "
                    f"nothing was marked as replaced")
        else:
            old = (col.get(ids=[target], include=["metadatas"])["metadatas"] or [{}])[0] or {}
            col.update(ids=[target], metadatas=[
                {**old, "status": STATUS_SUPERSEDED, "superseded_by": node_id}])
            _log_event(project, "supersede", f"{target}->{node_id}")
            note = f" || replaced {target[:8]}"

    # Backstop past the cap: the oldest notes are moved aside, never deleted. They
    # keep their text, stop being returned by recall, and remain countable in the
    # store — so the promise that nothing is lost holds even at the limit.
    if _staging_count(col) > MEMORY_CAP:
        all_meta = col.get(include=["metadatas"])
        live = [
            (pid, meta or {})
            for pid, meta in zip(all_meta["ids"], all_meta["metadatas"])
            if (meta or {}).get("status", STATUS_ACTIVE) == STATUS_ACTIVE
            and (meta or {}).get("source") != MIRROR_SOURCE
        ]
        live.sort(key=lambda pair: str(pair[1].get("created", "")))
        overflow = len(live) - MEMORY_CAP
        for pid, meta in live[:max(0, overflow)]:
            col.update(ids=[pid], metadatas=[{**meta, "status": STATUS_ARCHIVED}])
        if overflow > 0:
            _log_event(project, "archive", f"{overflow} of {len(live)} notes archived")
            note += f" || {overflow} oldest note(s) archived (kept, no longer returned)"

    _log_event(project, "learn", kind)

    staged = _staging_count(col)
    result = node_id + note
    if staged >= CONSOLIDATE_THRESHOLD:
        result += (
            f" || CONSOLIDATION DUE: staging at {staged} >= "
            f"{CONSOLIDATE_THRESHOLD} — run scripts/consolidate.py {project} "
            f"--apply (dry-run first)"
        )
    return result


@mcp.tool()
def retire(node_id: str, reason: str = "", project: str = "") -> str:
    """Withdraw a memory note that nothing replaces — a fact found to be wrong, or
    one that no longer applies. The note stays in the store (so the withdrawal is
    reversible and the history readable) but recall stops returning it.

    `node_id` is the short id recall prints, or any unique prefix of it. An
    ambiguous or unknown reference changes nothing and says so."""
    project = _resolve_project(project)
    client = _client()
    col = client.get_or_create_collection(_collection_names(project)["memory"])
    target = _resolve_node_id(col, node_id.strip())
    if target is None:
        return (f"no single memory note matches {node_id!r} — nothing changed. "
                f"Call recall first and use the id it prints.")
    meta = (col.get(ids=[target], include=["metadatas"])["metadatas"] or [{}])[0] or {}
    if meta.get("status") in NON_LIVE_STATUSES:
        return f"{target[:8]} is already {meta.get('status')} — nothing changed"
    col.update(ids=[target], metadatas=[
        {**meta, "status": STATUS_RETIRED, "retired_reason": reason[:200]}])
    _log_event(project, "retire", f"{target} {reason[:100]}")
    return f"retired {target[:8]}" + (f" ({reason[:80]})" if reason else "")


@mcp.tool()
def stats(project: str = "") -> str:
    """Layer counts + wiki-earn suggestion for a project. The suggestion rule:
    (≥3 sessions AND ≥5 learnings) OR (≥3 architecture-phrased recalls) on a
    project with no wiki → suggest generating the wiki."""
    project = _resolve_project(project)
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
    # The cross-project layer is reported on its own: it is not this project's
    # memory, and lumping it in would make the counts lie.
    global_memory = _count(client, _collection_names(GLOBAL_PROJECT)["memory"])
    has_wiki = counts["wiki"] > 0
    earned = (not has_wiki) and ((sessions >= 3 and learnings >= 5) or arch_recalls >= 3)

    # "memory" in layers counts every note ever stored, including replaced and
    # moved-aside ones. The number that drives merging is the live staging count.
    staged = _staging_count(client.get_or_create_collection(names["memory"]))

    report = {
        "project": project,
        "layers": counts,
        "staged_notes": staged,
        "global_memory_notes": global_memory,
        "distinct_active_days": sessions,
        "learnings_deposited": learnings,
        "architecture_phased_recalls": arch_recalls,
        "wiki_suggestion": (
            "GENERATE — this repo has earned a wiki" if earned
            else ("wiki present" if has_wiki else "not yet earned")
        ),
        "consolidation_due": staged >= CONSOLIDATE_THRESHOLD,
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
                             metadata=_stamp(
                                 {"path": str(p.relative_to(root_path)), "layer": "code"},
                                 source="ingest_code")))
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
                metadata=_stamp(
                    {"path": str(p.relative_to(root)), "layer": "wiki",
                     "section": "", "chunk_index": idx}, source="ingest_wiki"),
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
