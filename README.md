# hermes-memory-rag

Memory for [Hermes Agent](https://hermes-agent.nousresearch.com/docs) that does not have to fit in a
context window.

## The problem this solves

Hermes keeps a small memory file that is placed in front of the model on every message, with a hard
size limit. Once the limit is reached, adding a fact means deleting another one, and the deleted
text is gone. That is fine for a handful of standing rules. It is not fine for the accumulated
knowledge of a project.

This project keeps the knowledge elsewhere and hands the model only what is relevant to the message
in front of it. Nothing has to be deleted to make room.

## What it does

- Keeps facts in a local database (Chroma) on your machine. Nothing is sent anywhere.
- Splits them into three sets per project, described below.
- Answers three kinds of question cheaply: *where is this in the code*, *how does this work*, and
  *what did we already learn about this*.
- Tells a model when its staged learnings should be merged into the project's written notes, and
  merges them when asked.
- Exports everything to plain text files so it can be read, kept in git, and rebuilt on another
  machine.

## The three layers

Each project gets three collections, named `rag_<project>__<layer>`:

| layer | holds | written by |
|---|---|---|
| `code` | the project's source files, split into pieces, so "where is X" does not mean reading files | `ingest_code` |
| `wiki` | markdown pages describing how the project works — the long, written knowledge | `ingest_wiki` |
| `memory` | short learnings from sessions: a root cause, a decision and why, a trap that cost time | `learn` |

The `memory` layer is a staging area, not an archive. When it reaches a threshold, its notes are
merged into the wiki pages and the notes are removed. Written knowledge therefore grows in the wiki,
which is a directory of markdown files you can read and keep in git.

## The tools it gives the agent

Served over MCP as a server named `hermes-memory-rag`:

```python
recall(query, project="", layer="auto", top_wiki=2, top_code=3, top_memory=1) -> str
learn(text, kind="learning", project="") -> str
stats(project="") -> str
ingest_code(root, project="", rebuild=True) -> str
ingest_wiki(wiki_dir, project="") -> str
```

`recall` searches wiki, then code, then memory, and returns a bounded amount of text — two wiki
sections, three code pieces and one learning by default, 600 characters each. If it returned more,
every message would cost more.

`learn` is idempotent: storing the same text twice changes nothing. When staging reaches ten notes
it reports that a merge is due.

## Where the data lives

`~/hermes-rag` by default. Override with `HERMES_RAG_STORE`.

Keep it on a local disk. Chroma's SQLite write lock breaks across a network or shared-virtual-machine
filesystem in both directions, so a store on a mounted share will lose writes.

## Settings

All optional; the defaults are the ones above.

| setting | default | what it changes |
|---|---|---|
| `HERMES_RAG_STORE` | `~/hermes-rag` | where the store lives |
| `HERMES_RAG_EMBED_MODEL` | `nomic-embed-text` | which ollama model embeds text |
| `HERMES_RAG_OLLAMA_URL` | `http://localhost:11434` | where ollama is |
| `HERMES_RAG_WIKI_ROOT` | `~/.hermes/wikis` | where the markdown pages live |
| `HERMES_RAG_MEMORY_CAP` | `50` | notes allowed in the staging layer before the oldest are dropped |
| `HERMES_RAG_CONSOLIDATE_THRESHOLD` | `10` | when a merge is reported as due |
| `HERMES_RAG_MAX_CHARS` | `600` | characters allowed per returned hit |
| `HERMES_RAG_CHUNK_CHAR_CAP` | `4500` | largest piece of text sent to the embedder |
| `HERMES_RAG_SKIP_JSON_DIRS` | empty | comma-separated directories whose generated JSON should never be indexed |

If your project keeps generated JSON indexes or caches in a directory of their own, add it to
`HERMES_RAG_SKIP_JSON_DIRS`. They are worthless for searching and, because JSON is dense, a
4500-character piece of it can exceed the embedding model's limit.

## Requirements

- Python 3.11 or newer
- [ollama](https://ollama.com) with `nomic-embed-text` pulled (274 MB) — everything is embedded and
  searched locally
- optionally `granite4:3b` (2.1 GB) for the merge step; without it the merge falls back to appending
  the notes under a heading

Two limits worth knowing, both from `nomic-embed-text`: it has a hard 2048-token ceiling, and ollama
truncates longer input **silently**. This is why ingest splits text at 4500 characters and the wiki
pages are split at headings.

## Backup and restore

```
python scripts/export_store.py --out backup.jsonl
python scripts/restore_store.py --in backup.jsonl --store ~/hermes-rag --force
python scripts/compare_stores.py --a ~/hermes-rag --b /tmp/rebuilt --collection rag_myproject__wiki --query "..."
```

`export_store.py` writes two files:

- `backup.jsonl` — every fact's text and its labels. Readable, diffable, worth keeping in git.
- `backup.embedding-numbers.jsonl` — for every fact, the list of numbers the search compares.

The second file exists because of something measurable: different parts of the store keep their
number lists differently — some as the model produced them (list size around 20), some resized to 1.
Distance between two lists is measured in a way that notices the difference, so **recomputing the
numbers from the text does not reproduce the original search order for the resized parts**. The first
result usually survives; the second and third can swap. Carrying the numbers makes a restore exact,
and makes it seconds instead of minutes.

`restore_store.py` refuses to run when the export was made with a different embedding model, rather
than quietly producing a store that answers differently.

## What it stores, and privacy

Everything stays on your machine. Nothing is uploaded. But be clear about what the store contains:
the learnings you deposit, your wiki pages, and **excerpts of your project's source files**. Treat
`~/hermes-rag` like a copy of your notes and parts of your code — back it up if it matters, and keep
it out of any repository.

## Status

Built and testable: the server, and the scripts for merging, exporting, restoring and comparing.

Not built yet: the installer, the part that retrieves memory before each message, the desktop window,
and the scheduled jobs. The layout above is the shape the rest will fit into.

## Licence

MIT.