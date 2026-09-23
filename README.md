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
learn(text, kind="learning", project="", supersedes="") -> str
retire(node_id, reason="", project="") -> str
stats(project="") -> str
ingest_code(root, project="", rebuild=True) -> str
ingest_wiki(wiki_dir, project="") -> str
```

`recall` searches wiki, then code, then memory, and returns a bounded amount of text — two wiki
sections, three code pieces and one learning by default, 600 characters each. If it returned more,
every message would cost more. It also searches a cross-project layer last, labelled `[global]`, so a
fact that applies everywhere is reachable without being filed under a project.

`learn` is idempotent: storing the same text twice changes nothing. When staging reaches ten notes
it reports that a merge is due.

## Nothing is deleted

Facts get out of date. The tempting move is to overwrite or delete, which is how a memory system
loses the very thing it exists to keep. Instead every note carries a status, and recall returns only
live ones:

| status | meaning | how it happens |
|---|---|---|
| `active` | current, returned by recall | every write |
| `superseded` | something replaced it; `superseded_by` names the successor | `learn(..., supersedes=<id>)` |
| `retired` | withdrawn; nothing replaced it | `retire(<id>, reason=...)` |
| `archived` | moved out of the way when staging overflows | automatic, past the cap |

A replaced or withdrawn fact stays in the store with its text intact, so the change is reversible and
you can still read what the earlier belief was. Recall prints a short id for memory hits — that is
what you pass to `supersedes` or `retire`. An unknown or ambiguous id changes nothing and says so,
because marking the wrong fact is worse than doing nothing.

This is also why the staging cap no longer destroys anything: past it, the oldest notes are marked
`archived` rather than dropped.

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
| `HERMES_RAG_PROJECTS_FILE` | `~/.hermes/projects.yaml` | which project a directory belongs to |
| `HERMES_RAG_EDITOR_MODEL` | `granite4:3b` | the model that merges notes into the wiki |

If your project keeps generated JSON indexes or caches in a directory of their own, add it to
`HERMES_RAG_SKIP_JSON_DIRS`. They are worthless for searching and, because JSON is dense, a
4500-character piece of it can exceed the embedding model's limit.

## Which project am I in?

Left alone, the engine names a project after the working directory it happens to be in. That is wrong
often enough to matter: two projects can share a directory name, and a session's working directory is
not necessarily the project being discussed. A small file removes the guess:

```yaml
# ~/.hermes/projects.yaml
- path: /home/me/code/site-a
  project: site-a
- path: /home/me/code/api-server
  project: api
```

The longest matching path wins, so a directory inside another can name its own project. With no file,
or no match, the directory's name is used as before.

## Install

```
python scripts/install.py                    # every profile it finds
python scripts/install.py --dry-run          # show the plan, change nothing
python scripts/install.py --profile default  # just one profile
python scripts/install.py --migrate-from hermes-rag   # replace an older server entry
```

It checks what it needs first and stops if something is missing, saying the exact command to fix
it. Then it builds a **private environment** for the memory server and installs this package into
that — deliberately not into Hermes's own environment, because installing a vector database into
Hermes is how a working Hermes stops working.

It proves the server starts *before* it touches any config, because a config pointing at a server
that does not start is worse than no config at all. Config files are backed up before being edited,
and the edit is textual, so your comments survive. Running it twice changes nothing.

Two things it does that are easy to miss:

- it sets `memory.provider` to the bundled **holographic** provider, which is the part that recalls
  facts before each message — see "Where the facts live" below;
- it loads the entries already in `MEMORY.md` / `USER.md` into that fact store, so the standing rules
  the agent has been following are there from the first session rather than the next one.

## Where the facts live

Two different things, in two different places, on purpose:

| | what | where |
|---|---|---|
| facts, preferences, decisions — recalled before every message | the bundled `holographic` provider | `~/.hermes/memory_store.db` (one SQLite file) |
| code and written pages — searched when asked for | this project's server | `~/hermes-rag` plus `~/.hermes/wikis/` |

The split exists because they have different jobs. A preference needs to come back on its own,
every turn. A page about how a project works is looked up when a question needs it. The first is a
few hundred short facts; the second is thousands of text pieces.

## The window

A chip in the desktop status bar opens a small panel: what each project holds, how many notes are
waiting to be merged, when the last backup ran, and buttons for **Backup now**, **Export…**,
**Import…**, **Claude Code memory** and **Review**, plus the switch that turns pushing the backup to
a remote on and off.

The window cannot run commands itself, and it must not — the store's libraries are not in Hermes's
environment. Every button calls this plugin's own backend routes
(`/api/plugins/hermes-memory-rag/...`), which run the scripts below in the environment they were
installed into.

## Scheduled jobs

Three, as separate triggers rather than one bundled job, so a failure in one is obvious:

| when | what | runs |
|---|---|---|
| daily 03:00 | back up and commit | `scripts/backup.py` |
| daily 04:00 | read Claude Code's memory files, then re-index the pages | `scripts/claude_memory_import.py`, `scripts/reindex_wiki.py` |
| weekly Monday 05:00 | review: inventory, near-duplicates, names that no longer exist | `scripts/weekly_review.py` |

These are three separate jobs rather than one do-everything job: a failure in the import must not stop
the backup, and each can be paused on its own. They run as scripts with no model involved
(`--no-agent`), so they cost nothing and cannot invent anything.

The jobs are shell wrappers under `~/.hermes/scripts/` (`scripts/jobs/*.sh` here, with the paths filled
in at install time). A wrapper is needed because the two halves want different interpreters: the
importer touches the fact store, which lives inside Hermes, while re-indexing touches the vector
library, which deliberately lives outside it. **Copying pages without re-indexing them leaves files
that nothing can find**, which is why the second step is part of the same job.

The daily jobs stay silent when there is nothing to say and print when something changed or failed
(failures go to Telegram so a quiet job cannot fail invisibly). The weekly review speaks every time.

The weekly review speaks every time, because a report that only appears when it is worried is a
report nobody reads.

## Bringing in Claude Code's memory

If you also use Claude Code, it keeps curated notes per project at
`~/.claude/projects/<slug>/memory/*.md`. `scripts/claude_memory_import.py` files them where they will
be used: files named `feedback*` become facts (they are about how to work, so they belong in the
per-turn store), and everything else is copied into `~/.hermes/wikis/<project>/imported/` as a wiki
page and indexed with the rest. The raw conversation transcripts next to them are left alone —
hundreds of megabytes of logs is not knowledge.

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

One command writes everything worth keeping into a git repository and commits it:

```
python scripts/backup.py                  # write, commit, do not push
python scripts/backup.py --push           # push, if the window has pushing turned on
```

That repository holds every fact as readable text plus the numbers search compares, the wiki, each
profile's `MEMORY.md` / `USER.md`, the fact store as JSON, the project map, and a short record of
which profiles had which server registered. The Hermes config itself is deliberately **not** copied
— it holds credentials, and a backup repository that ends up on a remote must never carry those.
Pushing is off until it is turned on, and it refuses outright to push to a public repository.

The pieces, if you want them separately:

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

A store written by an older version of this project has notes with no status. `scripts/migrate_status.py`
fills that in — dry run by default, `--apply` to write. It only adds fields: nothing is rewritten or
removed, and running it twice changes nothing.

## What it stores, and privacy

Everything stays on your machine. Nothing is uploaded. But be clear about what the store contains:
the learnings you deposit, your wiki pages, and **excerpts of your project's source files**. Treat
`~/hermes-rag` like a copy of your notes and parts of your code — back it up if it matters, and keep
it out of any repository.

## Status

Built and testable: the installer, the server (recall, learn, retire, stats, ingest), the status
model that makes replacement and withdrawal reversible, the cross-project layer, project attribution
from a file of paths, the Claude Code importer, the backup and review scripts, the scheduled jobs,
the desktop window with its backend routes, and the scripts for merging, migrating, exporting,
restoring and comparing. Covered by tests, and exercised against a copy of a real 380-node store.

Known limits, stated rather than discovered later:

- the fact half is Hermes's bundled provider, not this project — this project supplies the code and
  written-page half;
- that provider's automatic fact extraction (off by default) stores the *whole message* it matched
  on, not a distilled fact, so a chatty week leaves near-duplicates behind; `fact_store` can remove
  them;
- its retrieval ranks by keyword, not by meaning, so a question worded differently from the note can
  miss it;
- merging notes into the written pages uses a small local model. It validates its own output before
  writing and falls back to appending when the check fails, which happens on roughly a third of
  pages. The fallback text is verbatim, including whatever phrasing the note was written with;
- `scripts/migrate_to_cosine.py` (changing how distance is measured) is written but unproven: the
  first attempt to test it compared two stores that measured distance differently, so the result
  meant nothing. Do not trust it until it has been measured properly.

## Licence

MIT.