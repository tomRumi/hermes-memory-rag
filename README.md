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
learn(text, kind="learning", project="", supersedes="", status="", origin="") -> str
retire(node_id, reason="", project="") -> str
verify(node_id, project="", note="") -> str
stats(project="") -> str
list_projects() -> str
ingest_code(root, project="", rebuild=True) -> str
ingest_wiki(wiki_dir, project="") -> str
reindex_wiki(wiki_dir, project="", mode="incremental", prune=True, dry_run=False,
             exclude="", budget_seconds=60, allow_empty=False) -> str
```

`recall` searches wiki, then code, then memory, and returns a bounded amount of text — two wiki
sections, three code pieces and one learning by default, 600 characters each. If it returned more,
every message would cost more. It also searches a cross-project layer last, labelled `[global]`, so a
fact that applies everywhere is reachable without being filed under a project.

`recall(query, project="*")` searches every project of the caller's namespace (never another
namespace) and keeps the best hits per layer overall. Each hit is tagged with its project:

```
Recall [*] for: deploy lane
[wiki@site deploy.md] ## Deploy the site deploys with the blue lane
[code@api service.py] def deploy_lane(): ...
[memory@site memory #c129e934 unverified] site deploy claim from an agent
(1 unverified note(s): written by an agent, not yet confirmed; treat as a lead and verify before relying on it)
```

The `layer` argument works as usual, a layer with nothing anywhere prints `[wiki: not built for *]`,
and it writes no usage-ledger row. `list_projects()` returns the caller's projects (slugs) with node
counts per layer and unverified counts. Both are read-class, so a read-only token may use them.
Calls without `"*"` print exactly what they always did.

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
| `unverified` | live, but an agent's claim nobody has confirmed; recall labels it | `learn(..., status="unverified")`, or a token that forces it |

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
| `HERMES_RAG_TOKENS_FILE` | `~/.config/memory-rag/tokens.json` | bearer tokens accepted over HTTP (see below) |
| `MEMORY_RAG_LOCAL_NAMESPACE` | `legacy` | namespace of the local caller (stdio, in-process scripts) |
| `HERMES_RAG_TOKENS_DIR` | `tokens.d` beside the tokens file | the token directory (see below) |
| `HERMES_RAG_NAMESPACES_FILE` | `namespaces.json` beside the tokens file | which folders a namespace may index (see below) |
| `HERMES_RAG_RESERVED_NAMESPACES` | `vps-host` | comma list of namespaces the token directory may never vouch for (`legacy` always is) |
| `HERMES_RAG_FOLDER_DENY` | empty | colon list of folders that may never be indexed, whatever a namespace allows |

If your project keeps generated JSON indexes or caches in a directory of their own, add it to
`HERMES_RAG_SKIP_JSON_DIRS`. They are worthless for searching and, because JSON is dense, a
4500-character piece of it can exceed the embedding model's limit.

## Who may call it (HTTP)

Over stdio the caller is the local user and nothing is checked: it may write and ingest,
in the namespace `MEMORY_RAG_LOCAL_NAMESPACE` names (default `legacy`; an invalid value is
refused, not replaced). Served over HTTP
(`HERMES_RAG_TRANSPORT=streamable-http`) every request needs
`Authorization: Bearer <token>`, and the server will not start without a tokens file or a token
directory (below).
The file holds the sha256 of each token, never the token. It must be a regular file
(not a symlink) owned by the server user with mode 0600, in a directory of mode 0700;
otherwise every token is refused:

```json
{"<sha256 hex>": {"namespace": "acme", "ingest": false, "write": true}}
```

A token belongs to one namespace and everything it reads or writes lives inside it:
collections become `rag_<namespace>.<project>__<layer>`, and the usage ledger is
scoped the same way. The reserved namespace `legacy` is unprefixed, i.e. the store as
it was before namespaces. `ingest` (default false) allows `ingest_code` / `ingest_wiki` /
`reindex_wiki`, which read the server's disk and change whole layers (and see "Which folders may be
indexed"). `write` (default true, so older
entries keep working) set to `false` makes a read-only token: it may call `recall` and
`stats`, and every tool that changes the store (`learn`, `retire`, `verify`,
`ingest_code`, `ingest_wiki`, `reindex_wiki`) refuses it with "this token is read-only" before touching anything.
`"ingest": true` with `"write": false` contradicts itself: that entry is refused and
logged, the rest of the file stays valid. A `write` that is not a JSON boolean (e.g.
`"false"`) is malformed and, like any malformed entry, invalidates the whole file. The
file is re-read when it changes and consulted on every call, so tokens can be added,
revoked or have their `ingest` / `write` rights changed without a restart, even inside
an open session. Generate one with
`python3 -c "import secrets;print(secrets.token_urlsafe(32))"` and store
`printf %s "$TOKEN" | sha256sum`.

To ingest a wiki into a namespace locally, e.g. `vps-host`:

```bash
HERMES_RAG_STORE=/home/ubuntu/memory-store .venv/bin/python scripts/reindex_wiki.py \
    --wiki-root ~/.hermes/wikis --project my-project --namespace vps-host
```

### Rights a token can carry

Besides `namespace`, `ingest` and `write`, an entry (in the tokens file or a token file) may carry:

- `"verify": true` (default false) lets the token call `verify`, which turns an unverified note into
  an active one. Like `ingest` it needs `write`; the two contradict each other otherwise and that
  entry is refused.
- `"learn_status": "unverified"` forces every note the token learns to be unverified, whatever the
  call asks for. An agent token should carry it; the token that confirms notes should not.

## Unverified notes

An agent's note is a claim. `learn(..., status="unverified")` (or a token with `learn_status`) stores
it with `status: unverified`. It is live, so `recall` returns it, but labelled inside the bracket,
after the id, and one footer line counts them:

```
[memory memory #1a2b3c4d unverified] the deploy lane is blue
(1 unverified note(s): written by an agent, not yet confirmed; treat as a lead and verify before relying on it)
```

Active notes and wiki or code hits print exactly as before. `verify(node_id)` (a token with the
`verify` right, or the local owner) promotes a note to `active` and records `verified_at`,
`verified_by` and an optional note. Learning the exact text of an existing unverified note promotes
it only for a caller that may verify and passes `status="active"`; an active note is never demoted.
`origin` (at most 80 characters) records who wrote a note. Unverified notes count toward the staging
cap (so they cannot pile up without bound). Past the cap the oldest *unverified* notes are archived
first, and active notes only for whatever excess remains, so an agent that writes a lot can only
push out its own unconfirmed claims, are never
merged into the wiki by `scripts/consolidate.py`, and `scripts/weekly_review.py` lists the ones
waiting longer than `HERMES_RAG_UNVERIFIED_REVIEW_DAYS` (default 7). `stats` reports
`unverified_notes`.

## Provisioning tokens: the token directory

A provisioner (the Paperclip plugin, for one) should not edit the operator's `tokens.json`. It
writes one file per namespace into the token directory instead, `tokens.d/` beside the tokens file
(`HERMES_RAG_TOKENS_DIR` overrides it). Create it once, no root needed:

```bash
install -d -m700 ~/.config/memory-rag/tokens.d
```

Format v1, one file `<namespace>.json` (mode 600) per namespace:

```json
{"version": 1,
 "namespace": "acme",
 "tokens": {
   "<sha256 of the agent token>": {"write": true, "ingest": false, "verify": false, "learn_status": "unverified"},
   "<sha256 of the admin token>": {"write": true, "ingest": true, "verify": true}
 },
 "allowed_folders": ["/srv/acme/llm-wiki"]}
```

**Writer protocol.** Write the new content to a dotfile in the same directory, `.acme.json.tmp`
(mode 600), then `rename` it onto `acme.json`. Anything that does not look like `<name>.json`
(dotfiles, `*.tmp`, other suffixes) is ignored, so a half-written file is never read. Adding a
company is writing its file, rotating a token is rewriting it, revoking is deleting it; nobody
reads, edits and writes back a shared file. The server looks again on every call, so all three take
effect at once, even inside an open session. Keep the plaintext tokens in the provisioner's own
state; only digests go here.

What the server enforces: the directory must be a real directory (not a link) owned by the server
user, mode 700; each file a regular file (not a link), same owner, mode 600. The file stem must
equal `namespace`, which must be valid and **not reserved**: `legacy` always, plus
`HERMES_RAG_RESERVED_NAMESPACES` (default `vps-host`). A reserved namespace's tokens can only come
from the operator's own tokens file, so no provisioner bug or web request can mint a host token;
add the host's token to `tokens.json` by hand. A malformed file (unknown key, wrong type, version
other than 1, more than 8 tokens) is ignored and logged without touching other files; at most 200
files are read. A digest claimed under two namespaces is refused everywhere; on the same namespace
the tokens file wins. A broken `tokens.json` refuses every token, directory included.
`ingest` and `verify` need `write`; `ingest` is granted only if the file lists `allowed_folders`.

The server starts over HTTP with only the token directory (even empty), so a fresh install can be
provisioned later. With neither a tokens file nor the directory it still refuses to start.

**What this does not protect against.** The provisioner and the server normally run as the same unix
user. These rules stop mistakes, agents and web-page requests; they cannot stop code already running
as that user, which can write any file the server trusts. Do not count on them against it.

## Which folders may be indexed

`ingest_code`, `ingest_wiki` and `reindex_wiki` read the server's own disk, so the `ingest` right
alone would let a token index any file the service user can read. Give a namespace a list of folders
in the operator's `namespaces.json` (`HERMES_RAG_NAMESPACES_FILE`; same ownership and mode rules as
the tokens file, re-read when it changes):

```json
{"vps-host": {"allowed_folders": ["/home/me/wikis", "/home/me/code"]},
 "factory-lab": {"allowed_folders": []}}
```

A path is accepted only if its **real** path (links and `..` resolved) is one of those folders or
inside one, compared by whole path components. The check runs first on every index path, before
anything is read or deleted, and the error does not list the folders. Rules:

- a namespace with an entry here is capped by it; a token file's own `allowed_folders` count only
  where they lie inside that list;
- a namespace with a token file always has a list, empty (nothing may be indexed) unless the file
  says otherwise;
- a namespace with neither is **unrestricted**, as before this existed. Add an entry for `vps-host`
  and any other namespace whose token has `ingest`;
- an entry with an empty list indexes nothing; an unusable `namespaces.json` refuses all indexing
  until fixed;
- never allowed, whatever the lists say: the store, the directories holding the tokens file, the
  token directory and `namespaces.json`, anything in `HERMES_RAG_FOLDER_DENY`, and `/` or your home
  directory as an allowed folder. A walk from a folder above them does not enter them.

Inside a folder, links never lead out: a directory link is not entered, a file link is read only if
its real target is inside the folder (otherwise it is skipped and reported), and the file is opened
without following a link at the last step. `ingest_code` and `ingest_wiki` also no longer delete a
layer before finding anything to index: an empty or mistyped folder leaves the layer alone. The
local scripts are checked too whenever their namespace has an entry.

## Re-indexing a wiki folder in place

`ingest_wiki` rebuilds a layer from scratch. `reindex_wiki` keeps it in step with a folder instead:

```bash
# over MCP: reindex_wiki(wiki_dir="/srv/acme/llm-wiki", project="llm-wiki", exclude="raw")
.venv/bin/python scripts/reindex_wiki.py --incremental --prune --project llm-wiki --namespace acme
```

Node ids are derived from the folder, the page, the position and the text, so unchanged pages cost
a hash and no embedding, an edited page replaces its own chunks (new vectors are stored before old
ones go), and repeated or interrupted runs converge. `prune` (default) removes pages that are gone
or can no longer be read safely. Only nodes this tool wrote for this folder are touched; pages
`ingest_wiki` put in the same project stay. `mode="incremental"` (default) stops starting
pages after `budget_seconds` and answers `partial: true` with `remaining`; call again to continue
(at least one page is done per call). `mode="full"` re-embeds every page and runs to completion. A folder with no pages while the index holds some from it is refused (an unmounted
disk must not empty the index); `mode="full", prune=true, allow_empty=true` does it on purpose.
`exclude` is a comma list of top-level subfolders; hidden folders are always skipped.
`dry_run` returns the plan and writes nothing. The tool is asynchronous and runs in a worker thread,
so other namespaces keep being served; a second call for the same project while one runs answers
`{"busy": true}`. It needs the `ingest` right and a folder the namespace may index. The answer is
JSON with `files` (seen, unchanged, indexed, removed, skipped), `nodes` (added, deleted), `partial`,
`remaining` and the first 20 `skipped` paths with a reason.

`GET /healthz` answers `{"ok": true}` with no token, for a settings page or a supervisor. It says
nothing about the store; the authenticated check is `stats`.

### Moving a namespace

`scripts/migrate_namespace.py` copies every collection of one namespace to another in
the same store (ids, documents, metadatas and embeddings exactly, no re-embedding, index
settings kept), copies the usage-ledger rows with their scope rewritten, rewrites a
`namespace` metadata field where one names the source, and verifies counts, id sets and
a sample of embeddings. It never deletes the source, refuses an existing target unless
`--force`, and refuses a store another process holds open unless `--i-stopped-the-service`.
Stop the service and back up first:

```bash
systemctl --user stop memory-rag
.venv/bin/python scripts/backup.py --store /home/ubuntu/memory-store
cp -a /home/ubuntu/memory-store /home/ubuntu/memory-store.pre-vps-host  # raw copy: fastest rollback
.venv/bin/python scripts/migrate_namespace.py --store /home/ubuntu/memory-store \
    --from legacy --to vps-host --dry-run          # names and counts only
.venv/bin/python scripts/migrate_namespace.py --store /home/ubuntu/memory-store \
    --from legacy --to vps-host --i-stopped-the-service
systemctl --user start memory-rag
```

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

## Run the tests

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e '.[test]' && .venv/bin/python -m pytest -q
```

The tests use temp directories and a built-in embedder: no Ollama, no real store, no tokens file and
no port are touched. `tests/conftest.py` points the tokens file, token directory and `namespaces.json`
at temp paths for every test and fails any test that still opens or lists `~/.config/memory-rag`.

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