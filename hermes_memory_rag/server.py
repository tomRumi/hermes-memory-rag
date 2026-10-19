"""hermes-memory-rag — layered project memory as an MCP server.

Three collections per project in one VM-local Chroma store (`~/hermes-rag/`):

- `rag_<slug>__code`   raw code chunks (detail layer)
- `rag_<slug>__wiki`   code-wiki markdown, heading-aware (map layer)
- `rag_<slug>__memory` episodic session learnings (staging area, capped)

Tools exposed to every Hermes session (any profile, any project):

- `recall(query, project?, layer?, top?)` — bounded layered retrieval.
- `learn(text, kind, project?)` — deposit ONE learning (never logging).
- `stats(project?)` — layer counts + wiki-earn suggestion signals.
- `retire`, `verify`, `ingest_code`, `ingest_wiki`, `reindex_wiki` — see their docs.

Session-end procedure (encoded in the `project-memory` skill): deposit
learnings via `learn`; check `stats` for the wiki-earn suggestion.

Access control (HTTP transport only): every request must carry a bearer token
whose sha256 is listed in the tokens file. A token maps to one namespace; every
collection and ledger row it touches is confined to that namespace, ingest_* is
refused unless the token is explicitly allowed it, and a read-only token may call
only the tools that change nothing. See `_caller` and TOOL_ACCESS below.

Design invariants:
- embed-before-store; nothing reaches Chroma without a vector
- deterministic IDs → idempotent upserts
- loud failures: a partial or empty index never looks like success
- retrieval is bounded (char caps) and honest ("not built" vs "no results")
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio.to_thread
import chromadb
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import fsafe

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
# Wiki text is prose, often Hungarian/accented: ~1.5-2 chars/token, so a lower cap.
WIKI_CHUNK_CHAR_CAP = int(os.environ.get("HERMES_RAG_WIKI_CHUNK_CHAR_CAP", "2400"))

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
# Written by an agent and not yet confirmed by anyone who may confirm it. It is live:
# recall returns it, but labelled, with a footer saying how far to trust it. `verify`
# promotes it to active; nothing else does. See `learn` and `verify`.
STATUS_UNVERIFIED = "unverified"
NON_LIVE_STATUSES = (STATUS_SUPERSEDED, STATUS_RETIRED, STATUS_ARCHIVED)
LIVE_STATUSES = (STATUS_ACTIVE, STATUS_UNVERIFIED)

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

# ── who is calling? ───────────────────────────────────────────────────────
# Served over HTTP the store is shared by several companies, so a request must
# prove who it is. A token is known by its sha256 (never the token itself), which
# maps to a namespace and the rights it carries. Two places can vouch for a digest:
#
# 1. The operator's tokens file, one JSON object for the whole server:
#
#      {"<sha256 hex>": {"namespace": "acme", "ingest": false, "write": true,
#                        "verify": false, "learn_status": ""}, ...}
#
# 2. The token directory `tokens.d/` next to it, one file per namespace, written by
#    software that provisions companies (see `_read_token_file` for the format and
#    the README for the writer protocol). Adding a company is writing a file,
#    revoking it is deleting the file; nobody edits a shared file.
#
# A namespace scopes everything a token can read or write. The reserved namespace
# "legacy" is unprefixed: it is the store as it was before namespaces existed, so
# existing data stays reachable under it. Rights, all re-read on every call:
#
# - `ingest` (default false) lets a token run ingest_code / ingest_wiki /
#   reindex_wiki, which read the server's own disk and rebuild layers; agent tokens
#   should not have it. Which folders it may read is `_check_index_path`'s business.
# - `write` (default true, so entries written before it existed keep working) set to
#   false makes a read-only token: it may call the read tools in TOOL_ACCESS only.
# - `verify` (default false) lets a token promote an unverified note with `verify`
#   (the local owner always may).
# - `learn_status` "unverified" forces every note the token learns to be unverified,
#   whatever the call asks for.
#
# "ingest": true or "verify": true with "write": false contradicts itself; that
# entry is refused (fail closed) and logged, while the file's other entries stay
# valid. A `write` that is present but not a JSON boolean, or a `learn_status` that
# is not "" or "unverified", is malformed and rejects the whole file, like any other
# malformed entry: "false" as a string must never mean writable.
#
# Fail closed: a missing, unreadable, group/world-accessible or malformed file
# means no token is valid. Over stdio (or an in-process call from a script) there
# is no network peer, so the caller is the local process owner: ingest and write
# allowed, in the namespace named by MEMORY_RAG_LOCAL_NAMESPACE (default
# "legacy", as before namespaces). That lets an operator ingest straight into, say,
# "vps-host". The variable is read on every call; a value that is not a valid
# namespace is refused, never quietly replaced by "legacy".
#
# Token and plugin share a unix user in the usual setup, so these rules stop
# mistakes, agents and web-page requests, not code already running as that user.

LEGACY_NAMESPACE = "legacy"
LOCAL_NAMESPACE_ENV = "MEMORY_RAG_LOCAL_NAMESPACE"
TOKENS_FILE = Path(os.environ.get(
    "HERMES_RAG_TOKENS_FILE", str(Path.home() / ".config" / "memory-rag" / "tokens.json")))
TOKENS_DIR_ENV = "HERMES_RAG_TOKENS_DIR"
NAMESPACES_FILE_ENV = "HERMES_RAG_NAMESPACES_FILE"
RESERVED_NAMESPACES_ENV = "HERMES_RAG_RESERVED_NAMESPACES"
DEFAULT_RESERVED_NAMESPACES = ("vps-host",)
FOLDER_DENY_ENV = "HERMES_RAG_FOLDER_DENY"
TOKEN_FILE_VERSION = 1
MAX_TOKEN_FILES = 200
MAX_TOKENS_PER_FILE = 8
MAX_FOLDERS_PER_LIST = 64
_NAMESPACE_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_log = logging.getLogger("hermes_memory_rag")


def _valid_namespace(namespace: object) -> bool:
    return (isinstance(namespace, str) and len(namespace) <= 40
            and _NAMESPACE_RE.fullmatch(namespace) is not None)


def _tokens_dir() -> Path:
    """The token directory: HERMES_RAG_TOKENS_DIR, else `tokens.d` beside the tokens file."""
    configured = os.environ.get(TOKENS_DIR_ENV, "").strip()
    return Path(configured).expanduser() if configured else TOKENS_FILE.parent / "tokens.d"


def _namespaces_file() -> Path:
    """The operator's folder policy: HERMES_RAG_NAMESPACES_FILE, else `namespaces.json`
    beside the tokens file."""
    configured = os.environ.get(NAMESPACES_FILE_ENV, "").strip()
    return Path(configured).expanduser() if configured else TOKENS_FILE.parent / "namespaces.json"


def _reserved_namespaces() -> frozenset[str]:
    """Namespaces no token directory may vouch for: `legacy` always, plus the
    comma list in HERMES_RAG_RESERVED_NAMESPACES (default: vps-host). Their tokens
    can only come from the operator's own tokens file."""
    configured = os.environ.get(RESERVED_NAMESPACES_ENV)
    extra = (DEFAULT_RESERVED_NAMESPACES if configured is None
             else tuple(n.strip() for n in configured.split(",") if n.strip()))
    return frozenset(extra) | {LEGACY_NAMESPACE}


@dataclass(frozen=True)
class Caller:
    namespace: str
    ingest: bool
    write: bool = True
    verify: bool = False
    learn_status: str = ""


def _entry_caller(namespace: str, entry: dict, *, where: str, strict: bool) -> Caller | None:
    """The Caller an entry describes; None if the entry contradicts itself.

    Raises ValueError if it is malformed. `strict` (token directory files) demands
    real booleans for every right and refuses keys it does not know, so a typo such
    as "ingset" fails loudly instead of quietly granting less or more.
    """
    known = {"ingest", "write", "verify", "learn_status"} | (set() if strict else {"namespace"})
    if strict and set(entry) - known:
        raise ValueError(f"{where} has unknown keys {sorted(set(entry) - known)}")
    write = entry.get("write", True)
    if not isinstance(write, bool):
        raise ValueError(f"{where} has a malformed write")
    if strict:
        for right in ("ingest", "verify"):
            if not isinstance(entry.get(right, False), bool):
                raise ValueError(f"{where} has a malformed {right}")
    learn_status = entry.get("learn_status", "")
    if learn_status not in ("", STATUS_UNVERIFIED):
        raise ValueError(f"{where} has a malformed learn_status")
    ingest = entry.get("ingest") is True
    verify = entry.get("verify") is True
    if (ingest or verify) and not write:
        # A contradiction, not a typo to guess around: neither right is granted.
        _log.error("%s: %s without write — refusing that token", where,
                   "ingest" if ingest else "verify")
        return None
    return Caller(namespace, ingest, write, verify, learn_status)


def _read_tokens(path: Path) -> dict[str, Caller]:
    """sha256 hex -> Caller from the tokens file. Raises ValueError if it is unfit.

    One malformed entry rejects the whole file: half-trusting a file nobody has
    reviewed is how a typo becomes an open door. A well-formed entry that asks for
    ingest without write is contradictory; only that entry is refused.
    """
    raw = fsafe.read_private_json(path, what="tokens file")
    if not isinstance(raw, dict):
        raise ValueError(f"tokens file {path} must be a JSON object")
    tokens: dict[str, Caller] = {}
    for digest, entry in raw.items():
        namespace = entry.get("namespace") if isinstance(entry, dict) else None
        if not _DIGEST_RE.fullmatch(digest) or not _valid_namespace(namespace):
            raise ValueError(f"tokens file {path} has a malformed entry for {digest[:12]!r}")
        where = f"tokens file {path}: entry {digest[:12]!r}"
        try:
            caller = _entry_caller(namespace, entry, where=where, strict=False)
        except ValueError as exc:
            raise ValueError(f"tokens file {path} has a malformed entry for {digest[:12]!r}") from exc
        if caller is not None:
            tokens[digest] = caller
    return tokens


def _clean_folders(value: object, *, where: str) -> tuple[str, ...]:
    """A list of absolute folder paths, as written in a policy or token file.

    Raises ValueError unless it is a list of at most MAX_FOLDERS_PER_LIST strings,
    each absolute (a leading ~ is expanded). Nothing is resolved here: a folder is
    resolved at the moment it is used, because links can change.
    """
    if not isinstance(value, list) or len(value) > MAX_FOLDERS_PER_LIST:
        raise ValueError(f"{where} allowed_folders must be a list of at most {MAX_FOLDERS_PER_LIST}")
    folders = []
    for item in value:
        if not isinstance(item, str) or not item or "\0" in item:
            raise ValueError(f"{where} allowed_folders holds a non-path")
        expanded = os.path.expanduser(item)
        if not os.path.isabs(expanded):
            raise ValueError(f"{where} allowed_folders holds a relative path {item!r}")
        folders.append(os.path.normpath(expanded))
    return tuple(folders)


@dataclass(frozen=True)
class _TokenFile:
    namespace: str
    tokens: dict[str, Caller]
    folders: tuple[str, ...]


def _read_token_file(path: Path, *, reserved: frozenset[str]) -> _TokenFile:
    """One file of the token directory (format v1). Raises ValueError if unfit.

    ``{"version": 1, "namespace": "<file stem>",
       "tokens": {"<sha256 hex>": {"ingest": false, "write": true, "verify": false,
                                   "learn_status": "unverified"}},
       "allowed_folders": ["/abs/dir"]}``

    The file must be a regular file owned by the server user, mode 600, in a directory
    of mode 700. The namespace must equal the file stem, be valid, and not be reserved
    (an operator-held namespace such as `legacy` or the host's can never be minted from
    here). `ingest` needs `write`, and is only granted when the file lists at least one
    folder: a token that may read the disk with nowhere to read is a mistake.
    """
    raw = fsafe.read_private_json(path, what="token file")
    where = f"token file {path}"
    if not isinstance(raw, dict) or set(raw) - {"version", "namespace", "tokens", "allowed_folders"}:
        raise ValueError(f"{where} must be a JSON object with version, namespace, tokens, allowed_folders")
    if raw.get("version") != TOKEN_FILE_VERSION or isinstance(raw.get("version"), bool):
        raise ValueError(f"{where} has an unsupported version")
    namespace = raw.get("namespace")
    if not _valid_namespace(namespace) or namespace != path.stem:
        raise ValueError(f"{where}: namespace must be valid and equal to the file name")
    if namespace in reserved:
        raise ValueError(f"{where}: namespace {namespace!r} is reserved to the operator's tokens file")
    entries = raw.get("tokens")
    if not isinstance(entries, dict) or len(entries) > MAX_TOKENS_PER_FILE:
        raise ValueError(f"{where}: tokens must be an object of at most {MAX_TOKENS_PER_FILE} entries")
    folders = _clean_folders(raw.get("allowed_folders", []), where=where)
    tokens: dict[str, Caller] = {}
    for digest, entry in entries.items():
        if not _DIGEST_RE.fullmatch(digest) or not isinstance(entry, dict):
            raise ValueError(f"{where} has a malformed entry for {digest[:12]!r}")
        caller = _entry_caller(namespace, entry, strict=True,
                               where=f"{where}: entry {digest[:12]!r}")
        if caller is None:
            continue
        if caller.ingest and not folders:
            _log.error("%s: entry %r asks for ingest but the file lists no allowed_folders "
                       "— ingest not granted", where, digest[:12])
            caller = Caller(namespace, False, caller.write, caller.verify, caller.learn_status)
        tokens[digest] = caller
    return _TokenFile(namespace, tokens, folders)


def _token_file_candidates(directory: Path) -> list[os.DirEntry]:
    """The directory entries that may be token files: `<name>.json`, not a dotfile.

    Everything else (temporary files, dotfiles, other suffixes) is ignored, which is
    what lets a writer stage `.<ns>.json.tmp` and rename it into place atomically.
    """
    with os.scandir(directory) as it:
        names = sorted((e for e in it if e.name.endswith(".json") and not e.name.startswith(".")),
                       key=lambda e: e.name)
    if len(names) > MAX_TOKEN_FILES:
        _log.error("token directory %s holds %d files; only the first %d are read",
                   directory, len(names), MAX_TOKEN_FILES)
    return names[:MAX_TOKEN_FILES]


@dataclass(frozen=True)
class _TokenSet:
    tokens: dict[str, Caller]
    # namespace -> folders listed by its token file (present even when empty)
    provisioned: dict[str, tuple[str, ...]]


class _FileTokenVerifier:
    """Verifies bearer tokens against the tokens file and the token directory,
    re-read when either changes so a token can be added, revoked or changed without
    restarting the server."""

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._set = _TokenSet({}, {})

    @staticmethod
    def _stat_key(path: Path) -> tuple:
        try:
            st = os.lstat(path)
        except OSError:
            return (str(path), None)
        return (str(path), st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def _fingerprint(self, reserved: frozenset[str]) -> tuple:
        """What identifies the inputs now: both files and their directories (a
        directory's mode is judged too), the token directory's listing, and the
        reserved-namespace setting. Costs a few stats per call."""
        directory = _tokens_dir()
        parts: list = [self._stat_key(TOKENS_FILE), self._stat_key(TOKENS_FILE.parent),
                       self._stat_key(directory), tuple(sorted(reserved))]
        try:
            parts.extend(self._stat_key(Path(e.path)) for e in _token_file_candidates(directory))
        except OSError:
            pass
        return tuple(parts)

    def _load(self, reserved: frozenset[str]) -> _TokenSet:
        claims: dict[str, list[tuple[str, Caller]]] = {}
        provisioned: dict[str, tuple[str, ...]] = {}
        if os.path.lexists(TOKENS_FILE):
            try:
                for digest, caller in _read_tokens(TOKENS_FILE).items():
                    claims.setdefault(digest, []).append((caller.namespace, caller))
            except ValueError as exc:
                _log.error("%s — refusing every token", exc)
                return _TokenSet({}, {})
        directory = _tokens_dir()
        if os.path.lexists(directory):
            try:
                fsafe.check_private_dir(directory, what="token directory")
                candidates = _token_file_candidates(directory)
            except (OSError, ValueError) as exc:
                _log.error("%s — ignoring the token directory", exc)
                candidates = []
            for entry in candidates:
                try:
                    found = _read_token_file(Path(entry.path), reserved=reserved)
                except ValueError as exc:
                    _log.error("%s — ignoring that file", exc)
                    # A broken file must not lift the namespace's folder limit: it
                    # still counts as provisioned, with no folders at all.
                    stem = Path(entry.name).stem
                    if _valid_namespace(stem) and stem not in reserved:
                        provisioned.setdefault(stem, ())
                    continue
                provisioned[found.namespace] = found.folders
                for digest, caller in found.tokens.items():
                    claims.setdefault(digest, []).append((found.namespace, caller))
        tokens: dict[str, Caller] = {}
        for digest, claimed in claims.items():
            if len({namespace for namespace, _ in claimed}) > 1:
                _log.error("token %r is listed under several namespaces — refusing it everywhere",
                           digest[:12])
                continue
            tokens[digest] = claimed[0][1]  # the tokens file is read first, so it wins
        return _TokenSet(tokens, provisioned)

    def _current_set(self) -> _TokenSet:
        reserved = _reserved_namespaces()
        key = self._fingerprint(reserved)
        if key != self._key:
            self._set = self._load(reserved)
            self._key = key
        return self._set

    def _current(self) -> dict[str, Caller]:
        return self._current_set().tokens

    def provisioned_folders(self) -> dict[str, tuple[str, ...]]:
        """namespace -> folders its token file lists, for every namespace that has one."""
        return self._current_set().provisioned

    async def verify_token(self, token: str) -> AccessToken | None:
        digest = hashlib.sha256(token.encode()).hexdigest()
        caller = self._current().get(digest)
        if caller is None:
            return None
        # client_id is what the SDK compares to keep a session with its creator, so
        # it names the token, not just the namespace. What the token may do is not
        # carried here: _caller asks the file again on every call.
        return AccessToken(
            token=digest, client_id=f"{caller.namespace}:{digest[:12]}", scopes=[])


_verifier = _FileTokenVerifier()


def _local_namespace() -> str:
    """The namespace of the local process owner (stdio / in-process caller)."""
    namespace = os.environ.get(LOCAL_NAMESPACE_ENV, "").strip() or LEGACY_NAMESPACE
    if not _valid_namespace(namespace):
        raise PermissionError(f"{LOCAL_NAMESPACE_ENV}={namespace!r} is not a valid namespace")
    return namespace


def _serving_remote() -> bool:
    return os.environ.get("HERMES_RAG_TRANSPORT", "stdio") != "stdio"


def _caller(*, write: bool = False) -> Caller:
    """The authenticated caller of the tool now running; PermissionError if none.

    A session lives longer than the request that opened it, so the token is looked
    up again on every call: revoking it, or changing its namespace, ingest or write
    right, takes effect at once, not when the session ends.

    `write=True` is what every tool that changes the store passes, as its first
    act, so a read-only token is turned away before anything is touched.
    """
    access = get_access_token()
    if access is not None:
        caller = _verifier._current().get(access.token)
        if caller is None:
            raise PermissionError("token no longer valid")
    elif _serving_remote():
        raise PermissionError("authentication required")
    else:
        caller = Caller(_local_namespace(), True, True)
    if write and not caller.write:
        raise PermissionError("this token is read-only: it may not change the store")
    return caller


def _caller_label(caller: Caller) -> str:
    """Who to record as the actor of a change: namespace and token digest prefix
    (never the token), or `<namespace>:local` for the local owner."""
    access = get_access_token()
    return access.client_id if access is not None else f"{caller.namespace}:local"


def _require_ingest(caller: Caller) -> None:
    if not caller.ingest:
        raise PermissionError("this token may not run ingest_*")


# ── which folders may be indexed? ─────────────────────────────────────────
# ingest_code, ingest_wiki and reindex_wiki read the server's own disk, so an ingest
# right alone would let a token index any file the service user can read, and make it
# recallable by that namespace. A namespace can therefore be given a list of folders;
# a path is accepted only if its REAL path (links resolved) lies inside one of them.
#
# Where the list comes from:
# - the operator's `namespaces.json`:
#     {"vps-host": {"allowed_folders": ["/abs/a", "/abs/b"]}}
#   for a namespace it names, that list is a ceiling nothing can raise;
# - the namespace's token file (`allowed_folders`), honoured only for the parts that
#   lie inside the operator's ceiling when there is one.
# A namespace with neither is unrestricted, as before this existed, so an install
# that never configures it behaves as it did. A namespace with a token file always
# has a list (empty means nothing may be indexed). An unusable policy file fails
# closed: nothing can be indexed until it is fixed.
#
# Whatever the lists say, the store, the directory of the tokens file, the token
# directory, the policy file's directory and HERMES_RAG_FOLDER_DENY (a colon list)
# are never allowed folders (nor anything inside them), `/` and the user's home are
# never allowed folders, and a walk from a folder above them does not enter them. An
# unrestricted namespace may still be pointed straight at such a place: restricting it
# is what a namespace entry is for.

_OUTSIDE_MESSAGE = "path is outside the folders allowed for this namespace"


class _OperatorFolders:
    """The operator's namespaces.json, re-read when it changes."""

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._value: dict[str, tuple[str, ...]] = {}
        self._error = ""

    def current(self) -> dict[str, tuple[str, ...]]:
        """namespace -> ceiling. A missing file is an empty policy; a file that is
        there but unfit raises PermissionError (fail closed)."""
        path = _namespaces_file()
        key = (_FileTokenVerifier._stat_key(path), _FileTokenVerifier._stat_key(path.parent))
        if key != self._key:
            self._value, self._error = {}, ""
            if os.path.lexists(path):
                try:
                    self._value = self._read(path)
                except ValueError as exc:
                    self._error = str(exc)
                    _log.error("%s — refusing every index request", exc)
            self._key = key
        if self._error:
            raise PermissionError("the folder policy file is unusable; indexing is refused")
        return self._value

    @staticmethod
    def _read(path: Path) -> dict[str, tuple[str, ...]]:
        raw = fsafe.read_private_json(path, what="namespaces file")
        if not isinstance(raw, dict):
            raise ValueError(f"namespaces file {path} must be a JSON object")
        policy: dict[str, tuple[str, ...]] = {}
        for namespace, entry in raw.items():
            if (not _valid_namespace(namespace) or not isinstance(entry, dict)
                    or set(entry) != {"allowed_folders"}):
                raise ValueError(f"namespaces file {path} has a malformed entry for {namespace!r}")
            policy[namespace] = _clean_folders(
                entry["allowed_folders"], where=f"namespaces file {path}: {namespace!r}")
        return policy


_operator_folders = _OperatorFolders()


def _deny_dirs() -> list[str]:
    """Real paths that may never be indexed, whatever a namespace's list says."""
    dirs = [STORE_DIR, TOKENS_FILE.parent, _tokens_dir(), _namespaces_file().parent]
    dirs += [Path(d).expanduser() for d in os.environ.get(FOLDER_DENY_ENV, "").split(":") if d.strip()]
    return sorted({os.path.realpath(d) for d in dirs})


def _effective_folders(namespace: str, deny: list[str]) -> list[str] | None:
    """The real folders a namespace may index; None means unrestricted."""
    ceiling = _operator_folders.current().get(namespace)
    own = _verifier.provisioned_folders().get(namespace)
    if ceiling is None and own is None:
        return None

    def usable(entries: tuple[str, ...]) -> list[str]:
        out = []
        for entry in entries:
            real = os.path.realpath(entry)
            if real in ("/", os.path.realpath(Path.home())) or any(
                    fsafe.is_inside(real, d) for d in deny):
                _log.error("namespace %s: allowed folder %r is not permitted — ignored",
                           namespace, entry)
                continue
            out.append(real)
        return out

    ceiling_real = None if ceiling is None else usable(ceiling)
    if own is None:
        return ceiling_real or []
    mine = usable(own)
    if ceiling_real is None:
        return mine
    return [m for m in mine if any(fsafe.is_inside(m, c) for c in ceiling_real)]


def _check_index_path(namespace: str, raw_path: str) -> Path:
    """The real path of a folder to index, if `namespace` may index it.

    Raises PermissionError otherwise, without saying which folders would be allowed.
    Nothing is read or deleted before this passes: callers run it first. Whether the
    path exists is not decided here (that would tell a caller about places it may not
    look); the caller finds out after this check.
    """
    deny = _deny_dirs()
    real = os.path.realpath(os.path.expanduser(raw_path))
    allowed = _effective_folders(namespace, deny)
    if allowed is not None and (
            not any(fsafe.is_inside(real, a) for a in allowed)
            or any(fsafe.is_inside(real, d) for d in deny)):
        raise PermissionError(_OUTSIDE_MESSAGE)
    return Path(real)


def _walk_deny(root_real: Path | str) -> Callable[[str, str], bool]:
    """A `prune_dir` for `fsafe.walk_files`: true for a protected place (see
    `_deny_dirs`) that lies below the folder being walked, so indexing a parent
    directory never reaches into the store or the token directories."""
    inside = [d for d in _deny_dirs()
              if d != str(root_real) and fsafe.is_inside(d, str(root_real))]
    return lambda rel, real: any(fsafe.is_inside(real, d) for d in inside)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stamp(metadata: dict, *, source: str) -> dict:
    """Add the fields every node carries: status, source (and created where absent)."""
    metadata.setdefault("status", STATUS_ACTIVE)
    metadata.setdefault("source", source)
    metadata.setdefault("created", _now())
    return metadata


def _meta_of(hit_or_meta) -> dict:
    meta = getattr(hit_or_meta, "metadata", None)
    if meta is None:
        meta = hit_or_meta if isinstance(hit_or_meta, dict) else {}
    return meta


def _is_live(hit_or_meta) -> bool:
    """False for a node that must not be handed to the model.

    Missing status counts as live: an un-migrated store still answers.
    """
    return (_meta_of(hit_or_meta).get("status") or STATUS_ACTIVE) not in NON_LIVE_STATUSES


def _is_unverified(hit_or_meta) -> bool:
    return _meta_of(hit_or_meta).get("status") == STATUS_UNVERIFIED


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

def _wiki_chunks(text: str, max_chars: int = WIKI_CHUNK_CHAR_CAP) -> list[str]:
    """Heading-aware chunks: a `##` section stays whole when it fits max_chars,
    otherwise it is split on whitespace and the heading repeats on every piece
    (the heading counts toward the budget)."""
    out: list[str] = []
    for i, part in enumerate(text.split("\n## ")):
        if i > 0:
            head, _, rest = part.partition("\n")
            body, title = rest.strip(), head.strip()
        else:
            body, title = part.strip(), ""
        if not body:
            continue
        prefix = f"## {title}\n" if title else ""
        if len(prefix) + len(body) <= max_chars:
            out.append(prefix + body)
            continue
        room = max(max_chars - len(prefix), 200)
        out.extend(prefix + piece for piece in _cap_chunks([body], max_chars=room))
    return out


_BASE_URL = "http://{}:{}".format(
    os.environ.get("HERMES_RAG_HOST", "127.0.0.1"), os.environ.get("HERMES_RAG_PORT", "8765"))
mcp = FastMCP(
    "hermes-memory-rag",
    token_verifier=_verifier,
    # Only used by the HTTP transports. The server is its own resource server and
    # validates tokens itself (there is no OAuth flow), hence no resource check.
    auth=AuthSettings(issuer_url=_BASE_URL, resource_server_url=_BASE_URL,
                      validate_token_resource=False),
)

# ── shared plumbing ───────────────────────────────────────────────────────


def _slugify(project: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", project.strip().lower()).strip("-")
    return slug or "default"


def _scope(project: str, namespace: str = LEGACY_NAMESPACE) -> str:
    """The project's slug inside a namespace. Slugs hold only [a-z0-9-], so the
    "." can never be forged by a project name and namespaces cannot collide."""
    slug = _slugify(project)
    return slug if namespace == LEGACY_NAMESPACE else f"{namespace}.{slug}"


def _collection_names(project: str, namespace: str = LEGACY_NAMESPACE) -> dict[str, str]:
    slug = _scope(project, namespace)
    return {
        "code": f"rag_{slug}__code",
        "wiki": f"rag_{slug}__wiki",
        "memory": f"rag_{slug}__memory",
    }


def _client() -> chromadb.PersistentClient:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(STORE_DIR))


def _count(client: chromadb.PersistentClient, name: str) -> int:
    """How many nodes a collection holds, without creating it.

    get_or_create was used here, which meant merely ASKING how big a layer was
    brought it into existence — so read-only calls (recall's global-layer check,
    stats) left empty collections behind, and the review then listed phantom
    projects. Nothing should be created except by a write.
    """
    try:
        return client.get_collection(name).count()
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
        if _is_unverified(meta):
            label += " unverified"  # inside the bracket, after the id; active notes unchanged
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


def _log_event(project: str, event: str, detail: str = "",
               namespace: str = LEGACY_NAMESPACE) -> None:
    conn = _ledger()
    with conn:
        conn.execute(
            "INSERT INTO events (ts, project, event, detail) VALUES (?,?,?,?)",
            (time.time(), _scope(project, namespace), event, detail[:500]),
        )
    conn.close()


# ── tools ─────────────────────────────────────────────────────────────────
# What each tool does to the store. A "write" tool calls _caller(write=True)
# before anything else, so a read-only token is refused with no side effect.
# "read" tools change no node and create no collection (recall and stats do add a
# row to the usage ledger, which is what stats' suggestions are counted from).
# Every registered tool must be listed; tests/test_auth.py fails otherwise.
TOOL_ACCESS = {
    "recall": "read",
    "stats": "read",
    "list_projects": "read", # names and counts of the caller's projects
    "learn": "write",        # adds a note, may supersede or archive others
    "retire": "write",       # changes a note's status
    "verify": "write",       # promotes an unverified note (also needs the verify right)
    "ingest_code": "write",  # deletes and rebuilds the code layer (also needs ingest)
    "ingest_wiki": "write",  # deletes and rebuilds the wiki layer (also needs ingest)
    "reindex_wiki": "write", # updates the wiki layer from a folder (also needs ingest)
}


ALL_PROJECTS = "*"


def _unverified_footer(count: int) -> str:
    return (f"({count} unverified note(s): written by an agent, not yet confirmed; "
            f"treat as a lead and verify before relying on it)")


def _namespace_projects(client, namespace: str) -> dict[str, dict[str, str]]:
    """{project slug: {layer: collection name}} for one namespace, read from the
    collection names alone. A slug never holds ".", so `rag_<ns>.<slug>__<layer>`
    cannot be forged by a project name, and a namespace never matches another's
    prefix; the legacy namespace is the names with no "." at all."""
    found: dict[str, dict[str, str]] = {}
    for collection in client.list_collections():
        match = re.fullmatch(r"rag_(.+)__(code|wiki|memory)", collection.name)
        if match is None:
            continue
        scope, layer = match.groups()
        if namespace == LEGACY_NAMESPACE:
            slug = None if "." in scope else scope
        else:
            prefix = f"{namespace}."
            slug = scope[len(prefix):] if scope.startswith(prefix) else None
        if slug:
            found.setdefault(slug, {})[layer] = collection.name
    return found


def _recall_everywhere(namespace: str, query: str, layer: str,
                       top_wiki: int, top_code: int, top_memory: int) -> str:
    """recall(project="*"): each layer is searched in every project of the namespace
    and the best hits overall are kept, so the bound on output is the same as for one
    project. Every hit's tag names its project (`wiki@site`). The usage ledger is not
    written: its rows belong to one project each."""
    client = _client()
    embed = _embed_model()
    projects = _namespace_projects(client, namespace)
    lines = [f"Recall [{ALL_PROJECTS}] for: {query}"]
    unverified = 0
    for which, top in (("wiki", top_wiki), ("code", top_code), ("memory", top_memory)):
        if layer not in ("auto", which):
            continue
        targets = [(slug, layers[which]) for slug, layers in sorted(projects.items())
                   if which in layers and _count(client, layers[which])]
        if not targets:
            if which != "memory":
                lines.append(f"[{which}: not built for {ALL_PROJECTS}]")
            continue
        try:
            scored = []
            for slug, name in targets:
                for hit in _search_collection(client, name, query, top * 3, embed):
                    if _is_live(hit):
                        scored.append((hit.score if hit.score is not None else float("-inf"),
                                       slug, hit))
            scored.sort(key=lambda item: -item[0])  # stable: ties keep project order
            picked = scored[:top]
            unverified += sum(_is_unverified(hit) for _, _, hit in picked)
            rendered = [line for _, slug, hit in picked for line in
                        _render_hits([hit], f"{which}@{slug}", show_id=(which == "memory"))]
            lines += rendered or [f"[{which}: no results]"]
        except Exception as exc:  # noqa: BLE001 - retrieval never crashes a session
            lines.append(f"[{which} unavailable: {exc}]")
    if unverified:
        lines.append(_unverified_footer(unverified))
    return "\n".join(lines)


@mcp.tool()
def list_projects() -> str:
    """The projects of the caller's namespace and what each holds: node counts per
    layer and the number of unverified notes. Names are project slugs (lowercase,
    hyphenated), as `recall` labels them. Creates nothing."""
    ns = _caller().namespace
    client = _client()
    listing = []
    for slug, layers in sorted(_namespace_projects(client, ns).items()):
        entry = {"project": slug, "layers": {w: _count(client, layers[w])
                                              for w in ("wiki", "code", "memory") if w in layers},
                 "unverified_notes": 0}
        if "memory" in layers:
            try:
                metas = client.get_collection(layers["memory"]).get(include=["metadatas"])["metadatas"]
                entry["unverified_notes"] = sum(_is_unverified(m or {}) for m in metas or [])
            except Exception:  # noqa: BLE001
                pass
        listing.append(entry)
    return json.dumps({"namespace": ns, "projects": listing}, indent=2)


@mcp.tool()
def recall(query: str, project: str = "", layer: str = "auto",
           top_wiki: int = 2, top_code: int = 3, top_memory: int = 1) -> str:
    """Layered context recall for a project: wiki (map) → code (detail) →
    memory (prior session learnings). Bounded output; honest per-layer status.
    `project` defaults to whatever the projects file says the working directory
    belongs to, falling back to its name; `layer` = auto|wiki|code|memory.
    `project="*"` searches every project of the caller's namespace (and no other
    namespace) and labels each hit with its project: `[wiki@site index.md::Setup]`."""
    project = _resolve_project(project)
    ns = _caller().namespace
    if project == ALL_PROJECTS:
        return _recall_everywhere(ns, query, layer, top_wiki, top_code, top_memory)
    names = _collection_names(project, ns)
    client = _client()
    embed = _embed_model()

    lines = [f"Recall [{project}] for: {query}"]
    _log_event(project, "recall", query[:200], ns)
    unverified = 0

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
            unverified += sum(_is_unverified(h) for h in hits)
            rendered = _render_hits(hits, which, show_id=(which == "memory"))
            lines += rendered or [f"[{which}: no results]"]
        except Exception as exc:  # noqa: BLE001 - retrieval never crashes a session
            lines.append(f"[{which} unavailable: {exc}]")

    # Cross-project memory, after the project's own layers so a project hit always
    # comes first, and capped so it cannot crowd out project context. A project is
    # never silently read as the global one: this is a separate, labelled pass.
    if _wanted("memory") and _slugify(project) != GLOBAL_PROJECT:
        gname = _collection_names(GLOBAL_PROJECT, ns)["memory"]
        if _count(client, gname):
            try:
                ghits = [h for h in _search_collection(client, gname, query, top_memory * 3, embed)
                         if _is_live(h)][:top_memory]
                unverified += sum(_is_unverified(h) for h in ghits)
                lines += [f"[global] {line}" for line in
                          _render_hits(ghits, "memory", show_id=True)]
            except Exception as exc:  # noqa: BLE001
                lines.append(f"[global memory unavailable: {exc}]")
    if unverified:
        lines.append(_unverified_footer(unverified))
    return "\n".join(lines)


@mcp.tool()
def learn(text: str, kind: str = "learning", project: str = "",
          supersedes: str = "", status: str = "", origin: str = "") -> str:
    """Deposit ONE learning into the project's episodic memory (staging area).
    kind may carry a module prefix for consolidation routing, e.g.
    'rag:gotcha' or 'config:decision'. Learnings only — routine actions are
    logging, not learning, and belong in session transcripts.

    `supersedes` names an existing note (the short id recall prints, or a unique
    prefix of it) that this one replaces. The old note is marked as replaced and
    stops being returned, but is kept, so the change is reversible and you can
    still see what the earlier fact was. Use it instead of rewriting history.

    `status` is "" or "active" (the note is trusted, as always) or "unverified" (an
    agent's claim nobody has confirmed: recall returns it labelled, and `verify`
    promotes it). A token configured with learn_status "unverified" is held to that
    whatever it asks for. Learning text that already exists as an unverified note
    promotes it only for a caller that may verify and asks for "active" explicitly;
    an active note is never demoted. `origin` (80 characters) says who wrote it."""
    project = _resolve_project(project)
    caller = _caller(write=True)
    ns = caller.namespace
    if status not in ("", STATUS_ACTIVE, STATUS_UNVERIFIED):
        raise ValueError(f"status must be '', '{STATUS_ACTIVE}' or '{STATUS_UNVERIFIED}'")
    wanted = caller.learn_status or status or STATUS_ACTIVE
    origin = " ".join(origin.split())[:80]
    text = text.strip()
    if not text:
        raise ValueError("refusing to store an empty learning")
    if len(text) > 500:
        text = text[:497] + "…"

    names = _collection_names(project, ns)
    client = _client()
    col = client.get_or_create_collection(names["memory"])
    node_id = _det_id("mem", text)
    existing = col.get(ids=[node_id], include=["metadatas"])
    if existing["ids"]:
        # Idempotent: the same text is the same note and is not rewritten. The one
        # change allowed is the promotion of an unverified note, and only by a caller
        # that may verify and said "active" outright; nothing is ever demoted.
        meta = (existing["metadatas"] or [{}])[0] or {}
        if (meta.get("status") == STATUS_UNVERIFIED and status == STATUS_ACTIVE
                and wanted == STATUS_ACTIVE and _may_verify(caller)):
            _promote(col, node_id, meta, caller, note="")
            _log_event(project, "verify", node_id, ns)
            return f"{node_id} || promoted from unverified"
        _log_event(project, "learn_dup", namespace=ns)
        return node_id

    embed = _embed_model()
    _embedder_healthcheck(embed)
    vector = embed.get_text_embedding(text)
    metadata = _stamp({"path": "memory", "layer": "memory", "kind": kind, "status": wanted},
                      source="learn")
    if origin:
        metadata["origin"] = origin
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
            _log_event(project, "supersede", f"{target}->{node_id}", ns)
            note = f" || replaced {target[:8]}"

    # Backstop past the cap: the oldest notes are moved aside, never deleted. They
    # keep their text, stop being returned by recall, and remain countable in the
    # store — so the promise that nothing is lost holds even at the limit.
    if _staging_count(col) > MEMORY_CAP:
        all_meta = col.get(include=["metadatas"])
        live = [
            (pid, meta or {})
            for pid, meta in zip(all_meta["ids"], all_meta["metadatas"])
            if ((meta or {}).get("status") or STATUS_ACTIVE) in LIVE_STATUSES
            and (meta or {}).get("source") != MIRROR_SOURCE
        ]
        # Unverified notes go first, oldest first, so an agent that writes a lot
        # can only push out its own unconfirmed claims; active notes are archived
        # only for whatever excess is left after every unverified one has gone.
        live.sort(key=lambda pair: (pair[1].get("status") != STATUS_UNVERIFIED,
                                    str(pair[1].get("created", ""))))
        overflow = len(live) - MEMORY_CAP
        for pid, meta in live[:max(0, overflow)]:
            col.update(ids=[pid], metadatas=[{**meta, "status": STATUS_ARCHIVED}])
        if overflow > 0:
            _log_event(project, "archive", f"{overflow} of {len(live)} notes archived", ns)
            note += f" || {overflow} oldest note(s) archived (kept, no longer returned)"

    _log_event(project, "learn", kind, ns)

    staged = _staging_count(col)
    result = node_id + note
    if staged >= CONSOLIDATE_THRESHOLD and ns == LEGACY_NAMESPACE:  # the script knows no other
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
    ns = _caller(write=True).namespace
    client = _client()
    col = client.get_or_create_collection(_collection_names(project, ns)["memory"])
    target = _resolve_node_id(col, node_id.strip())
    if target is None:
        return (f"no single memory note matches {node_id!r} — nothing changed. "
                f"Call recall first and use the id it prints.")
    meta = (col.get(ids=[target], include=["metadatas"])["metadatas"] or [{}])[0] or {}
    if meta.get("status") in NON_LIVE_STATUSES:
        return f"{target[:8]} is already {meta.get('status')} — nothing changed"
    col.update(ids=[target], metadatas=[
        {**meta, "status": STATUS_RETIRED, "retired_reason": reason[:200]}])
    _log_event(project, "retire", f"{target} {reason[:100]}", ns)
    return f"retired {target[:8]}" + (f" ({reason[:80]})" if reason else "")


def _may_verify(caller: Caller) -> bool:
    """The token has the verify right, or the caller is the local owner (no token)."""
    return caller.verify or get_access_token() is None


def _promote(col, node_id: str, meta: dict, caller: Caller, note: str) -> None:
    promoted = {**meta, "status": STATUS_ACTIVE, "verified_at": _now(),
                "verified_by": _caller_label(caller)}
    if note.strip():
        promoted["verified_note"] = note.strip()[:200]
    col.update(ids=[node_id], metadatas=[promoted])


@mcp.tool()
def verify(node_id: str, project: str = "", note: str = "") -> str:
    """Confirm an unverified memory note, making it an ordinary active one.

    Only a token with the verify right may do this: it is the step that turns an
    agent's claim into something recall presents as established. `node_id` is the
    short id recall prints, or any unique prefix of it. The note records who
    confirmed it and when (and `note`, if given). An unknown or ambiguous reference,
    or a note that is not unverified, changes nothing and says so."""
    project = _resolve_project(project)
    caller = _caller(write=True)
    if not _may_verify(caller):
        raise PermissionError("this token may not run verify")
    ns = caller.namespace
    try:  # a refused or mistaken call creates nothing
        col = _client().get_collection(_collection_names(project, ns)["memory"])
    except Exception:  # noqa: BLE001 - no such layer
        return f"no memory notes in {project!r} — nothing changed"
    target = _resolve_node_id(col, node_id.strip())
    if target is None:
        return (f"no single memory note matches {node_id!r} — nothing changed. "
                f"Call recall first and use the id it prints.")
    meta = (col.get(ids=[target], include=["metadatas"])["metadatas"] or [{}])[0] or {}
    status = meta.get("status") or STATUS_ACTIVE
    if status != STATUS_UNVERIFIED:
        return f"{target[:8]} is {status}, not unverified — nothing changed"
    _promote(col, target, meta, caller, note)
    _log_event(project, "verify", target, ns)
    return f"verified {target[:8]}"


@mcp.tool()
def stats(project: str = "") -> str:
    """Layer counts + wiki-earn suggestion for a project. The suggestion rule:
    (≥3 sessions AND ≥5 learnings) OR (≥3 architecture-phrased recalls) on a
    project with no wiki → suggest generating the wiki."""
    project = _resolve_project(project)
    ns = _caller().namespace
    names = _collection_names(project, ns)
    client = _client()
    conn = _ledger()
    slug = _scope(project, ns)

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
    global_memory = _count(client, _collection_names(GLOBAL_PROJECT, ns)["memory"])
    has_wiki = counts["wiki"] > 0
    earned = (not has_wiki) and ((sessions >= 3 and learnings >= 5) or arch_recalls >= 3)

    # "memory" in layers counts every note ever stored, including replaced and
    # moved-aside ones. The number that drives merging is the live staging count.
    staged = unverified = 0
    try:  # never created by a read: a layer that does not exist has nothing staged
        memory = client.get_collection(names["memory"])
        staged = _staging_count(memory)
        unverified = sum(
            _is_unverified(meta or {})
            for meta in (memory.get(include=["metadatas"]) or {}).get("metadatas") or [])
    except Exception:  # noqa: BLE001
        pass

    report = {
        "project": project,
        "layers": counts,
        "staged_notes": staged,
        "unverified_notes": unverified,
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


# ── indexing ──────────────────────────────────────────────────────────────
# Shared by ingest_code, ingest_wiki and reindex_wiki. The order in all three is the
# same and deliberate: authorise, check the folder, look at the disk, and only then
# touch a collection. A refused or empty request must never cost an existing layer.

_CODE_SKIP_PARTS = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules",
    "dist", "build", "target", ".idea", ".vscode", "logs", ".hermes",
})
_CODE_SKIP_SUFFIXES = frozenset({
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".class", ".jar",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar",
    ".gz", ".whl", ".db", ".db-wal", ".db-shm", ".lock", ".log", ".min.js",
})


def _skipped_note(skipped: list) -> str:
    return f" ({len(skipped)} file(s) skipped: link out of the folder, or unreadable)" if skipped else ""


@mcp.tool()
def ingest_code(root: str, project: str = "", rebuild: bool = True) -> str:
    """Index a repo's source files into the project's code layer.
    Skips vendored/binary/hidden dirs; deterministic IDs make unchanged
    chunks idempotent; rebuild=True drops stale chunks of edited files.
    The folder must be one the token's namespace may index; links that lead
    out of it are skipped, never followed."""
    caller = _caller(write=True)
    _require_ingest(caller)
    ns = caller.namespace
    root_path = _check_index_path(ns, root)
    project = project or os.path.basename(root.rstrip("/"))
    names = _collection_names(project, ns)
    client = _client()
    embed = _embed_model()
    _embedder_healthcheck(embed)

    machine_skip = _machine_file_skip()
    in_protected_place = _walk_deny(root_path)

    def skip_file(rel: str) -> bool:
        return (any(part in _CODE_SKIP_PARTS for part in Path(rel).parts)
                or Path(rel).suffix.lower() in _CODE_SKIP_SUFFIXES
                or machine_skip(rel))

    files, skipped = fsafe.walk_files(
        str(root_path), skip_file=skip_file,
        prune_dir=lambda rel, real: (os.path.basename(rel) in _CODE_SKIP_PARTS
                                     or in_protected_place(rel, real)))

    from llama_index.core import Document
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.schema import TransformComponent
    from llama_index.vector_stores.chroma import ChromaVectorStore

    docs = []
    for f in files:
        try:
            text = fsafe.decode_text(fsafe.read_confined(f.real, str(root_path)))
        except OSError:
            skipped.append(fsafe.SkippedFile(f.rel, "unreadable"))
            continue
        if not text.strip():
            continue
        docs.append(Document(text=text[:200_000],
                             metadata=_stamp({"path": f.rel, "layer": "code"},
                                             source="ingest_code")))
    if not docs:
        return f"no indexable files under {root_path}"  # the existing layer is untouched

    if rebuild:
        try:
            client.delete_collection(names["code"])
        except Exception:
            pass

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
    _log_event(project, "ingest_code", f"{stored} nodes", ns)
    return f"code layer [{project}]: {stored} nodes from {len(docs)} files" + _skipped_note(skipped)


@mcp.tool()
def ingest_wiki(wiki_dir: str, project: str = "") -> str:
    """(Re)index a code-wiki directory (markdown) into the project's wiki
    layer. Heading-aware chunking keeps `##` sections atomic. Rebuilds the whole
    layer; `reindex_wiki` updates it in place. The folder must be one the token's
    namespace may index; links that lead out of it are skipped, never followed."""
    caller = _caller(write=True)
    _require_ingest(caller)
    ns = caller.namespace
    root = _check_index_path(ns, wiki_dir)
    project = project or os.path.basename(wiki_dir.rstrip("/"))
    names = _collection_names(project, ns)
    client = _client()
    embed = _embed_model()
    _embedder_healthcheck(embed)

    files, skipped = fsafe.walk_files(str(root), suffix=".md", prune_dir=_walk_deny(root))
    if not files:
        return f"no markdown under {root}"

    from llama_index.core import Document
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.schema import TransformComponent
    from llama_index.vector_stores.chroma import ChromaVectorStore

    docs = []
    for f in files:
        try:
            text = fsafe.decode_text(fsafe.read_confined(f.real, str(root)))
        except OSError:
            skipped.append(fsafe.SkippedFile(f.rel, "unreadable"))
            continue
        for idx, chunk in enumerate(_wiki_chunks(text)):
            docs.append(Document(
                text=chunk,
                metadata=_stamp(
                    {"path": f.rel, "layer": "wiki",
                     "section": "", "chunk_index": idx}, source="ingest_wiki"),
            ))
    if not docs:
        return f"no markdown under {root}"  # nothing to index: keep the existing layer

    try:
        client.delete_collection(names["wiki"])
    except Exception:
        pass

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
    _log_event(project, "ingest_wiki", f"{stored} nodes", ns)
    return f"wiki layer [{project}]: {stored} nodes" + _skipped_note(skipped)


# ── reindex_wiki: update the wiki layer in place ──────────────────────────
# `ingest_wiki` rebuilds the layer and gives its nodes random ids. This updates it:
# every node's id is derived from (folder, page, position, text), so a page that has
# not changed is recognised without embedding anything, an edited page replaces its
# own chunks, and a repeated or interrupted run converges on the same state. The
# index is its own record: each node carries the folder it came from (`root_id`) and
# `source` "reindex_wiki", and only nodes of that pair are ever touched, so pages
# indexed another way into the same layer are left alone.
#
# The work blocks (embedding, disk), so the tool is asynchronous and runs it in a
# worker thread; the event loop keeps answering other namespaces meanwhile.

REINDEX_SOURCE = "reindex_wiki"
_REINDEX_MODES = ("incremental", "full")
_MAX_BUDGET_SECONDS = 3600
_MAX_REPORTED_SKIPS = 20

_reindex_busy: set[tuple[str, str]] = set()
_reindex_guard = threading.Lock()
# One writer at a time into the store from these jobs, whatever project they serve.
_index_write_lock = threading.Lock()


@dataclass(frozen=True)
class _ReindexJob:
    namespace: str
    project: str
    root: Path
    mode: str
    prune: bool
    dry_run: bool
    exclude: tuple[str, ...]
    budget_seconds: float
    allow_empty: bool


def _reindex_prepare(wiki_dir: str, project: str, mode: str, prune: bool, dry_run: bool,
                     exclude: str, budget_seconds: float, allow_empty: bool) -> _ReindexJob:
    """Everything that can be refused is refused here, before any work or any write."""
    caller = _caller(write=True)
    _require_ingest(caller)
    if mode not in _REINDEX_MODES:
        raise ValueError(f"mode must be one of {', '.join(_REINDEX_MODES)}")
    if not 0 <= budget_seconds <= _MAX_BUDGET_SECONDS:
        raise ValueError(f"budget_seconds must be between 0 and {_MAX_BUDGET_SECONDS}")
    excluded = tuple(name.strip() for name in exclude.split(",") if name.strip())
    if any("/" in name or name in (".", "..") for name in excluded):
        raise ValueError("exclude takes top-level folder names, e.g. 'raw,drafts'")
    root = _check_index_path(caller.namespace, wiki_dir)
    if not root.is_dir():
        raise ValueError(f"{wiki_dir!r} is not a folder")
    name = os.path.basename(wiki_dir.rstrip("/"))
    project = project or (name if name not in ("", ".", "..") else root.name)
    return _ReindexJob(caller.namespace, project, root, mode, prune, dry_run, excluded,
                       float(budget_seconds), allow_empty)


def _run_reindex(job: _ReindexJob) -> dict:
    """Run a prepared job, unless one is already running for the same project."""
    key = (job.namespace, _slugify(job.project))
    with _reindex_guard:
        if key in _reindex_busy:
            return {"busy": True, "project": job.project}
        _reindex_busy.add(key)
    try:
        return _reindex(job)
    finally:
        with _reindex_guard:
            _reindex_busy.discard(key)


def _reindex(job: _ReindexJob) -> dict:
    started = time.monotonic()
    ns, root_real = job.namespace, str(job.root)
    root_id = hashlib.sha1(root_real.encode()).hexdigest()
    client = _client()
    name = _collection_names(job.project, ns)["wiki"]
    in_protected_place = _walk_deny(root_real)

    def prune_dir(rel: str, real: str) -> bool:
        return (os.path.basename(rel).startswith(".") or rel in job.exclude
                or in_protected_place(rel, real))

    def chunks_of(rel: str, data: bytes) -> list[tuple[str, int, str]]:
        """(node id, position, text) of every chunk of a page. The id is derived from
        the folder, the page, the position and the text, so it names the content."""
        return [(_det_id("wiki", root_id, rel, str(i), text), i, text)
                for i, text in enumerate(_wiki_chunks(fsafe.decode_text(data)))]

    # What is on disk: pages that can be read, and what had to be skipped and why.
    found, skipped = fsafe.walk_files(root_real, suffix=".md", prune_dir=prune_dir)
    wanted: dict[str, set[str]] = {}  # page -> ids its chunks get now
    for f in found:
        try:
            wanted[f.rel] = {node_id for node_id, _, _ in chunks_of(
                f.rel, fsafe.read_confined(f.real, root_real))}
        except OSError:
            skipped.append(fsafe.SkippedFile(f.rel, "unreadable"))
    real_of = {f.rel: f.real for f in found}
    pages = sorted(wanted, key=lambda rel: Path(rel).parts)

    # What the index holds for this folder: page -> {node id: metadata}.
    try:
        col = client.get_collection(name)
    except Exception:  # noqa: BLE001 - no wiki layer yet
        col = None
    held: dict[str, dict[str, dict]] = {}
    if col is not None:
        got = col.get(where={"$and": [{"source": REINDEX_SOURCE}, {"root_id": root_id}]},
                      include=["metadatas"])
        for node_id, meta in zip(got["ids"], got["metadatas"]):
            held.setdefault((meta or {}).get("path", ""), {})[node_id] = meta or {}

    # An unmounted or wrong folder looks exactly like "every page was deleted".
    if not pages and held and not (job.mode == "full" and job.prune and job.allow_empty):
        raise ValueError(
            f"refusing: no readable markdown under the folder, but the index holds "
            f"{sum(map(len, held.values()))} node(s) from it. Is it mounted? To empty "
            f"the index on purpose use mode='full', prune=true, allow_empty=true")

    # A page is current when the ids its chunks would get are exactly the ids held.
    changed = [rel for rel in pages if job.mode == "full" or wanted[rel] != set(held.get(rel, {}))]
    gone = [rel for rel in sorted(held) if rel not in wanted] if job.prune else []
    gone_ids = [node_id for rel in gone for node_id in held[rel]]

    added = deleted = indexed = removed = 0
    remaining = 0
    if job.dry_run:
        for rel in changed:
            added += len(wanted[rel] - set(held.get(rel, {})))
            deleted += len(set(held.get(rel, {})) - wanted[rel])
        indexed, removed, deleted = len(changed), len(gone), deleted + len(gone_ids)
    else:
        if gone_ids:
            with _index_write_lock:
                col.delete(ids=gone_ids)
            removed, deleted = len(gone), len(gone_ids)
        embed = None
        done = 0
        for rel in changed:
            # Incremental stops starting pages once the budget is spent and reports
            # what is left; at least one page is always done, so calling again
            # always makes progress. Full runs to completion: a partial full run
            # could not be resumed, because the next one would start over.
            if (job.mode == "incremental" and done
                    and time.monotonic() - started > job.budget_seconds):
                break
            done += 1
            try:
                data = fsafe.read_confined(real_of[rel], root_real)
            except OSError:
                skipped.append(fsafe.SkippedFile(rel, "unreadable"))
                continue
            sha = hashlib.sha256(data).hexdigest()
            chunks = chunks_of(rel, data)
            have = held.get(rel, {})
            fresh = [c for c in chunks if job.mode == "full" or c[0] not in have]
            if fresh:
                if embed is None:
                    embed = _embed_model()
                    _embedder_healthcheck(embed)
                try:
                    vectors = embed.get_text_embedding_batch([text for _, _, text in fresh])
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"embedding failed on {rel!r}: {exc}. {indexed} page(s) were indexed "
                        f"before it; this page's existing nodes are untouched") from exc
            current = {c[0] for c in chunks}
            fresh_ids = {c[0] for c in fresh}
            # Chunks that survive an edit keep their vectors; only the recorded hash
            # of the page moves on.
            kept = [k for k in current if k in have and k not in fresh_ids
                    and have[k].get("file_sha") != sha]
            stale = [k for k in have if k not in current]
            with _index_write_lock:
                if col is None:
                    col = client.get_or_create_collection(name)
                if fresh:
                    col.upsert(
                        ids=[node_id for node_id, _, _ in fresh],
                        embeddings=vectors,
                        documents=[text for _, _, text in fresh],
                        metadatas=[_stamp({"path": rel, "layer": "wiki", "section": "",
                                           "chunk_index": i, "root_id": root_id,
                                           "file_sha": sha}, source=REINDEX_SOURCE)
                                   for _, i, _ in fresh])
                if kept:
                    col.update(ids=kept, metadatas=[{**have[k], "file_sha": sha} for k in kept])
                if stale:
                    col.delete(ids=stale)  # only after the new vectors are stored
            added += len(current - set(have))
            deleted += len(stale)
            indexed += 1
        remaining = len(changed) - done
        if indexed or removed:
            _log_event(job.project, "reindex_wiki", f"+{added} -{deleted}", ns)

    return {
        "project": job.project,
        "mode": job.mode,
        "files": {"seen": len(found), "unchanged": len(pages) - len(changed),
                  "indexed": indexed, "removed": removed, "skipped": len(skipped)},
        "nodes": {"added": added, "deleted": deleted},
        "partial": remaining > 0,
        "remaining": remaining,
        "skipped": [{"path": s.rel, "reason": s.reason} for s in skipped[:_MAX_REPORTED_SKIPS]],
        "dry_run": job.dry_run,
    }


def reindex_wiki_blocking(wiki_dir: str, project: str = "", mode: str = "incremental",
                          prune: bool = True, dry_run: bool = False, exclude: str = "",
                          budget_seconds: float = 60, allow_empty: bool = False) -> str:
    """`reindex_wiki` for scripts and other in-process callers: same checks, same
    work, run on the calling thread. Returns the same JSON text."""
    job = _reindex_prepare(wiki_dir, project, mode, prune, dry_run, exclude,
                           budget_seconds, allow_empty)
    return json.dumps(_run_reindex(job), indent=2)


@mcp.tool()
async def reindex_wiki(wiki_dir: str, project: str = "", mode: str = "incremental",
                       prune: bool = True, dry_run: bool = False, exclude: str = "",
                       budget_seconds: float = 60, allow_empty: bool = False) -> str:
    """Bring the project's wiki layer up to date with a folder of markdown pages.

    Unlike `ingest_wiki` it does not rebuild: pages that have not changed cost a hash
    and no embedding, an edited page replaces its own chunks, and with `prune`
    (default) pages that are gone from the folder are removed from the index. Only
    nodes this tool wrote for this folder are touched. Pass a dedicated `project`.

    mode "incremental" (default) stops starting new pages after `budget_seconds` and
    answers `partial: true` with `remaining`; call again to continue. "full"
    re-embeds every page and runs to completion. `exclude` is a comma list of
    top-level subfolders to leave out (hidden folders are always left out).
    `dry_run` reports the plan and writes nothing. A folder with no pages is refused
    when the index holds pages from it (an unmounted disk must not empty the index);
    `allow_empty` with mode "full" and prune does it on purpose. Answers {"busy":
    true} while another run for the same project is in progress.
    Needs a token with the ingest right; the folder must be one its namespace may
    index. Returns JSON."""
    job = _reindex_prepare(wiki_dir, project, mode, prune, dry_run, exclude,
                           budget_seconds, allow_empty)
    return json.dumps(await anyio.to_thread.run_sync(_run_reindex, job), indent=2)


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> JSONResponse:
    """Liveness for a settings page or a supervisor: no auth, no data. The
    authenticated check that the store and the token work is `stats`."""
    return JSONResponse({"ok": True})


def _check_can_authenticate(transport: str) -> None:
    """Refuse to serve over a network transport when nobody could authenticate.

    Either the tokens file or the token directory must exist. A fresh install that
    has only the token directory (a provisioner will fill it) starts; one with
    neither does not. A file or directory that exists but is unfit stops the server
    rather than being served around.
    """
    directory = _tokens_dir()
    has_directory = os.path.lexists(directory)
    if has_directory:
        try:
            fsafe.check_private_dir(directory, what="token directory")
        except ValueError as exc:
            raise SystemExit(f"refusing to serve over {transport}: {exc}") from exc
    try:
        tokens = _read_tokens(TOKENS_FILE)
    except ValueError as exc:
        if os.path.lexists(TOKENS_FILE) or not has_directory:
            raise SystemExit(f"refusing to serve over {transport}: {exc}") from exc
        return
    if not tokens and not has_directory:
        raise SystemExit(f"refusing to serve over {transport}: no tokens in {TOKENS_FILE}")


def main() -> None:
    # stdio by default; HERMES_RAG_TRANSPORT=streamable-http runs ONE shared server
    # (single writer to the Chroma store) that every agent connects to over HTTP.
    transport = os.environ.get("HERMES_RAG_TRANSPORT", "stdio")
    if transport != "stdio":
        # Fail closed: a shared server with nobody able to authenticate is a
        # misconfiguration to stop on, not to serve through.
        _check_can_authenticate(transport)
        mcp.settings.host = os.environ.get("HERMES_RAG_HOST", "127.0.0.1")
        mcp.settings.port = int(os.environ.get("HERMES_RAG_PORT", "8765"))
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
