"""Filesystem helpers that never follow a link out of where they were pointed.

Two jobs, both about not trusting a path that a stranger can influence:

- `read_private_json` opens a configuration file the way the tokens file has always
  been opened: without following a symlink, judging the file actually held, and
  requiring the file and its directory to belong to the server user and be closed to
  everyone else. Whoever can replace the file or its directory decides what it says.
- `walk_files` / `read_confined` list and read the files under a directory root for
  indexing. A symlink inside the root is never a way to read something outside it:
  directory links are not entered, a file link is followed only if its real target
  lies inside the root, and the read is made through a descriptor that refuses a
  link at the last step.

Nothing here knows about namespaces, tokens or the store: callers pass what to
protect. Everything raises `OSError` / `ValueError` and decides for itself what that
means (the token loader fails closed, the indexer skips the file and says so).
"""
from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def check_private_dir(path: Path, *, what: str) -> None:
    """The directory must be a real one (not a symlink), ours, and closed to others.

    Raises ValueError otherwise. `what` names the directory, for the message.
    """
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"cannot read {what} {path}: {exc}") from exc
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
        raise ValueError(
            f"{what} {path} must be a real directory (not a symlink) "
            f"owned by the server user, mode 700")


def read_private_json(path: Path, *, what: str) -> object:
    """Parsed JSON of a private file; ValueError if it cannot be trusted or read.

    The file is opened without following a symlink and judged through the open
    descriptor, so what is checked is what is read. It must be a regular file owned
    by the server user with no group or other access, and its directory must pass
    `check_private_dir`.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            st = os.fstat(fd)
            with os.fdopen(fd, encoding="utf-8") as fh:
                fd = -1
                raw = json.load(fh)
        finally:
            if fd >= 0:
                os.close(fd)
        parent = os.lstat(path.parent)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {what} {path}: {exc}") from exc
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
        raise ValueError(f"{what} {path} must be a regular file owned by the server user, mode 600")
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.geteuid() or parent.st_mode & 0o077:
        raise ValueError(
            f"directory of {what} {path} must be a real directory (not a symlink) "
            f"owned by the server user, mode 700")
    return raw


# ── indexing: walk and read without leaving the root ──────────────────────


@dataclass(frozen=True)
class WalkedFile:
    rel: str    # path relative to the root, as stored in node metadata
    real: str   # the file's real path, proven to lie inside the root


@dataclass(frozen=True)
class SkippedFile:
    rel: str
    reason: str


def is_inside(path: str, root: str) -> bool:
    """True if `path` is `root` or below it. Compares whole path components, so
    /a/bc is not under /a/b; both arguments must already be real paths."""
    return Path(path).is_relative_to(root)


def walk_files(
    root_real: str,
    *,
    suffix: str | None = None,
    prune_dir: Callable[[str, str], bool] = lambda rel, real: False,
    skip_file: Callable[[str], bool] = lambda rel: False,
) -> tuple[list[WalkedFile], list[SkippedFile]]:
    """The files under `root_real` that can be read safely, plus those that cannot.

    `root_real` must be a real path (`os.path.realpath`). The walk never follows a
    directory link; a directory link is reported as skipped. A file is returned only
    if its real path lies inside the root; a link that leaves it is reported as
    skipped, never read. `suffix` keeps files whose name ends with it; `skip_file`
    (given the relative path) and `prune_dir` (given the relative and the real path of
    a directory) drop what the caller does not want, silently, since those are the
    caller's own filters and not something the operator needs to hear about.
    Results are in a stable order (by path components).
    """
    found: list[WalkedFile] = []
    skipped: list[SkippedFile] = []
    for dirpath, dirnames, filenames in os.walk(root_real, followlinks=False):
        keep = []
        for name in sorted(dirnames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root_real)
            if os.path.islink(full):
                skipped.append(SkippedFile(rel, "symlinked directory"))
            elif not prune_dir(rel, full):
                keep.append(name)
        dirnames[:] = keep
        for name in filenames:
            if suffix is not None and not name.endswith(suffix):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root_real)
            if skip_file(rel):
                continue
            real = os.path.realpath(full)
            if not is_inside(real, root_real):
                skipped.append(SkippedFile(rel, "escapes folder"))
                continue
            found.append(WalkedFile(rel, real))
    found.sort(key=lambda f: Path(f.rel).parts)
    skipped.sort(key=lambda f: Path(f.rel).parts)
    return found, skipped


def read_confined(real: str, root_real: str) -> bytes:
    """Read a file proven to be inside the root, through a descriptor.

    The path is opened without following a link at the last step and the opened
    object must be a regular file, so a file swapped for a link (or a pipe) between
    the walk and this read fails with OSError instead of being read. The real path
    is checked against the root again here: this function is safe to call on its own.
    """
    if not is_inside(real, root_real):
        raise OSError(f"{real} is outside {root_real}")
    # O_NONBLOCK keeps the open of a swapped-in FIFO from hanging; a regular file is
    # unaffected by it.
    fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"{real} is not a regular file")
        with os.fdopen(fd, "rb") as fh:
            fd = -1
            return fh.read()
    finally:
        if fd >= 0:
            os.close(fd)


def decode_text(data: bytes) -> str:
    """UTF-8 text with undecodable bytes replaced and newlines normalised, exactly
    as `Path.read_text(errors="replace")` produced them before reading was confined."""
    text = data.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")

