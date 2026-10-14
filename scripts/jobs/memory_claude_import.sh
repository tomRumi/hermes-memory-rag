#!/bin/bash
# Daily import of Claude Code's memory files, then a re-index of the pages.
#
# Registered as a Hermes cron job with no_agent=true: this script IS the job.
# The import is idempotent, so running it daily costs nothing when nothing has
# changed — and it stays silent then, because a daily "nothing to do" message is
# noise. It speaks up when pages or facts were added, or when it fails.
#
# The selected interpreter runs with uv-managed, isolated dependencies for each
# step. Copying pages without re-indexing leaves files that nothing can find.

set -uo pipefail
REPO="__REPO__"
source "$REPO/scripts/jobs/python_runtime.sh"
LOG_DIR="$HOME/.hermes/memory-backup"
PENDING="$LOG_DIR/claude-reindex.pending"

if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--reindex-only" ]; }; then
  echo "Usage: $0 [--reindex-only]" >&2
  exit 2
fi
prepare_python_runtime || exit $?

mkdir -p "$LOG_DIR"
ADDED=0 UPDATED=0 FACTS=0
if [ "${1:-}" != "--reindex-only" ]; then
  OUT="$(run_memory_python numpy "$REPO/scripts/claude_memory_import.py" --apply 2>&1)"
  STATUS=$?
  {
    echo "=== $(date '+%Y-%m-%d %H:%M') ==="
    echo "$OUT"
  } >> "$LOG_DIR/claude-import.log"

  if [ $STATUS -ne 0 ]; then
    echo "Claude Code memory import FAILED (exit $STATUS):"
    echo "$OUT" | tail -15
    exit $STATUS
  fi

  ADDED="$(echo "$OUT" | grep -o 'wiki pages added: *[0-9]*' | grep -o '[0-9]*$' | head -1)"
  UPDATED="$(echo "$OUT" | grep -o 'wiki pages updated: *[0-9]*' | grep -o '[0-9]*$' | head -1)"
  FACTS="$(echo "$OUT" | grep -o 'facts added: *[0-9]*' | grep -o '[0-9]*$' | head -1)"

  # Keep a durable reminder if the indexer fails after the importer wrote pages.
  if [ "${ADDED:-0}" -gt 0 ] || [ "${UPDATED:-0}" -gt 0 ]; then
    touch "$PENDING" || exit 1
  fi
  if [ "${ADDED:-0}" -eq 0 ] && [ "${UPDATED:-0}" -eq 0 ] && \
     [ "${FACTS:-0}" -eq 0 ] && [ ! -e "$PENDING" ]; then
    exit 0
  fi
fi

REINDEX="$(run_memory_python "${HERMES_RAG_SERVER_DEP:-$REPO}" "$REPO/scripts/reindex_wiki.py" 2>&1)"
RSTATUS=$?
echo "$REINDEX" >> "$LOG_DIR/claude-import.log"

if [ $RSTATUS -eq 0 ] && [ -e "$PENDING" ] && \
   ! echo "$REINDEX" | grep -Eq 'page\(s\) ->'; then
  REINDEX="$REINDEX
No indexed pages reported; leaving re-index pending."
  RSTATUS=1
fi
if [ $RSTATUS -ne 0 ]; then
  if [ "${1:-}" = "--reindex-only" ]; then
    echo "Claude Code memory: recovery re-index of existing pages failed."
  else
    echo "Claude Code memory: $ADDED new page(s), $UPDATED updated, $FACTS fact(s)."
  fi
  echo "Re-indexing FAILED — the pages are on disk but not searchable:"
  echo "$REINDEX" | tail -10
  exit $RSTATUS
fi

if [ -e "$PENDING" ]; then
  rm "$PENDING" || exit 1
fi

if [ "${1:-}" = "--reindex-only" ]; then
  echo "Claude Code memory: re-indexed existing wiki pages (import skipped). Indexed:"
else
  echo "Claude Code memory: $ADDED new page(s), $UPDATED updated, $FACTS fact(s). Indexed:"
fi
echo "$REINDEX" | grep -E 'page\(s\) ->' | sed 's/^/  /'
exit 0