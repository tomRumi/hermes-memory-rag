#!/bin/bash
# Daily import of Claude Code's memory files, then a re-index of the pages.
#
# Registered as a Hermes cron job with no_agent=true: this script IS the job.
# The import is idempotent, so running it daily costs nothing when nothing has
# changed — and it stays silent then, because a daily "nothing to do" message is
# noise. It speaks up when pages or facts were added, or when it fails.
#
# Two interpreters on purpose: the importer touches the fact store, which lives
# inside Hermes, so it runs with Hermes's Python. Re-indexing touches the vector
# library, which deliberately does NOT live inside Hermes, so it runs with the
# server's own Python. Copying pages without re-indexing them leaves files that
# nothing can find, which is why the second step is here and not left to memory.

set -uo pipefail

REPO="__REPO__"
HERMES_PY="__HERMES_PY__"
SERVER_PY="__SERVER_PY__"
LOG_DIR="$HOME/.hermes/memory-backup"

OUT="$("$HERMES_PY" "$REPO/scripts/claude_memory_import.py" --apply 2>&1)"
STATUS=$?

mkdir -p "$LOG_DIR"
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

if [ "${ADDED:-0}" -eq 0 ] && [ "${UPDATED:-0}" -eq 0 ] && [ "${FACTS:-0}" -eq 0 ]; then
  # Nothing changed, so nothing needs re-indexing either.
  exit 0
fi

REINDEX="$("$SERVER_PY" "$REPO/scripts/reindex_wiki.py" 2>&1)"
RSTATUS=$?
echo "$REINDEX" >> "$LOG_DIR/claude-import.log"

if [ $RSTATUS -ne 0 ]; then
  echo "Claude Code memory: $ADDED new page(s), $UPDATED updated, $FACTS fact(s)."
  echo "Re-indexing FAILED — the pages are on disk but not searchable:"
  echo "$REINDEX" | tail -10
  exit $RSTATUS
fi

echo "Claude Code memory: $ADDED new page(s), $UPDATED updated, $FACTS fact(s). Indexed:"
echo "$REINDEX" | grep -E 'page\(s\) ->' | sed 's/^/  /'
exit 0