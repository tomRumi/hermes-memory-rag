#!/bin/bash
# Daily memory backup.
#
# Registered as a Hermes cron job with no_agent=true: this script IS the job, and
# whatever it prints is delivered verbatim. So it stays quiet when the backup
# worked and nothing changed, and prints when something needs attention — a daily
# "all good" message would train the reader to ignore it.

set -uo pipefail

REPO="__REPO__"
PY="__SERVER_PY__"
BACKUP_DIR="$HOME/.hermes/memory-backup"
STORE="$HOME/hermes-rag"

OUT="$("$PY" "$REPO/scripts/backup.py" --dir "$BACKUP_DIR" --store "$STORE" 2>&1)"
STATUS=$?

if [ $STATUS -ne 0 ]; then
  echo "Memory backup FAILED (exit $STATUS):"
  echo "$OUT" | tail -15
  exit $STATUS
fi

# Report only when something was actually written.
COMMITTED="$(echo "$OUT" | grep -o 'committed [0-9]*' | grep -o '[0-9]*' | head -1)"
if [ "${COMMITTED:-0}" -gt 0 ]; then
  echo "Memory backup: $COMMITTED path(s) changed. $(echo "$OUT" | grep '^store:' ) $(echo "$OUT" | grep '^facts:')"
fi

# The run's own record is always written, so keep the log regardless.
echo "$OUT" >> "$BACKUP_DIR/backup.log" 2>/dev/null
exit 0