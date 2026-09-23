#!/bin/bash
# Weekly memory review.
#
# Registered as a Hermes cron job with no_agent=true: this script IS the job, and
# its output is delivered as-is. Unlike the daily jobs this one SPEAKS EVERY TIME:
# the point is to bring accumulating notes, near-duplicate facts and names that no
# longer exist in front of a human, and a report that only appears when it is
# worried is a report nobody reads.
#
# It merges nothing: the merge step rewrites pages and stays a deliberate action,
# either from a session or from the desktop window.

set -uo pipefail

REPO="__REPO__"
PY="__SERVER_PY__"
BACKUP_DIR="$HOME/.hermes/memory-backup"
STORE="$HOME/hermes-rag"

OUT="$("$PY" "$REPO/scripts/weekly_review.py" --no-apply \
        --store "$STORE" --out "$BACKUP_DIR/review.md" 2>&1)"
STATUS=$?

if [ $STATUS -ne 0 ]; then
  echo "Memory review FAILED (exit $STATUS):"
  echo "$OUT" | tail -15
  exit $STATUS
fi

# The summary is the report minus the wall of per-pair detail.
echo "$OUT" | sed -n '/^# Memory review/,/^## Notes worth reading/p' | head -30
echo
echo "Full report: $BACKUP_DIR/review.md"
exit 0