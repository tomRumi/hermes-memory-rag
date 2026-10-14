#!/bin/bash
# Shared runtime for the installed no-agent jobs; source after setting REPO.
# Dependencies live in uv's per-interpreter cache, not in the selected Python.
unset PYTHONPATH PYTHONHOME

prepare_python_runtime() {
  local choice="${HERMES_RAG_PYTHON:-python3}" version
  SELECTED_PYTHON="$(command -v "$choice" 2>/dev/null || true)"
  if [ -z "$SELECTED_PYTHON" ] || [ ! -x "$SELECTED_PYTHON" ]; then
    echo "Memory job: Python interpreter '$choice' not found (set HERMES_RAG_PYTHON)." >&2
    return 2
  fi
  version="$("$SELECTED_PYTHON" -I -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>&1)" || {
    echo "Memory job: cannot run interpreter '$choice': $version" >&2
    return 2
  }
  if ! "$SELECTED_PYTHON" -I -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "Memory job: Python 3.11+ required; '$choice' reports $version." >&2
    return 2
  fi
  RUNTIME_UV="${HERMES_RAG_UV:-$(command -v uv 2>/dev/null || printf '%s/.hermes/bin/uv' "$HOME")}"
  if [ ! -x "$RUNTIME_UV" ]; then
    echo "Memory job: uv not found at '$RUNTIME_UV' (set HERMES_RAG_UV)." >&2
    return 2
  fi
}

run_memory_python() {
  local dependency="$1" script="$2"
  shift 2
  "$RUNTIME_UV" run --no-project --no-config --no-python-downloads \
    --python "$SELECTED_PYTHON" --with "$dependency" python -I "$script" "$@"
}
