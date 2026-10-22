#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' "Missing $PYTHON. Run ./setup-macos.sh first." >&2
  exit 1
fi

exec "$PYTHON" -u -m server.main "$@"
