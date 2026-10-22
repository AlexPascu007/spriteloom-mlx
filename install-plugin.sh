#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

ASEPRITE_DIR="${ASEPRITE_USER_FOLDER:-$HOME/Library/Application Support/Aseprite}"
DEST="$ASEPRITE_DIR/extensions/spriteloom"
PORT="${SPRITELOOM_PORT:-8765}"
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  printf '%s\n' 'SPRITELOOM_PORT must be an integer between 1024 and 65535.' >&2
  exit 1
fi
if [[ ! -d "$SCRIPT_DIR/plugin" ]]; then
  printf '%s\n' "Plugin source not found: $SCRIPT_DIR/plugin" >&2
  exit 1
fi

mkdir -p "$DEST"
for source in "$SCRIPT_DIR/plugin"/*; do
  [[ -f "$source" ]] || continue
  cp -f "$source" "$DEST/$(basename "$source")"
done
printf '{\n  "port": %s\n}\n' "$PORT" > "$DEST/server.json"
printf 'Installed the Aseprite plugin to %s. Restart Aseprite.\n' "$DEST"
