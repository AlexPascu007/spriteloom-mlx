#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  printf '%s\n' 'build-macos.sh must run on macOS.' >&2
  exit 1
fi
case "$(uname -m)" in
  arm64|aarch64) ;;
  *)
    printf '%s\n' 'The macOS app recipe needs an Apple-silicon build host.' >&2
    exit 1
    ;;
esac

ROOT="$SCRIPT_DIR"
PYTHON="${SPRITELOOM_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' "Missing $PYTHON. Run ./setup-macos.sh first." >&2
  exit 1
fi

if ! "$PYTHON" -c 'import PyInstaller' >/dev/null 2>&1; then
  printf '%s\n' 'PyInstaller is missing from the selected .venv.' >&2
  printf '%s\n' 'Install it with: uv pip install --python .venv/bin/python pyinstaller' >&2
  exit 1
fi

printf '%s\n' 'Building Spriteloom.app for arm64 (ad-hoc local signature)...'
"$PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --distpath "$ROOT" \
  --workpath "$ROOT/build" \
  "$ROOT/build-macos.spec"

APP="$ROOT/Spriteloom.app"
EXECUTABLE="$APP/Contents/MacOS/Spriteloom"
if [[ ! -x "$EXECUTABLE" ]]; then
  printf '%s\n' "Build did not produce $EXECUTABLE." >&2
  exit 1
fi

printf '%s\n' "Built $APP"
file "$EXECUTABLE"
