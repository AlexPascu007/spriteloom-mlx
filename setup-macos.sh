#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  printf '%s\n' 'setup-macos.sh must run on macOS.' >&2
  exit 1
fi
case "$(uname -m)" in
  arm64|aarch64) ;;
  *)
    printf '%s\n' 'Spriteloom MLX setup needs Apple silicon (arm64).' >&2
    exit 1
    ;;
esac

ROOT="$SCRIPT_DIR"
VENV="$ROOT/.venv"

uv_bin="${UV_BIN:-}"
if [[ -z "$uv_bin" ]]; then
  for candidate in /opt/homebrew/bin/uv /usr/local/bin/uv; do
    if [[ -x "$candidate" ]]; then
      uv_bin="$candidate"
      break
    fi
  done
fi
if [[ -z "$uv_bin" ]] && command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  if [[ -n "$uv_bin" ]]; then
    printf '%s\n' "Creating .venv with Python 3.12 through $uv_bin"
    "$uv_bin" venv --python 3.12 "$VENV"
  else
    python_bin="${SPRITELOOM_PYTHON:-}"
    if [[ -z "$python_bin" ]]; then
      for candidate in /opt/homebrew/bin/python3.12 \
                      /usr/local/bin/python3.12 python3.12 \
                      /opt/homebrew/bin/python3.11 \
                      /usr/local/bin/python3.11 python3.11; do
        if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then
          python_bin="$candidate"
          break
        fi
      done
    fi
    if [[ -z "$python_bin" ]]; then
      printf '%s\n' 'Install Python 3.12 or uv, then run this script again.' >&2
      exit 1
    fi
    printf '%s\n' "Creating .venv with $python_bin"
    "$python_bin" -m venv "$VENV"
  fi
fi

PYTHON="$VENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' "The virtual environment has no executable at $PYTHON." >&2
  exit 1
fi
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  "$PYTHON" -m ensurepip --upgrade
fi

printf '%s\n' 'Installing launcher and MLX dependencies...'
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --no-compile -r "$ROOT/launcher/requirements.txt"
"$PYTHON" -m pip install --no-compile -r "$ROOT/server/requirements-macos.txt"

MODEL_ROOT="$ROOT/models"
HF_HUB_DISABLE_XET=1 SPRITELOOM_MODELS_DIR="$MODEL_ROOT" "$PYTHON" -u - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

model_root = Path(os.environ["SPRITELOOM_MODELS_DIR"])
model_folder = model_root / "models--black-forest-labs--FLUX.2-klein-4B"
marker = model_folder / ".complete"
if marker.is_file():
    print(f"Model already downloaded: {model_folder}", flush=True)
else:
    model_root.mkdir(parents=True, exist_ok=True)
    print("Downloading the original BFL FLUX.2 Klein 4B diffusers layout.",
          flush=True)
    snapshot_download(
        "black-forest-labs/FLUX.2-klein-4B",
        cache_dir=str(model_root),
        local_files_only=False,
        ignore_patterns=["flux-2-klein-4b.safetensors"],
    )
    marker.write_text("ok", encoding="utf-8")
    print(f"Model download complete: {model_folder}", flush=True)
PY

SPRITELOOM_PORT="${SPRITELOOM_PORT:-8765}" \
  "$ROOT/install-plugin.sh"

printf '%s\n' 'macOS setup complete. Start the launcher with:'
printf '  %q -m launcher.app\n' "$PYTHON"
