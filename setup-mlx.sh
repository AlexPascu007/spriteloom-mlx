#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  printf '%s\n' 'This setup requires macOS on Apple Silicon.' >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
  elif command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    printf '%s\n' 'Install uv or Python 3.12, then run this script again.' >&2
    exit 1
  fi
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r server/requirements-macos.txt
else
  .venv/bin/python -m ensurepip --upgrade
  .venv/bin/python -m pip install -r server/requirements-macos.txt
fi
.venv/bin/python -u - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    "black-forest-labs/FLUX.2-klein-4B",
    revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
    cache_dir="models",
    ignore_patterns=["flux-2-klein-4b.safetensors"],
)
from server.instruct_mlx import MLXKleinPipeline
MLXKleinPipeline()._resolve_model_path()
(Path("models") / "models--black-forest-labs--FLUX.2-klein-4B" / ".complete").write_text("ok")
print("MLX engine setup complete. Run ./start-server.sh or configure spriteloom-mcp.")
PY
