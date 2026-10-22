"""Settings shared by the launcher and the server, in one json file."""
import json
import os
import pathlib
import platform
import sys

DEFAULT_PORT = 8765
HOST = "127.0.0.1"

VRAM_MODES = ("auto", "bf16", "offload")  # fp8 retired; a saved "fp8" -> auto
DEFAULT_VRAM_MODE = "auto"

SETTING_KEYS = ("port", "vram_mode", "root", "python", "aseprite_dir",
                "models_dir", "backend", "mlx_quantize")
DEFAULT_MODELS_DIR = "models"


def config_path() -> pathlib.Path:
    if sys.platform == "darwin":
        return (pathlib.Path.home() / "Library" / "Application Support"
                / "Spriteloom" / "config.json")
    base = os.environ.get("APPDATA")
    root = pathlib.Path(base) if base else pathlib.Path.home()
    return root / "Spriteloom" / "config.json"


def load_port(path: pathlib.Path | None = None) -> int:
    target = path or config_path()
    try:
        value = json.loads(target.read_text(encoding="utf-8"))["port"]
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_PORT
    if isinstance(value, int) and not isinstance(value, bool) \
            and 1024 <= value <= 65535:
        return value
    return DEFAULT_PORT


def load_vram_mode(path: pathlib.Path | None = None) -> str:
    target = path or config_path()
    try:
        value = json.loads(target.read_text(encoding="utf-8"))["vram_mode"]
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_VRAM_MODE
    if value in VRAM_MODES:
        return value
    return DEFAULT_VRAM_MODE


def _read(target: pathlib.Path) -> dict:
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_settings(path: pathlib.Path | None = None) -> dict:
    target = path or config_path()
    data = _read(target)
    out = {key: data.get(key) for key in SETTING_KEYS}
    out["port"] = load_port(target)
    out["vram_mode"] = load_vram_mode(target)
    out["backend"] = load_backend(target)
    out["mlx_quantize"] = load_mlx_quantize(target)
    return out


def load_backend(path: pathlib.Path | None = None) -> str:
    value = os.environ.get("SPRITELOOM_BACKEND") or _read(
        path or config_path()).get("backend", "auto")
    if value not in ("auto", "cuda", "mlx"):
        raise ValueError("backend must be auto, cuda, or mlx")
    if value == "auto":
        return ("mlx" if sys.platform == "darwin" and
                platform.machine().lower() in ("arm64", "aarch64") else "cuda")
    return value


def load_mlx_quantize(path: pathlib.Path | None = None) -> int | None:
    value = _read(path or config_path()).get("mlx_quantize", 8)
    if value is None:
        return None
    if type(value) is not int or value not in (3, 4, 5, 6, 8):
        raise ValueError("mlx_quantize must be 3, 4, 5, 6, 8, or null")
    return value


def save_settings(values: dict, path: pathlib.Path | None = None) -> None:
    # merge, never replace: the file is shared with settings we do not own
    target = path or config_path()
    data = _read(target)
    data.update(values)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_models_dir(path: pathlib.Path | None = None) -> str:
    return load_settings(path).get("models_dir") or DEFAULT_MODELS_DIR


def save_port(port: int, path: pathlib.Path | None = None) -> None:
    save_settings({"port": port}, path)


def save_vram_mode(mode: str, path: pathlib.Path | None = None) -> None:
    if mode not in VRAM_MODES:
        return
    save_settings({"vram_mode": mode}, path)
