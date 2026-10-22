"""What is already in place. Detection only, never a side effect."""
import os
import pathlib
import shutil
import platform
import sys

from launcher import plugin_install
from launcher.paths import MIN_PYTHON, run_command
from launcher.server_proc import venv_python

OK = "ok"
MISSING = "missing"
BLOCKED = "blocked"

MODEL_FOLDER = "models--black-forest-labs--FLUX.2-klein-4B"
# a snapshot dir appears from the first file downloaded, long before it's done
MODEL_MARKER = ".complete"
MODEL_GB = 15
# Dependency metadata is checked without importing the heavyweight server.
_COMMON_DEPS = ("'websockets','Pillow','numpy','scipy','safetensors',"
                "'huggingface-hub','tqdm'")
_WINDOWS_DEPS = ("'diffusers','transformers','accelerate','peft',"
                 "'bitsandbytes'")
_MACOS_DEPS = "'mlx','mflux'"
TORCH_PROBE = "import importlib.metadata as m; print(m.version('torch'))"
MLX_PROBE = "import mlx; print(getattr(mlx, '__version__', 'installed'))"


def backend_name(requested: str | None = None, system: str | None = None,
                 machine: str | None = None) -> str:
    """Resolve the setup backend using the same automatic platform rule.

    The server owns the persisted backend setting.  Import it when that API
    exists, while retaining compatibility with the pre-MLX configuration in
    existing Windows checkouts.
    """
    if requested is None:
        try:
            from server import config
            loader = getattr(config, "load_backend", None)
            requested = loader() if loader else "auto"
        except (ImportError, OSError, TypeError, ValueError):
            requested = "auto"
    requested = str(requested or "auto").lower()
    if requested in ("mlx", "cuda"):
        return requested
    system = system or sys.platform
    machine = machine or platform.machine()
    if system == "darwin" and machine.lower() in ("arm64", "aarch64"):
        return "mlx"
    return "cuda"


def backend_label(backend: str | None = None) -> str:
    return "MLX" if (backend or backend_name()) == "mlx" \
        else "PyTorch with CUDA"


def deps_probe(backend: str | None = None) -> str:
    """Build a metadata-only dependency probe for the active backend."""
    backend = backend or backend_name()
    deps = _MACOS_DEPS if backend == "mlx" else _WINDOWS_DEPS
    return ("import importlib.metadata as m; "
            f"[m.version(d) for d in ({_COMMON_DEPS},{deps})]; "
            "print('ok')")


# Kept as a public compatibility constant for callers that used the old
# Windows-only dependency probe.  Its package list follows the active backend.
DEPS_PROBE = deps_probe()


def _cuda_tag(version):
    # torch's local version tag carries the CUDA build, e.g. 2.4.1+cu124 -> 12.4
    if not version or "+cu" not in version:
        return None
    digits = "".join(c for c in version.split("+cu", 1)[1] if c.isdigit())
    return f"{digits[:-1]}.{digits[-1]}" if len(digits) >= 2 else None


def _item(item_id, label, state, detail, required=True, needs=()):
    return {"id": item_id, "label": label, "state": state, "detail": detail,
            "required": required, "needs": list(needs)}


def _free_gb(folder) -> int:
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free // 2**30
    except OSError:
        return 0


def _plugin_item(paths) -> dict:
    dest = (plugin_install.dest_in(paths.aseprite_dir)
            if paths.aseprite_dir else None)
    info = plugin_install.status(plugin_install.source_dir(), dest)
    if info["state"] == "current":
        return _item("plugin", "Aseprite plugin", OK, info["installed"])
    if info["state"] == "no_aseprite":
        return _item("plugin", "Aseprite plugin", BLOCKED, "no Aseprite found")
    detail = ("not installed" if info["state"] == "missing"
              else f"{info['installed']}, {info['bundled']} available")
    return _item("plugin", "Aseprite plugin", MISSING, detail)


def _model_item(paths) -> dict:
    marker = paths.models_dir / MODEL_FOLDER / MODEL_MARKER
    if marker.is_file():
        return _item("model", f"Model, {MODEL_GB} GB", OK, "downloaded",
                     needs=["deps"])
    return _item("model", f"Model, {MODEL_GB} GB", MISSING,
                 f"{_free_gb(paths.models_dir)} GB free", needs=["deps"])


def shortcut_path() -> pathlib.Path | None:
    if sys.platform == "darwin":
        return None
    base = os.environ.get("APPDATA")
    return (pathlib.Path(base) / "Microsoft" / "Windows" / "Start Menu"
            / "Programs" / "Spriteloom.lnk") if base else None


def _shortcut_item(paths) -> dict:
    lnk = shortcut_path()
    state = OK if lnk and lnk.is_file() else MISSING
    return _item("shortcut", "Start Menu shortcut", state,
                 "created" if state == OK else "not created", required=False)


def refresh_live(items: list[dict], paths) -> list[dict]:
    # filesystem-only, no subprocess -- cheap enough to run on every poll
    fresh = {"plugin": _plugin_item, "model": _model_item,
             "shortcut": _shortcut_item}
    return [fresh[it["id"]](paths) if it["id"] in fresh else it
            for it in items]


def check_all(paths, run=None) -> list[dict]:
    runner = run or run_command
    items = []
    backend = backend_name()
    label = backend_label(backend)

    version = paths.python_version
    if paths.python and (version is None or version >= MIN_PYTHON):
        shown = ".".join(str(part) for part in version) if version else "found"
        items.append(_item("python", "Python 3.11+", OK, shown))
        python_ok = True
    else:
        items.append(_item("python", "Python 3.11+", MISSING, "not found"))
        python_ok = False

    interpreter = venv_python(paths.root)
    if interpreter:
        items.append(_item("venv", "Virtual environment", OK, ".venv",
                           needs=["python"]))
    else:
        state = MISSING if python_ok else BLOCKED
        items.append(_item("venv", "Virtual environment", state, "missing",
                           needs=["python"]))

    if not interpreter:
        items.append(_item("torch", label, MISSING,
                           "needs the environment", needs=["venv"]))
        items.append(_item("deps", "Server dependencies", MISSING,
                           "needs the environment", needs=["venv", "torch"]))
    else:
        probe = MLX_PROBE if backend == "mlx" else TORCH_PROBE
        got = runner([str(interpreter), "-c", probe])
        cuda = _cuda_tag(got)
        if backend == "mlx" and got:
            items.append(_item("torch", label, OK,
                               f"MLX {got}", needs=["venv"]))
        elif backend == "cuda" and cuda:
            items.append(_item("torch", label, OK,
                               f"CUDA {cuda}", needs=["venv"]))
        else:
            detail = "missing MLX" if backend == "mlx" and not got else \
                ("no CUDA" if got else "missing")
            items.append(_item("torch", label, MISSING, detail,
                               needs=["venv"]))
        got = runner([str(interpreter), "-c", deps_probe(backend)])
        items.append(_item("deps", "Server dependencies",
                           OK if got else MISSING,
                           "installed" if got else "missing",
                           needs=["venv", "torch"]))

    items.append(_plugin_item(paths))
    items.append(_model_item(paths))
    # There is no Start Menu on macOS.  Omitting this optional row also keeps
    # it out of the setup selection closure.
    if sys.platform != "darwin":
        items.append(_shortcut_item(paths))
    return items
