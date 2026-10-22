"""Copying the plugin into Aseprite; the replacement for install-plugin.bat."""
import json
import os
import pathlib
import shutil
import sys

FOLDER = "spriteloom"
PORT_FILE = "server.json"


def source_dir() -> pathlib.Path:
    # frozen: the plugin sits next to the exe; in the repo it is plugin/
    if getattr(sys, "frozen", False):
        return pathlib.Path(sys.executable).parent / "plugin"
    return pathlib.Path(__file__).resolve().parent.parent / "plugin"


def dest_in(aseprite_dir) -> pathlib.Path:
    """The install target under a given Aseprite folder."""
    return pathlib.Path(aseprite_dir) / "extensions" / FOLDER


def user_config_dir() -> pathlib.Path | None:
    """Return Aseprite's per-user folder on the current platform."""
    override = os.environ.get("ASEPRITE_USER_FOLDER")
    if override:
        return pathlib.Path(override)
    if sys.platform == "darwin":
        base = os.environ.get("APPDATA")
        return ((pathlib.Path(base) / "Aseprite") if base else
                pathlib.Path.home() / "Library" / "Application Support" /
                "Aseprite")
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        return pathlib.Path(base) / "Aseprite" if base else None
    base = os.environ.get("XDG_CONFIG_HOME")
    root = pathlib.Path(base) if base else pathlib.Path.home() / ".config"
    return root / "aseprite"


def extensions_dir() -> pathlib.Path | None:
    aseprite = user_config_dir()
    if not aseprite or not aseprite.is_dir():
        return None
    return dest_in(aseprite)


def read_version(folder: pathlib.Path) -> str | None:
    try:
        data = json.loads((folder / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = data.get("version")
    return version if isinstance(version, str) else None


def status(source: pathlib.Path, dest: pathlib.Path | None) -> dict:
    bundled = read_version(source)
    if dest is None:
        return {"state": "no_aseprite", "installed": None, "bundled": bundled}
    installed = read_version(dest)
    if installed is None:
        return {"state": "missing", "installed": None, "bundled": bundled}
    state = "current" if installed == bundled else "outdated"
    return {"state": state, "installed": installed, "bundled": bundled}


def install(source: pathlib.Path, dest: pathlib.Path, port: int) -> None:
    # top level files only: tests/ and .pytest_cache/ do not belong in Aseprite
    dest.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)
    (dest / PORT_FILE).write_text(json.dumps({"port": port}, indent=2),
                                  encoding="utf-8")
