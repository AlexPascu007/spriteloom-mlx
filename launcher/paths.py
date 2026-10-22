"""Where things live: detected first, overridden by the settings file."""
import dataclasses
import os
import pathlib
import subprocess
import sys

from server.config import load_settings

MIN_PYTHON = (3, 11)
# Keep the Windows launcher command first.  On macOS, Python 3.12 is checked
# before generic python3 so that a system Python 3.9 (or an unrelated newer
# install) is not selected when the supported interpreter is available.
PYTHON_CANDIDATES = (["py", "-3"], ["python"])
PROBE = "import sys; print(sys.executable + '|%d.%d.%d' % sys.version_info[:3])"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _python_candidates() -> list[list[str]]:
    """Return interpreter commands in platform-specific preference order."""
    if sys.platform == "win32":
        return [list(command) for command in PYTHON_CANDIDATES]
    if sys.platform == "darwin":
        return [
            ["/opt/homebrew/bin/python3.12"],
            ["/usr/local/bin/python3.12"],
            ["python3.12"],
            ["/opt/homebrew/bin/python3"],
            ["/usr/local/bin/python3"],
            ["python3"],
            ["python"],
            # Keep copied Windows test fixtures and cross-platform profiles
            # usable; macOS normally has no ``py`` launcher.
            ["py", "-3"],
        ]
    return [["python3.12"], ["python3"], ["python"]]


def app_root() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        executable = pathlib.Path(sys.executable)
        if sys.platform == "darwin":
            # A PyInstaller app runs from
            # ``<root>/Spriteloom.app/Contents/MacOS/Spriteloom``.  The
            # model, venv, server, and plugin remain beside the app bundle.
            for ancestor in executable.parents:
                if ancestor.suffix == ".app":
                    return ancestor.parent
        return executable.parent
    return pathlib.Path(__file__).resolve().parent.parent


def run_command(cmd: list[str]) -> str | None:
    """Stdout of a short probe, or None on failure; setup_checks shares it."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                              creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def probe_python(command, run=None):
    """(path, version) for a working 3.11+ interpreter, else None."""
    out = (run or run_command)(list(command) + ["-c", PROBE])
    if not out or "|" not in out:
        return None
    exe, _, raw = out.rpartition("|")
    try:
        version = tuple(int(part) for part in raw.split("."))
    except ValueError:
        return None
    if len(version) != 3 or version < MIN_PYTHON:
        return None
    return pathlib.Path(exe), version


def detect_python(run=None):
    runner = run or run_command
    # uv can manage a supported Python without changing the user's PATH.  Its
    # Homebrew path is checked explicitly because GUI apps often receive a
    # smaller PATH than a terminal shell.
    if sys.platform == "darwin":
        for uv in ("/opt/homebrew/bin/uv", "/usr/local/bin/uv", "uv"):
            found = runner([uv, "python", "find", "3.12"])
            if found:
                executable = found.splitlines()[-1].strip()
                if executable:
                    probed = probe_python([executable], run=runner)
                    if probed:
                        return probed
    for command in _python_candidates():
        found = probe_python(command, run=runner)
        if found:
            return found
    return None


def detect_aseprite_dir() -> pathlib.Path | None:
    # Aseprite supports ASEPRITE_USER_FOLDER for testing and alternate
    # installations.  The default macOS location is documented by Aseprite.
    override = os.environ.get("ASEPRITE_USER_FOLDER")
    if override:
        folder = pathlib.Path(override)
    elif sys.platform == "darwin":
        # APPDATA is accepted as a compatibility override for copied Windows
        # profiles and test fixtures.  A normal macOS install uses the path
        # below.
        base = os.environ.get("APPDATA")
        folder = ((pathlib.Path(base) / "Aseprite") if base else
                  pathlib.Path.home() / "Library" / "Application Support" /
                  "Aseprite")
    elif sys.platform == "win32":
        base = os.environ.get("APPDATA")
        if not base:
            return None
        folder = pathlib.Path(base) / "Aseprite"
    else:
        base = os.environ.get("XDG_CONFIG_HOME")
        folder = (pathlib.Path(base) if base else pathlib.Path.home() / ".config") \
            / "aseprite"
    return folder if folder.is_dir() else None


@dataclasses.dataclass
class Paths:
    root: pathlib.Path
    python: pathlib.Path | None
    python_version: tuple | None
    aseprite_dir: pathlib.Path | None
    models_dir: pathlib.Path


def resolve(settings: dict | None = None, run=None) -> Paths:
    values = settings if settings is not None else load_settings()
    root = pathlib.Path(values["root"]) if values.get("root") else app_root()
    if values.get("python"):
        python = pathlib.Path(values["python"])
        probed = probe_python([str(python)], run=run)
        version = probed[1] if probed else None
    else:
        # Once setup has created an environment, use its interpreter for all
        # launcher probes.  It is the runtime that will actually start the
        # server and it avoids showing a different system Python in the UI.
        venv_candidates = (root / ".venv" / "bin" / "python",
                           root / ".venv" / "Scripts" / "python.exe")
        found = None
        for candidate in venv_candidates:
            if candidate.exists():
                probed = probe_python([str(candidate)], run=run)
                if probed:
                    found = probed
                    break
        found = found or detect_python(run=run)
        python, version = found if found else (None, None)
    if values.get("aseprite_dir"):
        aseprite = pathlib.Path(values["aseprite_dir"])
    else:
        aseprite = detect_aseprite_dir()
    models = (pathlib.Path(values["models_dir"]) if values.get("models_dir")
              else root / "models")
    return Paths(root=root, python=python, python_version=version,
                 aseprite_dir=aseprite, models_dir=models)
