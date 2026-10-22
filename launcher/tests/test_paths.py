import pathlib

from launcher import paths


def fake_run(answers):
    def run(cmd):
        return answers.get(cmd[0])
    return run


def test_probe_python_accepts_new_enough():
    run = fake_run({"py": r"C:\Python314\python.exe|3.14.0"})
    got = paths.probe_python(["py", "-3"], run=run)
    assert got == (pathlib.Path(r"C:\Python314\python.exe"), (3, 14, 0))


def test_probe_python_rejects_old():
    run = fake_run({"py": r"C:\Python39\python.exe|3.9.7"})
    assert paths.probe_python(["py", "-3"], run=run) is None


def test_probe_python_handles_no_interpreter():
    assert paths.probe_python(["nope"], run=fake_run({})) is None


def test_probe_python_handles_garbage():
    assert paths.probe_python(["py"], run=fake_run({"py": "hello"})) is None


def test_detect_python_falls_back_to_second_candidate():
    run = fake_run({"python": r"C:\P\python.exe|3.12.1"})
    got = paths.detect_python(run=run)
    assert got == (pathlib.Path(r"C:\P\python.exe"), (3, 12, 1))


def test_resolve_prefers_overrides(tmp_path):
    settings = {"root": str(tmp_path), "python": r"C:\custom\python.exe",
                "aseprite_dir": str(tmp_path / "ase"),
                "models_dir": str(tmp_path / "big")}
    got = paths.resolve(settings, run=fake_run({}))
    assert got.root == tmp_path
    assert got.python == pathlib.Path(r"C:\custom\python.exe")
    assert got.aseprite_dir == tmp_path / "ase"
    assert got.models_dir == tmp_path / "big"


def test_resolve_detects_when_not_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    (tmp_path / "Aseprite").mkdir()
    run = fake_run({"py": r"C:\P\python.exe|3.13.0"})
    got = paths.resolve({}, run=run)
    assert got.aseprite_dir == tmp_path / "Aseprite"
    assert got.python == pathlib.Path(r"C:\P\python.exe")
    assert got.models_dir == got.root / "models"


def test_resolve_reports_missing_aseprite(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    got = paths.resolve({}, run=fake_run({}))
    assert got.aseprite_dir is None
    assert got.python is None


def test_macos_prefers_python312(monkeypatch):
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    run = fake_run({"python3.12": "/opt/homebrew/bin/python3.12|3.12.8",
                    "python3": "/opt/homebrew/bin/python3|3.14.0"})
    assert paths.detect_python(run=run) == (
        pathlib.Path("/opt/homebrew/bin/python3.12"), (3, 12, 8))


def test_macos_uv_python_is_supported(monkeypatch):
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    run = fake_run({"/opt/homebrew/bin/uv": "/opt/homebrew/opt/python@3.12/bin/python3.12",
                    "/opt/homebrew/opt/python@3.12/bin/python3.12":
                    "/opt/homebrew/opt/python@3.12/bin/python3.12|3.12.8"})
    assert paths.detect_python(run=run)[1] == (3, 12, 8)


def test_macos_aseprite_dir_uses_application_support(tmp_path, monkeypatch):
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    monkeypatch.setattr(paths.pathlib, "Path", pathlib.Path)
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = tmp_path / "Library" / "Application Support" / "Aseprite"
    folder.mkdir(parents=True)
    assert paths.detect_aseprite_dir() == folder


def test_resolve_prefers_existing_venv_interpreter(tmp_path):
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    run = fake_run({str(python): f"{python}|3.12.8"})
    got = paths.resolve({"root": str(tmp_path)}, run=run)
    assert got.python == python
    assert got.python_version == (3, 12, 8)


def test_frozen_macos_app_root_is_the_project_parent(tmp_path, monkeypatch):
    executable = (tmp_path / "Spriteloom.app" / "Contents" / "MacOS" /
                  "Spriteloom")
    executable.parent.mkdir(parents=True)
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    monkeypatch.setattr(paths.sys, "executable", str(executable))
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    assert paths.app_root() == tmp_path
