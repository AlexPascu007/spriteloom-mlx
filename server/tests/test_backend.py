import json
import sys
import types
import os
import signal
import subprocess

import pytest

from server import backend, config


def test_apple_silicon_selects_mlx(monkeypatch, tmp_path):
    monkeypatch.delenv("SPRITELOOM_BACKEND", raising=False)
    monkeypatch.setattr(config.sys, "platform", "darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")
    assert config.load_backend(tmp_path / "missing") == "mlx"


def test_windows_keeps_cuda(monkeypatch, tmp_path):
    monkeypatch.delenv("SPRITELOOM_BACKEND", raising=False)
    monkeypatch.setattr(config.sys, "platform", "win32")
    assert config.load_backend(tmp_path / "missing") == "cuda"


def test_backend_override_and_invalid_values(monkeypatch, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"backend": "cuda", "mlx_quantize": 4}))
    monkeypatch.setenv("SPRITELOOM_BACKEND", "mlx")
    assert config.load_backend(path) == "mlx"
    assert config.load_mlx_quantize(path) == 4
    monkeypatch.setenv("SPRITELOOM_BACKEND", "mxl")
    with pytest.raises(ValueError, match="backend"):
        config.load_backend(path)
    path.write_text(json.dumps({"mlx_quantize": True}))
    with pytest.raises(ValueError, match="mlx_quantize"):
        config.load_mlx_quantize(path)


def test_mlx_factory_does_not_import_cuda_pipeline(monkeypatch):
    class FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    monkeypatch.setattr(config, "load_backend", lambda: "mlx")
    monkeypatch.setattr(config, "load_models_dir", lambda: "local-cache")
    monkeypatch.setattr(config, "load_mlx_quantize", lambda: 8)
    monkeypatch.setitem(sys.modules, "server.instruct_mlx",
                        types.SimpleNamespace(MLXKleinPipeline=FakePipeline))
    assert backend.create_pipeline().kwargs == {
        "models_dir": "local-cache", "quantize": 8}


@pytest.mark.skipif(os.name != "posix", reason="macOS/Linux parent monitor")
def test_server_exits_if_launcher_is_killed():
    # Child shares the stdout pipe. communicate() cannot return until the
    # child has also exited, even after the launcher process is killed.
    child = (
        "from server.main import _watch_launcher; import time; "
        "_watch_launcher(); print('ready', flush=True); time.sleep(30)"
    )
    parent = (
        "import os, subprocess, sys, time; "
        "env=dict(os.environ, SPRITELOOM_PARENT_PID=str(os.getpid())); "
        f"subprocess.Popen([sys.executable, '-u', '-c', {child!r}], env=env); "
        "time.sleep(30)"
    )
    proc = subprocess.Popen([sys.executable, "-u", "-c", parent],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        proc.kill()
        proc.communicate(timeout=5)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
