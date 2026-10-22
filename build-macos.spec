"""PyInstaller recipe for the macOS launcher app.

The app contains only the launcher and its Cocoa webview.  The server, MLX,
MFLUX, model, virtual environment, and plugin stay beside the app bundle in
the project directory and are installed by the macOS setup script.
"""

from pathlib import Path

from PyInstaller.building.api import EXE, PYZ
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.osx import BUNDLE


# PyInstaller executes a spec in its own namespace, where SPECPATH is the
# directory containing this file (``__file__`` is not defined).
ROOT = Path(SPECPATH).resolve()


a = Analysis(
    [str(ROOT / "launcher" / "app.py")],
    pathex=[str(ROOT)],
    datas=[(str(ROOT / "launcher" / "ui"), "ui")],
    hiddenimports=[
        # pywebview selects its backend at runtime.
        "webview.platforms.cocoa",
        "server.config",
    ],
    excludes=[
        # The frozen launcher never imports the server's ML stack.  Keeping
        # these out makes the app small and leaves the runtime in .venv.
        "mlx", "mflux", "torch", "diffusers", "transformers", "accelerate",
        "peft", "bitsandbytes", "numpy", "scipy", "safetensors", "PIL",
        "pytest", "tkinter",
        # Cocoa is the only pywebview backend used by this bundle.
        "webview.platforms.android", "webview.platforms.cef",
        "webview.platforms.edgechromium", "webview.platforms.gtk",
        "webview.platforms.mshtml", "webview.platforms.qt",
        "webview.platforms.win32", "webview.platforms.winforms",
    ],
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Spriteloom",
    console=False,
    target_arch="arm64",
    codesign_identity=None,  # PyInstaller uses an ad-hoc signature locally.
    upx=False,
)
app = BUNDLE(
    exe,
    a.binaries,
    a.datas,
    name="Spriteloom.app",
    bundle_identifier="local.spriteloom.launcher",
    version="1.0.0",
    info_plist={
        "CFBundleName": "Spriteloom",
        "CFBundleDisplayName": "Spriteloom",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
    },
)
