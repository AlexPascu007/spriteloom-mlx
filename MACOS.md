# Run SpriteLoom on this Mac

Open `~/Applications/Spriteloom.app`, then press **Start**. Keep the launcher
open while you use the extension. Closing the launcher stops its server.

Open `~/Applications/Aseprite.app`. Create or open a sprite, then choose
**Sprite → Spriteloom...**. Start with one variant.

The launcher app is linked to this project folder. Keep the project folder,
its `.venv`, `server`, `plugin`, and `models` folders together. The app bundle
contains the launcher; the model runs in the external Python environment.
Run `./build-macos.sh` after you change launcher code.

## Local verification, 2026-10-22

- Machine reports Apple M5, 32 GB unified memory, macOS 27.2.
- Python 3.12.13; MLX 0.32.4; MFLUX commit
  `0fea52bada4ade619f1cdb7ae26087b32d87dd89`.
- FLUX.2 Klein 4B snapshot `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
- Eight-bit weights, one variant at a time, eight steps, 512-pixel working
  image, 64×64 final sprite.
- Requests through the real local WebSocket server: Generate 16.61 seconds;
  Edit 18.64 seconds; Rotate/Instruct 19.22 seconds; Inpaint 21.36 seconds.
  Generate includes the first model load in this process.
- Peak MLX allocation across these requests: 12.281 GB. This measures MLX
  allocations, rather than total system memory use.
- The generation and edit views share the transformer and VAE. Request
  callbacks are removed after use. The inpaint result has no alpha outside
  the selection's existing two-pixel margin.
- Python checks: 230 passed, 2 skipped. Lua plugin tests passed. Luacheck:
  zero warnings and errors.
- Aseprite was built from source at
  `b3e34acbae812b8a1d814c2e6285343b351c8945`, version
  `1.3.18.6-6-gb3e34acb-dev`. Its batch checks confirm that the SpriteLoom
  command, Lua, Timer, and WebSocket are available.
- A real Aseprite batch run decoded all four model results through the
  plugin's RGBA decoder, inserted them as four layers, checked that the
  existing source layer stayed unchanged, and saved `integration.aseprite`.
- The app launcher's Start button loaded the MLX server. An edit through
  that server opened the chest lid in 17.78 seconds. See `edit-open.png`.

Images and measurements are in `output/macos-validation/`. To repeat the
real server check:

```sh
.venv/bin/python -m server.macos_check
~/Applications/Aseprite.app/Contents/MacOS/aseprite --batch --script plugin/tests/test_aseprite_integration.lua
```

## Limits

The launcher was checked in its native window. Automated reading of the
Aseprite window returned `timeoutReached`, so the complete extension click
workflow has not been checked. The installed extension command and the real
server protocol were checked separately, along with result insertion in
the real Aseprite scripting runtime.

Image quality and angle instructions depend on the model. Editing normally
uses the source sprite's palette; a requested new color can map back to an
existing palette color. MLX and CUDA can produce different images from the
same seed. Other Apple chips and memory sizes have not been tested here.

The launcher has a local ad-hoc signature. Aseprite is a personal source
build. These are local development installations.

## Setup on another checkout

```sh
./setup-macos.sh
./build-macos.sh
open ./Spriteloom.app
```

The setup script installs the Python environment, dependencies, original
model weights, and Aseprite extension. Aseprite itself must be installed or
built separately. Use `./start-server.sh` for a terminal server, and
`.venv/bin/python -m server.diagnose` to check the model loader.
