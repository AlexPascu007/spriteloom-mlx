# spriteloom-mlx

Local pixel-art generation and editing on Apple Silicon, using MLX and
FLUX.2 Klein 4B. Run it as a headless engine for
[spriteloom-mcp](https://github.com/AlexPascu007/spriteloom-mcp), or use the
optional Aseprite extension.

**Based on [SpriteLoom](https://github.com/vkarach/spriteloom), created by
Valentyn Karachevtsev (vkarach).** This independent port keeps the original
Apache 2.0 license and Git history. The original author built the application,
pixel-art workflow, prompts, protocol, image processing, and Aseprite extension.
This repository adds an MLX backend, masked latent processing, macOS setup,
and a shared headless service. See [NOTICE](NOTICE).

## What it does

- Generate sprites from text.
- Edit an existing sprite by instruction.
- Redraw a masked region while preserving source latents outside it.
- Generate another view of a subject.
- Reduce model output to a pixel-art canvas and palette.

The generation and editing paths share one resident model. Inference runs
locally. Setup downloads the model weights; generation does not use a cloud
image API.

## Quick start

Requires Apple Silicon, Python 3.12 or [uv](https://docs.astral.sh/uv/), and
about 16 GB of free disk space for the original weights. The tested machine
has an Apple M5 and 32 GB of unified memory. Other configurations need tests.

```sh
git clone https://github.com/AlexPascu007/spriteloom-mlx.git
cd spriteloom-mlx
./setup-mlx.sh
./start-server.sh
```

The engine listens on `127.0.0.1:8765`. Aseprite and a graphical launcher
are optional. For an agent workflow, install the separate
[MCP server and skill](https://github.com/AlexPascu007/spriteloom-mcp).

## Tested behavior

On the test machine, one 64×64 sprite took about 17–21 seconds, using a
512-pixel working image, eight steps, and 8-bit MLX weights. Peak MLX
allocation across all four operations was 12.3 GB. This is a small local test,
rather than a hardware benchmark. [MACOS.md](MACOS.md) has the test details.

The MLX masked path restores the source/noise mixture outside the mask after
each denoising step. The native protocol returns a transparent patch with
the upstream two-pixel selection margin. The MCP layer clips that patch to
the exact input mask and can return a complete edited sprite.

Angle instructions and artistic quality depend on the model. MLX and CUDA
can produce different images from the same seed. Palette preservation is on
by default; API clients can disable it when a request needs new colors.

## Development

```sh
.venv/bin/python -m pytest server/tests launcher/tests
.venv/bin/python -m server.macos_check
.venv/bin/python -m server.diagnose
```

MFLUX is pinned at `0fea52bada4ade619f1cdb7ae26087b32d87dd89`. The model setup
uses BFL snapshot `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
Model files, virtual environments, and generated outputs are excluded from Git.

[ASEPRITE.md](ASEPRITE.md) describes the optional extension and legacy Windows
workflow. Windows/CUDA code is retained from upstream; the new local tests
used MLX on macOS.

## License and credits

- Original SpriteLoom: Valentyn Karachevtsev, [Apache 2.0](LICENSE).
- MLX model implementation: [MFLUX](https://github.com/mflux-community/mflux), MIT.
- Model: [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B),
  Black Forest Labs, Apache 2.0. Weights are downloaded separately.

This is an independent port, with no claimed endorsement from the original
author or model providers.
