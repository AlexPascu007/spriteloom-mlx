"""Spriteloom server: WebSocket endpoint on localhost."""
import asyncio
import functools
import hmac
import json
import logging
import os
import pathlib
import sys
import time
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import websockets
import websockets.exceptions

from server import models
from server.postprocess import (crop_to_subject, fit_into, mirror_symmetry,
                                subject_palette, sprite_palette,
                                remove_background)
from server.protocol import ProtocolError, parse_request, error_msg, \
    progress_msg, result_msg

log = logging.getLogger("spriteloom")


class DropBenignHandshakeAborts(logging.Filter):
    """The panel's ping keeps a short timeout and aborts sockets that don't
    open fast enough, e.g. while the GPU worker's GIL-heavy model load stalls
    the accept loop; websockets logs that abort as an ERROR. Drop only that
    exact case so real handshake failures still show up."""

    def filter(self, record):
        exc = record.exc_info[0] if record.exc_info else None
        return not (exc and issubclass(exc, websockets.exceptions.ConnectionClosed))


logging.getLogger("websockets.server").addFilter(DropBenignHandshakeAborts())

_gpu_executor = ThreadPoolExecutor(max_workers=1)
DEBUG_DIR = pathlib.Path("output")
DEBUG_SAVE = True  # tests set this to False so fake runs don't litter output/
RAW_KEEP = 10  # runs whose full-size raw_*.png survive pruning


def _slug(text, max_words=4, max_len=32):
    words = [w for w in "".join(
        c if c.isalnum() or c.isspace() else " " for c in text).split()]
    return "-".join(words[:max_words])[:max_len].rstrip("-") or "no-prompt"


def _run_folders():
    """Run folders newest first; names sort by timestamp."""
    if not DEBUG_DIR.exists():
        return []
    return sorted((p for p in DEBUG_DIR.iterdir() if p.is_dir()),
                  key=lambda p: p.name, reverse=True)


def _prune_raw(keep=RAW_KEEP):
    """Raws are only useful for tuning postprocess; history shows finals."""
    for folder in _run_folders()[keep:]:
        for f in folder.glob("raw_*.png"):
            try:
                f.unlink()
            except OSError:
                log.debug("could not prune %s", f)


def _save_debug(req, raw_images, final_images, seeds=None):
    """One folder per request: raw originals, final sprites, settings."""
    if not DEBUG_SAVE:
        return
    try:
        base = f"{time.strftime('%Y%m%d-%H%M%S')}_{_slug(req.prompt)}"
        folder = DEBUG_DIR / base
        n = 2
        while folder.exists():  # same second + same prompt: rare but possible
            folder = DEBUG_DIR / f"{base}_{n}"
            n += 1
        folder.mkdir(parents=True)
        for n, img in enumerate(raw_images):
            img.save(folder / f"raw_{n}.png")
        for n, img in enumerate(final_images):
            img.save(folder / f"final_{n}.png")
        meta = {"mode": req.mode, "prompt": req.prompt,
                "variants": req.variants,
                "target_size": list(req.target_size),
                "seeds": seeds or []}
        (folder / "settings.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        _prune_raw()
    except OSError:
        log.exception("debug save failed")  # never break generation over this


def _history_msg(offset, limit, preview=False):
    """Page of past runs, newest first; preview=True sends one image each."""
    from PIL import Image
    from server.protocol import image_to_raw
    folders = _run_folders()
    runs = []
    for pos, folder in enumerate(folders[offset:offset + max(limit, 0)],
                                 start=offset):
        try:
            meta = json.loads((folder / "settings.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        files = sorted(folder.glob("final_*.png"))
        images = []
        for f in (files[:1] if preview else files):
            with Image.open(f) as im:  # else the handle lives until GC
                images.append(image_to_raw(im))
        if images:
            # absolute folder index: imageless runs are skipped, so a list
            # position is not a valid server offset
            runs.append({"name": folder.name, "offset": pos,
                         "mode": meta.get("mode", "?"),
                         "prompt": meta.get("prompt", ""),
                         "count": len(files),
                         "seeds": meta.get("seeds", []),
                         "images": images})
    return json.dumps({"type": "history", "total": len(folders),
                       "offset": offset, "runs": runs})


def _register_default_models():
    """Idempotent registration of the real pipeline."""
    def klein():
        from server.backend import create_pipeline
        p = create_pipeline()
        p.load()
        return p

    models.register("klein", klein)


def _run(req, on_progress, on_stage):
    """Blocking generation + postprocess. Runs in the GPU worker thread."""
    def gen_progress(v):
        on_progress(v)
        if v >= 0.999:  # decode runs next, inside the pipe call
            on_stage("Decoding images")
    pipe = models.get("klein", on_stage=on_stage)
    seeds = pipe.variant_seeds(req.seed, req.variants)
    if req.mode in ("instruct", "edit"):
        # same Klein edit; they differ only in the panel UI
        raw = pipe.edit_by_instruction(req.prompt, req.frames[0].image,
                                       variants=req.variants,
                                       on_progress=gen_progress, seeds=seeds)
        palette_src = req.frames[0].image
    elif req.mode == "generate":
        on_stage("Preparing prompt")  # text encode runs before step ticks
        raw = pipe.txt2img(req.prompt, req.target_size,
                           variants=req.variants, on_progress=gen_progress,
                           seeds=seeds)
        palette_src = None
    else:  # inpaint - parse_request guarantees image+mask exist
        raw = pipe.inpaint(req.prompt, req.frames[0].image,
                           req.frames[0].mask, variants=req.variants,
                           on_progress=gen_progress, seeds=seeds)
        palette_src = req.frames[0].image

    # strip background at full res BEFORE shrinking: after the palette snap
    # the flood ate subject pixels that snapped near background colors
    pal = req.palette or (
        sprite_palette(palette_src) if palette_src is not None and req.preserve_palette else None)
    out = []
    on_stage(f"Post-processing 0/{len(raw)}")
    for img in raw:
        # inpaint's alpha is already exact from compositing, no flood needed
        if req.background == "keep" or req.mode == "inpaint":
            cut = img.convert("RGBA")
        else:
            cut = remove_background(img, tolerance=16,
                                    force=req.background == "remove")
        # Not inpaint: cropping would break its mask alignment.
        if req.mode in ("generate", "edit") or req.symmetry:
            cut = crop_to_subject(cut)
        small = fit_into(cut, req.target_size,
                         palette=pal or subject_palette(cut, 16))
        if req.symmetry:
            small = mirror_symmetry(small)
        out.append(small)
        on_stage(f"Post-processing {len(out)}/{len(raw)}")
    _save_debug(req, raw, out, seeds)
    return out, seeds


class JobCancelled(Exception):
    """Client disconnected mid-generation; abort the GPU job."""


async def handle_request(ws, req, state=None):
    loop = asyncio.get_running_loop()

    def send(msg):
        # runs in the GPU worker thread; a closed socket means Cancel
        if getattr(ws, "close_code", None) is not None:
            raise JobCancelled()
        f = asyncio.run_coroutine_threadsafe(ws.send(msg), loop)
        # blocking hands the GIL to the loop: measured up to 2x faster
        # delivery than fire-and-forget (ordering holds either way)
        try:
            f.result(5)
        except Exception as e:
            log.debug("progress send failed: %r", e)

    def on_progress(v):
        send(progress_msg(req.id, v))

    def on_stage(text):
        send(progress_msg(req.id, 0.0, stage=text))

    if state:
        state.active += 1
    try:
        images, seeds = await loop.run_in_executor(
            _gpu_executor, functools.partial(_run, req, on_progress, on_stage))
        await ws.send(result_msg(req.id, images, seeds))
    finally:
        if state:
            state.active -= 1
            state.last_activity = time.monotonic()


async def _handler(ws, state=None):
    try:
        await _serve_connection(ws, state)
    except websockets.exceptions.ConnectionClosed:
        # the plugin opens a fresh socket per ping/request and drops it without
        # a close handshake; that is normal, not a handler failure to log
        log.debug("client closed connection without a close frame")


async def _serve_connection(ws, state=None):
    async for message in ws:
        try:
            data = json.loads(message)
            if isinstance(data, dict) and data.get("type") == "ping":
                # the panel shows a load bar until Klein is resident
                model = "ready" if models.is_ready("klein") else "loading"
                from server.config import load_backend
                pong = {"type": "pong", "model": model, "server": "spriteloom",
                        "protocol": 1, "pid": os.getpid(), "backend": load_backend(),
                        "managed": bool(state and state.token),
                        "busy": state.active if state else 0,
                        "capabilities": {"palette_control": True, "masked_inpaint": True}}
                from server.config import load_mlx_quantize
                pong["settings"] = {"model_id": "black-forest-labs/FLUX.2-klein-4B",
                                    "steps": 8, "working_max_side": 512,
                                    "quantize": load_mlx_quantize() if load_backend() == "mlx" else None}
                if model == "loading":
                    prog = models.load_progress()
                    pong["progress"] = round(prog[0], 3) if prog else 0.0
                    if prog and prog[1]:
                        pong["stage"] = prog[1]
                await ws.send(json.dumps(pong))
                continue
            if isinstance(data, dict) and data.get("type") == "shutdown":
                token = data.get("token")
                if (not state or not state.token or not isinstance(token, str)
                        or not hmac.compare_digest(state.token, token)):
                    await ws.send(error_msg("", "service shutdown is not authorized"))
                elif state.active:
                    await ws.send(error_msg("", "service is busy; wait for its requests to finish"))
                else:
                    await ws.send(json.dumps({"type": "shutdown", "stopped": True}))
                    if not state.stop.done():
                        state.stop.set_result(None)
                    return
                continue
            if isinstance(data, dict) and data.get("type") == "history":
                await ws.send(_history_msg(int(data.get("offset", 0)),
                                           int(data.get("limit", 1)),
                                           bool(data.get("preview"))))
                continue
        except (json.JSONDecodeError, ValueError, TypeError):
            pass  # fall through to parse_request for a proper error
        try:
            req = parse_request(message)
        except ProtocolError as e:
            await ws.send(error_msg("", str(e)))
            continue
        try:
            await handle_request(ws, req, state)
        except JobCancelled:
            log.info("request %s cancelled by client", req.id)
            break  # socket is already closed; stop serving this connection
        except websockets.exceptions.ConnectionClosed:
            raise  # client vanished mid-request; _handler logs it at debug
        except Exception as e:  # never die silently
            log.exception("request failed")
            try:
                await ws.send(error_msg(req.id, f"{type(e).__name__}: {e}"))
            except Exception:
                log.debug("client gone before error could be delivered")


def _preload_klein(on_fatal=None):
    """Warm the model at startup; a request arriving mid-load queues behind."""
    def done(f):
        exc = f.exception()
        if exc:
            log.error("model did not load: %s", exc)
            if on_fatal:
                on_fatal()
        else:
            log.info("klein preloaded, ready for prompts")
    _gpu_executor.submit(models.get, "klein").add_done_callback(done)


async def serve(host="127.0.0.1", port=8765, stop=None, on_ready=None,
                preload=False):
    if not models._factories:
        _register_default_models()
    # GIL-heavy worker; a short switch interval keeps pings answered
    sys.setswitchinterval(0.001)
    loop = asyncio.get_running_loop()
    if stop is None:
        stop = loop.create_future()
    state = _ServiceState(stop)
    if preload:
        # a failed load needs the user to act; exit rather than hang on 'loading'
        def on_fatal():
            loop.call_soon_threadsafe(
                lambda: stop.done() or stop.set_result(None))
        _preload_klein(on_fatal)
    # no keepalive: the 20s ping timeout killed sockets mid-job
    monitor = asyncio.create_task(state.watch_idle()) if state.token and state.idle > 0 else None
    try:
        async with websockets.serve(functools.partial(_handler, state=state), host, port,
                                    max_size=64 * 2**20, ping_interval=None):
            log.info("Spriteloom server on ws://%s:%s", host, port)
            if on_ready:
                on_ready()
            await stop
    finally:
        if monitor:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)


class _ServiceState:
    def __init__(self, stop):
        self.stop = stop
        self.token = os.environ.get("SPRITELOOM_SERVICE_TOKEN", "")
        self.idle = float(os.environ.get("SPRITELOOM_IDLE_SECONDS", "600")) if self.token else 0
        self.active = 0
        self.last_activity = time.monotonic()

    async def watch_idle(self):
        while not self.stop.done():
            await asyncio.sleep(min(1, self.idle))
            if not self.active and time.monotonic() - self.last_activity >= self.idle:
                log.info("idle model service stopped")
                self.stop.set_result(None)


def _watch_launcher():
    """Release the GPU even if the macOS launcher is force-quit."""
    owner = os.environ.get("SPRITELOOM_PARENT_PID")
    if os.name != "posix" or not owner:
        return
    try:
        owner = int(owner)
    except ValueError:
        raise ValueError("SPRITELOOM_PARENT_PID must be a process ID") from None

    def watch():
        while os.getppid() == owner:
            time.sleep(1)
        # A model can be inside a native GPU call. A normal Python exit would
        # wait for its worker thread. The OS releases GPU memory on exit.
        os._exit(0)

    threading.Thread(target=watch, name="launcher-watch", daemon=True).start()


if __name__ == "__main__":
    from server.config import HOST, load_port
    logging.basicConfig(level=logging.INFO)
    _watch_launcher()
    # The panel pings every 10s; websockets logs every open/close at INFO.
    logging.getLogger("websockets").setLevel(logging.WARNING)
    # Benign on Windows: torch.distributed cannot redirect stdio; warns once
    # at import. Silence before torch loads.
    logging.getLogger(
        "torch.distributed.elastic.multiprocessing.redirects"
    ).setLevel(logging.ERROR)
    # bitsandbytes casts to fp16 for 8-bit matmul on purpose; not a bug
    warnings.filterwarnings("ignore", message="MatMul8bitLt")
    try:
        port = int(os.environ.get("SPRITELOOM_PORT", load_port()))
        asyncio.run(serve(host=HOST, port=port, preload=True))
    except KeyboardInterrupt:
        log.info("Spriteloom server stopped. Bye.")
