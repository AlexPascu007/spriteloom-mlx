"""Run all four tasks through a local WebSocket server with the real MLX model.

Run from the project root: .venv/bin/python -m server.macos_check
Results and machine measurements are saved under output/macos-validation/.
"""
import argparse
import asyncio
import json
import pathlib
import platform
import time

from PIL import Image
import numpy as np
import websockets

from server import main as server
from server.protocol import image_from_raw, image_to_b64


async def check(port, output):
    import mlx.core as mx
    from server import config, models

    if config.load_backend() != "mlx":
        raise RuntimeError("this check requires the MLX backend")
    output.mkdir(parents=True, exist_ok=True)
    report = {"device": dict(mx.device_info()), "python": platform.python_version(),
              "mlx": mx.__version__, "quantize": config.load_mlx_quantize(), "runs": []}
    payloads = []
    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    ready = asyncio.Event()
    task = asyncio.create_task(server.serve(port=port, stop=stop, on_ready=ready.set))
    await ready.wait()

    async def request(mode, prompt, source=None, mask=None):
        payload = {"id": f"check-{mode}", "mode": mode, "prompt": prompt,
                   "target_size": [64, 64], "variants": 1, "seed": 42}
        if source is not None:
            frame = {"image": image_to_b64(source)}
            if mask is not None:
                frame["mask"] = image_to_b64(mask)
            payload["frames"] = [frame]
        started = time.perf_counter()
        values = []
        async with websockets.connect(f"ws://127.0.0.1:{port}", ping_interval=None) as ws:
            await ws.send(json.dumps(payload))
            while True:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=900))
                if message["type"] == "error":
                    raise RuntimeError(message["message"])
                if message["type"] == "progress":
                    values.append(message["value"])
                if message["type"] == "result":
                    break
        image = image_from_raw(message["images"][0])
        payloads.append(message["images"][0])
        image.save(output / f"{mode}.png")
        data = {"mode": mode, "prompt": prompt, "seconds": round(time.perf_counter() - started, 2),
                "size": list(image.size), "progress_events": len(values),
                "peak_mlx_gb": round(mx.get_peak_memory() / 1e9, 3)}
        if image.size != (64, 64) or message["seeds"] != [42]:
            raise AssertionError("wrong output size or seed")
        if not values or max(values) < 1:
            raise AssertionError("missing generation progress")
        if mode == "inpaint":
            from server.instruct import INPAINT_MARGIN
            from PIL import ImageFilter
            allowed = mask.filter(ImageFilter.MaxFilter(INPAINT_MARGIN * 2 + 1))
            alpha = np.asarray(image.getchannel("A"))
            if np.any(alpha[np.asarray(allowed) == 0]):
                raise AssertionError("inpaint wrote outside the mask margin")
            data["alpha_outside_mask_margin"] = 0
        report["runs"].append(data)
        (output / "measurements.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(data), flush=True)
        return image

    try:
        source = await request("generate", "A small closed medieval wooden treasure chest with a brass lock, front view")
        await request("edit", "Open the lid of the treasure chest halfway. Keep its materials and brass lock.", source)
        await request("instruct", "Show this same small wooden treasure chest from the back", source)
        mask = Image.new("L", source.size)
        mask.paste(255, (20, 16, 44, 40))
        mask.save(output / "selection.png")
        await request("inpaint", "Add a small bright red gem to the treasure chest", source, mask)
        pipe = models.get("klein")
        report["model_snapshot"] = str(pipe._model_path)
        from server.instruct_mlx import MFLUX_COMMIT
        report["mflux_commit"] = MFLUX_COMMIT
        report["shared_transformer"] = pipe._model.transformer is pipe._edit_model.transformer
        report["shared_vae"] = pipe._model.vae is pipe._edit_model.vae
        report["callbacks_after_requests"] = {
            name: len(getattr(pipe._model.callbacks, name))
            for name in ("before_loop", "in_loop", "after_loop", "interrupt")}
        (output / "aseprite-payload.json").write_text(json.dumps({"images": payloads}))
        (output / "measurements.json").write_text(json.dumps(report, indent=2))
    finally:
        if not stop.done():
            stop.set_result(None)
        await task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("output/macos-validation"))
    args = parser.parse_args()
    asyncio.run(check(args.port, args.output))


if __name__ == "__main__":
    main()
