import asyncio
from contextlib import asynccontextmanager
import json
import socket

import websockets

from server import main, models


@asynccontextmanager
async def running(monkeypatch, *, idle="600"):
    monkeypatch.setenv("SPRITELOOM_SERVICE_TOKEN", "test-token")
    monkeypatch.setenv("SPRITELOOM_IDLE_SECONDS", idle)
    models.reset()
    models.register("klein", object)
    models.get("klein")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    stop = asyncio.get_running_loop().create_future()
    ready = asyncio.Event()
    task = asyncio.create_task(main.serve(port=port, stop=stop, on_ready=ready.set))
    await asyncio.wait_for(ready.wait(), 3)
    try:
        yield f"ws://127.0.0.1:{port}", task
    finally:
        if not stop.done():
            stop.set_result(None)
        await asyncio.wait_for(task, 3)
        models.reset()


def test_identity_and_authenticated_shutdown(monkeypatch):
    async def run():
        async with running(monkeypatch) as (url, task):
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps({"type": "ping"}))
                status = json.loads(await ws.recv())
                assert status["server"] == "spriteloom"
                assert status["protocol"] == 1
                assert status["capabilities"]["palette_control"]
                await ws.send(json.dumps({"type": "shutdown", "token": "wrong"}))
                assert json.loads(await ws.recv())["type"] == "error"
                assert not task.done()
                await ws.send(json.dumps({"type": "shutdown", "token": "test-token"}))
                assert json.loads(await ws.recv())["stopped"]
            await asyncio.wait_for(task, 3)
    asyncio.run(run())


def test_idle_service_stops_without_generation(monkeypatch):
    async def run():
        async with running(monkeypatch, idle="0.05") as (_, task):
            await asyncio.wait_for(task, 3)
    asyncio.run(run())
