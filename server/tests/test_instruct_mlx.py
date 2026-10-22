import sys
import json
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from server.instruct import STYLE_SUFFIX, T2I_SUFFIX, t2i_size
from server.instruct_mlx import MLXKleinPipeline


class _FakeCallbacks:
    def __init__(self):
        self.in_loop = []

    def register(self, callback):
        self.in_loop.append(callback)


class _FakeResult:
    def __init__(self, image):
        self.image = image


class _FakeKlein(dict):
    init_calls = []

    def __init__(self, **kwargs):
        super().__init__()
        type(self).init_calls.append(kwargs)
        self.kwargs = kwargs
        self.callbacks = _FakeCallbacks()
        self.vae = object()
        self.transformer = object()
        self.tokenizers = object()
        self.model_config = object()
        self.calls = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        config = SimpleNamespace(num_inference_steps=kwargs["num_inference_steps"])
        for step in range(config.num_inference_steps):
            for callback in list(self.callbacks.in_loop):
                callback.call_in_loop(step, config=config)
        return _FakeResult(Image.new("RGB", (kwargs["width"], kwargs["height"]), "red"))


class _FakeKleinEdit(dict):
    def __init__(self, **_kwargs):
        super().__init__()

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        reference = Path(kwargs["image_paths"][0])
        assert reference.is_file()
        assert Image.open(reference).mode == "RGB"
        config = SimpleNamespace(num_inference_steps=kwargs["num_inference_steps"])
        for step in range(config.num_inference_steps):
            for callback in list(self.callbacks.in_loop):
                callback.call_in_loop(step, config=config)
        return _FakeResult(Image.new("RGB", (kwargs["width"], kwargs["height"]), "blue"))


def _install_fake_mflux(monkeypatch):
    mflux = types.ModuleType("mflux")
    mflux.__path__ = []
    models = types.ModuleType("mflux.models")
    models.__path__ = []
    flux2 = types.ModuleType("mflux.models.flux2")
    flux2.__path__ = []
    variants = types.ModuleType("mflux.models.flux2.variants")
    variants.Flux2Klein = _FakeKlein
    variants.Flux2KleinEdit = _FakeKleinEdit
    monkeypatch.setitem(sys.modules, "mflux", mflux)
    monkeypatch.setitem(sys.modules, "mflux.models", models)
    monkeypatch.setitem(sys.modules, "mflux.models.flux2", flux2)
    monkeypatch.setitem(sys.modules, "mflux.models.flux2.variants", variants)
    _FakeKlein.init_calls.clear()


def _model_snapshot(tmp_path):
    snapshot = tmp_path / "models--black-forest-labs--FLUX.2-klein-4B" / "snapshots" / "abc"
    (snapshot / "transformer").mkdir(parents=True)
    (snapshot / "vae").mkdir()
    (snapshot / "text_encoder").mkdir()
    header = json.dumps({"tensor": {"dtype": "F32", "shape": [1],
                                    "data_offsets": [0, 4]}}).encode()
    for component in ("transformer", "vae", "text_encoder"):
        (snapshot / component / "model.safetensors").write_bytes(
            len(header).to_bytes(8, "little") + header + b"\0" * 4)
    return snapshot


@pytest.fixture(autouse=True)
def local_snapshot(tmp_path):
    return _model_snapshot(tmp_path)


def test_load_constructs_one_model_and_shares_components(monkeypatch, tmp_path):
    _install_fake_mflux(monkeypatch)
    snapshot = next((tmp_path / "models--black-forest-labs--FLUX.2-klein-4B" / "snapshots").iterdir())
    pipe = MLXKleinPipeline(models_dir=str(tmp_path), quantize=8)

    assert pipe.load() is pipe
    assert pipe.load() is pipe
    assert len(_FakeKlein.init_calls) == 1
    assert Path(_FakeKlein.init_calls[0]["model_path"]) == snapshot
    assert _FakeKlein.init_calls[0]["quantize"] == 8
    assert pipe._model.vae is pipe._edit_model.vae
    assert pipe._model.transformer is pipe._edit_model.transformer
    assert pipe._model.callbacks is pipe._edit_model.callbacks


def test_txt2img_uses_prompt_scaling_seeds_and_serial_variants(monkeypatch, tmp_path):
    _install_fake_mflux(monkeypatch)
    pipe = MLXKleinPipeline(models_dir=str(tmp_path))
    target_size = (17, 33)

    images = pipe.txt2img("a red coin", target_size, variants=3, seeds=[11, 22, 33])

    assert len(images) == 3
    assert all(image.mode == "RGBA" for image in images)
    assert [call["seed"] for call in pipe._model.calls] == [11, 22, 33]
    assert all(call["prompt"].endswith(T2I_SUFFIX) for call in pipe._model.calls)
    width, height = t2i_size(target_size)
    assert [(call["width"], call["height"]) for call in pipe._model.calls] == [
        (width, height),
        (width, height),
        (width, height),
    ]


def test_edit_uses_shared_weights_and_style_prompt(monkeypatch, tmp_path):
    _install_fake_mflux(monkeypatch)
    pipe = MLXKleinPipeline(models_dir=str(tmp_path))
    image = Image.new("RGBA", (8, 4), (50, 60, 70, 255))

    images = pipe.edit_by_instruction("add a hat", image, variants=2, seeds=[7, 8])

    assert len(images) == 2
    assert all(result.mode == "RGBA" for result in images)
    assert [call["seed"] for call in pipe._edit_model.calls] == [7, 8]
    assert all(call["prompt"].endswith(STYLE_SUFFIX) for call in pipe._edit_model.calls)
    assert all(call["image_paths"] and call["image_paths"][0].name == "reference.png"
               for call in pipe._edit_model.calls)
    assert all((call["width"], call["height"]) == (512, 256)
               for call in pipe._edit_model.calls)
    assert pipe._model.vae is pipe._edit_model.vae


def test_progress_covers_serial_variants_and_callback_is_removed(monkeypatch, tmp_path):
    _install_fake_mflux(monkeypatch)
    pipe = MLXKleinPipeline(models_dir=str(tmp_path))
    progress = []

    pipe.txt2img("coin", (32, 32), variants=2, on_progress=progress.append, seeds=[1, 2])

    assert progress[-1] == 1.0
    assert progress == sorted(progress)
    assert pipe._model.callbacks.in_loop == []


def test_progress_cancellation_propagates_and_cleans_callback(monkeypatch, tmp_path):
    _install_fake_mflux(monkeypatch)
    pipe = MLXKleinPipeline(models_dir=str(tmp_path))

    class Cancelled(Exception):
        pass

    def cancel(value):
        if value >= 0.25:
            raise Cancelled("client disconnected")

    with pytest.raises(Cancelled, match="client disconnected"):
        pipe.txt2img("coin", (32, 32), variants=2, on_progress=cancel, seeds=[1, 2])
    assert pipe._model.callbacks.in_loop == []


def test_missing_model_is_an_explicit_setup_error(tmp_path):
    pipe = MLXKleinPipeline(models_dir=str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="not downloaded"):
        pipe._resolve_model_path()


def test_partial_snapshot_and_missing_shard_are_rejected(tmp_path, local_snapshot):
    # A component directory appears before its model download is finished.
    (local_snapshot / "vae" / "model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        MLXKleinPipeline(str(tmp_path))._resolve_model_path()


def test_truncated_safetensor_is_rejected(tmp_path, local_snapshot):
    file = local_snapshot / "transformer" / "model.safetensors"
    file.write_bytes(file.read_bytes()[:-1])
    with pytest.raises(RuntimeError, match="incomplete"):
        MLXKleinPipeline(str(tmp_path))._resolve_model_path()
