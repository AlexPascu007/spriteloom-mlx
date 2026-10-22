"""MLX adapter for the FLUX.2 Klein model.

MFLUX loads weights when its model object is constructed.  This adapter keeps
that work behind :meth:`MLXKleinPipeline.load`, and gives the server the same
small interface as the Diffusers pipeline in ``server.instruct``.
"""
from __future__ import annotations

import pathlib
import json
import random
import tempfile
from collections.abc import Mapping
from typing import Any

from PIL import Image

from server.instruct import (  # no MLX or torch import is needed for the server tests
    STYLE_SUFFIX,
    T2I_SUFFIX,
    INPAINT_SUFFIX,
    KleinPipeline as _InputHelpers,
    safetensors_expected_size,
    t2i_size,
)

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
MODEL_FOLDER = "models--black-forest-labs--FLUX.2-klein-4B"
# The adapter targets the API reviewed at this MFLUX revision.
MFLUX_COMMIT = "0fea52bada4ade619f1cdb7ae26087b32d87dd89"
MFLUX_STEPS = 8  # keep SpriteLoom's tuned schedule on both backends
GUIDANCE = 1.0


class _ProgressCallback:
    """Translate one MFLUX denoising callback into a request progress value."""

    def __init__(self, on_progress, variant_index: int, variants: int):
        self._on_progress = on_progress
        self._variant_index = variant_index
        self._variants = variants

    def call_in_loop(self, t, *args, config=None, **_kwargs):
        if config is None:
            config = next((arg for arg in args if hasattr(arg, "num_inference_steps")), None)
        steps = getattr(config, "num_inference_steps", None) if config is not None else None
        if steps is None:
            steps = MFLUX_STEPS
        steps = max(int(steps), 1)
        try:
            completed = int(t) + 1
        except (TypeError, ValueError):
            completed = 1
        completed = min(max(completed, 0), steps)
        value = (self._variant_index + completed / steps) / self._variants
        # Do not catch exceptions from the caller.  The server uses one to cancel
        # a disconnected request, and MFLUX must stop immediately in that case.
        self._on_progress(min(max(float(value), 0.0), 1.0))


def _remove_callback(registry, callback) -> None:
    """Remove a request callback from every callback list it could enter."""
    if registry is None or callback is None:
        return
    for name in ("in_loop", "before_loop", "after_loop", "interrupt"):
        callbacks = getattr(registry, name, None)
        if callbacks is None:
            continue
        try:
            while callback in callbacks:
                callbacks.remove(callback)
        except (AttributeError, TypeError, ValueError):
            # MFLUX uses lists.  This keeps the adapter harmless with a small
            # test double or a future registry that exposes an immutable view.
            continue


def _image_from_result(result: Any) -> Image.Image:
    """Return an MFLUX result as an RGBA Pillow image."""
    image = getattr(result, "image", result)
    if not isinstance(image, Image.Image):
        raise TypeError("MFLUX returned a result without a Pillow image")
    return image.convert("RGBA")


def _component_directory(path: pathlib.Path) -> bool:
    """Check all component files before selecting a local model snapshot."""
    if not path.is_dir():
        return False
    try:
        for name in ("vae", "transformer", "text_encoder"):
            component = path / name
            files = list(component.glob("*.safetensors"))
            if not files:
                return False
            for index in component.glob("*.safetensors.index.json"):
                shards = set(json.loads(index.read_text())["weight_map"].values())
                if not shards or any(not (component / shard).is_file() for shard in shards):
                    return False
            for file in files:
                if not file.is_file() or file.stat().st_size < safetensors_expected_size(file):
                    return False
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return False
    return True


def _newest_directory(paths: list[pathlib.Path]) -> pathlib.Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


class _SharedEditModel:
    """Create an edit-model object without running its weight initializer.

    MFLUX's two public Klein classes both call ``Flux2Initializer.init`` in
    their constructors.  The edit class only needs a different set of methods;
    its VAE, transformer, tokenizer, and callback registry can all be references
    to the already-loaded text-to-image model.  A thin subclass preserves the
    edit class implementation while copying only references, never arrays.
    """

    @staticmethod
    def create(edit_class, source):
        class SharedEdit(edit_class):
            def __init__(self, shared):
                if isinstance(shared, Mapping):
                    # MLX's nn.Module overrides ``dict.update`` to update
                    # existing parameter trees.  We need a shallow mapping
                    # copy here, because the edit view has no own parameters.
                    dict.update(self, shared)
                self.__dict__.update(getattr(shared, "__dict__", {}))
                # A small fake model may not inherit dict/nn.Module.  Its
                # ordinary attributes still need to be visible on the view.
                if not isinstance(shared, Mapping):
                    self.__dict__.update(vars(shared))

        return SharedEdit(source)


class MLXKleinPipeline:
    """SpriteLoom's MFLUX-backed FLUX.2 Klein pipeline.

    ``models_dir`` points to the existing local Hugging Face cache.  The
    adapter resolves a snapshot before constructing MFLUX, so a missing model
    raises from the local loader instead of starting a network download.
    """

    def __init__(self, models_dir: str = "models", quantize: int | None = 8):
        self.models_dir = pathlib.Path(models_dir).expanduser()
        self.quantize = quantize
        self._model = None
        self._edit_model = None
        # Keep names close to the Diffusers pipeline for diagnostics and tests.
        self._pipe = None
        self._edit_pipe = None
        self._model_path: pathlib.Path | None = None

    @staticmethod
    def variant_seeds(seed, variants):
        """Return one reproducible 32-bit seed for each requested variant."""
        if seed is None:
            return [random.randrange(2**32) for _ in range(variants)]
        return [(seed + index) % 2**32 for index in range(variants)]

    def _resolve_model_path(self) -> pathlib.Path:
        """Find an existing local snapshot without invoking a downloader."""
        root = self.models_dir
        if not root.is_dir():
            raise RuntimeError("model is not downloaded or incomplete; run setup-macos.sh first")

        # SpriteLoom's setup uses a custom HF cache root.
        cache_model = root / MODEL_FOLDER
        snapshots = cache_model / "snapshots"
        snapshot = _newest_directory([path for path in snapshots.glob("*") if _component_directory(path)])
        if snapshot is not None:
            return snapshot
        if _component_directory(cache_model):
            return cache_model

        # Also accept a direct snapshot or an mflux-save directory.  This makes
        # the adapter useful with an already converted local checkpoint.
        if _component_directory(root):
            return root
        children = [path for path in root.iterdir() if _component_directory(path)]
        found = _newest_directory(children)
        if found is None:
            raise RuntimeError("model is not downloaded or incomplete; run setup-macos.sh first")
        return found

    def load(self):
        """Load one resident MFLUX model and a shared edit view."""
        if self._model is not None:
            return self

        model_path = self._resolve_model_path()

        # Keep MLX/MFLUX imports lazy.  CUDA installations and unit tests that
        # only exercise protocol code must not import either framework.
        from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit

        from server import models
        models.set_load_progress(0.1, "Loading MLX weights", ceiling=0.95)
        model = Flux2Klein(model_path=str(model_path), quantize=self.quantize)
        edit_model = _SharedEditModel.create(Flux2KleinEdit, model)
        self._model_path = model_path
        self._model = model
        self._edit_model = edit_model
        self._pipe = model
        self._edit_pipe = edit_model
        return self

    def unload(self):
        """Drop both views so the model manager can release unified memory."""
        self._edit_pipe = None
        self._edit_model = None
        self._pipe = None
        self._model = None
        self._model_path = None

    def _run_one(self, model, variant_index: int, variants: int, on_progress, **kwargs):
        callback = (
            _ProgressCallback(on_progress, variant_index, variants)
            if on_progress is not None
            else None
        )
        registry = getattr(model, "callbacks", None)
        if callback is not None and registry is not None and hasattr(registry, "register"):
            registry.register(callback)
        try:
            return model.generate_image(**kwargs)
        finally:
            _remove_callback(registry, callback)

    @staticmethod
    def _seeds(seeds, variants):
        return list(seeds) if seeds else MLXKleinPipeline.variant_seeds(None, variants)

    def txt2img(self, prompt, target_size, variants=4, on_progress=None, seeds=None):
        """Generate serial variants at the server's working resolution."""
        if variants <= 0:
            return []
        self.load()
        width, height = t2i_size(target_size)
        selected_seeds = self._seeds(seeds, variants)
        images = []
        for index in range(variants):
            result = self._run_one(
                self._model,
                index,
                variants,
                on_progress,
                seed=selected_seeds[index],
                prompt=prompt + T2I_SUFFIX,
                num_inference_steps=MFLUX_STEPS,
                height=height,
                width=width,
                guidance=GUIDANCE,
                scheduler="flow_match_euler_discrete",
            )
            images.append(_image_from_result(result))
        return images

    def edit_by_instruction(self, instruction, image, variants=4, on_progress=None, seeds=None):
        """Edit a reference image with MFLUX's image-conditioned model."""
        if variants <= 0:
            return []
        self.load()
        # Keep the current SpriteLoom input policy: nearest-neighbour integer
        # scaling, key-colour flattening, and a 16-pixel model canvas.
        prepared, (content_width, content_height) = _InputHelpers._prep_input(image)
        selected_seeds = self._seeds(seeds, variants)
        images = []
        with tempfile.TemporaryDirectory(prefix="spriteloom-mlx-edit-") as temp_dir:
            reference_path = pathlib.Path(temp_dir) / "reference.png"
            prepared.save(reference_path)
            for index in range(variants):
                result = self._run_one(
                    self._edit_model,
                    index,
                    variants,
                    on_progress,
                    seed=selected_seeds[index],
                    prompt=instruction + STYLE_SUFFIX,
                    num_inference_steps=MFLUX_STEPS,
                    height=prepared.height,
                    width=prepared.width,
                    guidance=GUIDANCE,
                    image_paths=[reference_path],
                    scheduler="flow_match_euler_discrete",
                )
                edited = _image_from_result(result)
                images.append(edited.crop((0, 0, content_width, content_height)))
        return images

    def inpaint(self, prompt, image, mask, variants=4, on_progress=None, seeds=None):
        """Lock unselected latents at each step and return a transparent layer."""
        if variants <= 0:
            return []
        from server.mlx_inpaint import latent_mask, transparent_overlay
        self.load()
        prepared, content_size = _InputHelpers._prep_input(image)
        prepared_mask = _InputHelpers._prep_mask(mask, prepared.size, content_size)
        selected_seeds = self._seeds(seeds, variants)
        images = []
        with tempfile.TemporaryDirectory(prefix="spriteloom-mlx-inpaint-") as temp_dir:
            reference_path = pathlib.Path(temp_dir) / "reference.png"
            prepared.save(reference_path)
            with latent_mask(self._edit_model, reference_path, prepared_mask):
                for index in range(variants):
                    result = self._run_one(
                        self._edit_model, index, variants, on_progress,
                        seed=selected_seeds[index],
                        prompt=prompt + STYLE_SUFFIX + INPAINT_SUFFIX,
                        num_inference_steps=MFLUX_STEPS,
                        height=prepared.height, width=prepared.width,
                        guidance=GUIDANCE, image_paths=[reference_path],
                        scheduler="flow_match_euler_discrete")
                    images.append(transparent_overlay(
                        _image_from_result(result), prepared_mask, content_size))
        return images
