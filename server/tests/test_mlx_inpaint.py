from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from server.mlx_inpaint import LatentMaskCallback, packed_mask, transparent_overlay


def test_mask_sampling_preserves_canvas_alignment():
    mask = Image.new("L", (32, 16))
    mask.paste(255, (16, 0, 32, 16))
    np.testing.assert_array_equal(packed_mask(mask, 2, 1), [[[0], [1]]])


def test_mask_locks_source_at_each_step_and_final_clean_latents():
    mx = pytest.importorskip("mlx.core")
    from mflux.callbacks.callback_registry import CallbackRegistry

    clean = mx.array([[[2.0], [3.0]]])
    latents = mx.array([[[10.0], [20.0]]])
    mask = mx.array([[[0.0], [1.0]]])
    config = SimpleNamespace(scheduler=SimpleNamespace(sigmas=mx.array([1.0, 0.5, 0.0])),
                             time_steps=[0, 1])
    registry = CallbackRegistry()
    callback = LatentMaskCallback(clean, mask)
    registry.register(callback)
    context = registry.start(seed=1, prompt="hat", config=config)
    context.before_loop(latents)
    latents = latents + 100
    context.in_loop(0, latents)
    mx.eval(latents)
    # Same latents object used by the pinned MFLUX loop is changed by callback.
    np.testing.assert_allclose(np.array(latents), [[[6.0], [120.0]]])
    latents = latents + 100
    context.in_loop(1, latents)
    mx.eval(latents)
    np.testing.assert_allclose(np.array(latents), [[[2.0], [220.0]]])
    np.testing.assert_array_equal(np.array(callback.noise), [[[10.0], [20.0]]])


def test_overlay_is_transparent_outside_mask():
    image = Image.new("RGBA", (32, 32), (255, 0, 255, 255))
    image.paste((120, 50, 20, 255), (8, 8, 24, 24))
    mask = Image.new("L", (32, 32))
    mask.paste(255, (8, 8, 24, 24))
    result = transparent_overlay(image, mask, (32, 32))
    assert result.mode == "RGBA"
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((12, 12))[3] == 255
