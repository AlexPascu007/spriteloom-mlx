"""Keep source latents outside an inpaint mask at every MFLUX step.

The callback changes the array supplied by MFLUX in place. The pinned
FLUX.2 loop evaluates that same array before its next transformer call.
This applies the flow-matching source/noise blend used by the Diffusers
Klein inpaint pipeline, while retaining MFLUX's model and scheduler code.
"""
from contextlib import contextmanager

import numpy as np
from PIL import Image


def packed_mask(mask, width, height):
    """Bilinear sampling at latent pixel centers, without an antialias filter."""
    source = (np.asarray(mask.convert("L"), dtype=np.float32) >= 128).astype(np.float32)
    xs = np.clip((np.arange(width) + 0.5) * mask.width / width - 0.5, 0, mask.width - 1)
    ys = np.clip((np.arange(height) + 0.5) * mask.height / height - 0.5, 0, mask.height - 1)
    x0, y0 = xs.astype(int), ys.astype(int)
    x1, y1 = np.minimum(x0 + 1, mask.width - 1), np.minimum(y0 + 1, mask.height - 1)
    ax, ay = (xs - x0)[None, :], (ys - y0)[:, None]
    top = source[y0[:, None], x0] * (1 - ax) + source[y0[:, None], x1] * ax
    bottom = source[y1[:, None], x0] * (1 - ax) + source[y1[:, None], x1] * ax
    return ((1 - ay) * top + ay * bottom).astype(np.float32).reshape(1, -1, 1)


class LatentMaskCallback:
    def __init__(self, clean, mask):
        self.clean = clean
        self.mask = mask
        self.noise = None

    def call_before_loop(self, *, latents, **kwargs):
        import mlx.core as mx
        if latents.shape != self.clean.shape:
            raise ValueError("inpaint source and generated latents have different shapes")
        self.noise = mx.array(latents)
        mx.eval(self.noise, self.clean, self.mask)

    def call_in_loop(self, *, t, latents, config, **kwargs):
        sigma = config.scheduler.sigmas[int(t) + 1].astype(latents.dtype)
        source = (1 - sigma) * self.clean + sigma * self.noise
        latents[:] = (1 - self.mask) * source + self.mask * latents


@contextmanager
def latent_mask(model, reference_path, mask):
    """Install the mask callback for this request and remove it on all exits."""
    import mlx.core as mx
    from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers

    clean, _ = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
        vae=model.vae, tiling_config=model.tiling_config,
        image_paths=[reference_path], batch_size=1)
    weights = mx.array(packed_mask(mask, mask.width // 16, mask.height // 16),
                       dtype=clean.dtype)
    callback = LatentMaskCallback(clean, weights)
    model.callbacks.register(callback)
    try:
        yield callback
    finally:
        for name in ("before_loop", "in_loop"):
            callbacks = getattr(model.callbacks, name)
            if callback in callbacks:
                callbacks.remove(callback)


def transparent_overlay(image, mask, content_size):
    from PIL import ImageChops
    from server.postprocess import remove_background

    width, height = content_size
    full = image.crop((0, 0, width, height)).convert("RGBA")
    cleaned = remove_background(full, tolerance=16)
    keep = ImageChops.multiply(mask.crop((0, 0, width, height)), cleaned.getchannel("A"))
    output = Image.new("RGBA", content_size, (0, 0, 0, 0))
    output.paste(cleaned, (0, 0), keep)
    return output
