"""Select the model backend without importing an unused GPU framework."""
from server import config


def create_pipeline():
    backend = config.load_backend()
    if backend == "mlx":
        from server.instruct_mlx import MLXKleinPipeline
        return MLXKleinPipeline(models_dir=config.load_models_dir(),
                                quantize=config.load_mlx_quantize())
    from server.instruct import KleinPipeline
    return KleinPipeline(models_dir=config.load_models_dir(),
                         mode=config.load_vram_mode())
