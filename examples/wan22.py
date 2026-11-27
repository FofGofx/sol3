"""Enable Sol³ around Wan2.2 generation with a shared sparse attention target."""

from functools import partial

from sol3_attn import get_preset
from sol3_integrations import wan22


# For hosts that need to construct their own sparse attention callable.
sparse_attention = partial(wan22.attention, cfg=get_preset("wan")["kernel_config"])


def generate(pipeline, target, *args, application=None, release_points=(), fusion_modules=(), **kwargs):
    """Pass (owner, attribute) for the sparse call shared by the two experts."""
    # Both experts keep their own routing configuration and dense warmup steps.
    with wan22.activate(target=target, application=application, release_points=release_points,
                        fusion_modules=fusion_modules):
        return pipeline(*args, **kwargs)
