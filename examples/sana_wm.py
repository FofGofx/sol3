"""Use Sol³ for the video self-attention calls in the SANA-WM Stage 2 Refiner."""

from functools import partial

from sol3_attn import get_preset
from sol3_integrations import sana_wm


# For hosts that need to construct their own sparse attention callable.
sparse_attention = partial(sana_wm.attention, cfg=get_preset("sana")["kernel_config"])


def refine(run_refiner, target, stage1_latents, *args, release_points=(), fusion_modules=(), **kwargs):
    """Pass the Refiner callable and (owner, attribute) for its sparse branch."""
    # Run Stage 1 normally. The Refiner still handles its anchor and dense steps.
    with sana_wm.activate(target=target, release_points=release_points, fusion_modules=fusion_modules):
        return run_refiner(stage1_latents, *args, **kwargs)
