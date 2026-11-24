"""Use Sol³ for the video self-attention calls in the SANA-WM Stage 2 Refiner."""

from sol3_integrations import sana_wm


def refine(run_refiner, target, stage1_latents, *args, release_points=(), fusion_modules=(), **kwargs):
    """Pass the Refiner callable and (owner, attribute) for its sparse branch."""
    # Run Stage 1 normally. The Refiner still handles its anchor and dense steps.
    with sana_wm.activate(target=target, release_points=release_points, fusion_modules=fusion_modules):
        return run_refiner(stage1_latents, *args, **kwargs)
