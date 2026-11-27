"""Connect workspace release to a two-stage LTX pipeline."""

from functools import partial

from sol3_attn import get_preset
from sol3_integrations import ltx25


# For hosts that need to construct their own sparse attention callable.
sparse_attention = partial(ltx25.attention, cfg=get_preset("ltx")["kernel_config"])


def generate(entry, runner, *args, release_points=(), fusion_modules=(), **kwargs):
    """Use the entry module's original Stage 1 release before Stage 2 proceeds."""
    with ltx25.activate(release_points=release_points, fusion_modules=fusion_modules) as installed:
        installed.release_after(entry, "release_workspace")
        return runner.generate(*args, **kwargs)
