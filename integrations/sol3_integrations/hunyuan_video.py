"""HunyuanVideo joint attention with the original protected text suffix."""

from sol3_attn import sol3_attn_with_route
from . import _binding


def attention(q, k, v, *, cfg, **kwargs):
    """Run the sparse branch with the configuration selected by the model controller."""
    out, _ = sol3_attn_with_route(q, k, v, profile="hy", config=cfg, **kwargs)
    return out


def activate(*, counters=None, fusion_modules=(), release_points=(), binding=None, target=None):
    """Use the installed backend and fused kernels during normal Sol³ generation."""
    return _binding.activate("hy", __package__ + ".hunyuan_video_fusions", release_points,
                             counters=counters, fusion_modules=fusion_modules, binding=binding, target=target)
