"""SANA-WM Stage 2 current video rows with the native anchor."""

from sol3_attn import sol3_attn_with_route
from . import _binding


def attention(q, k, v, *, cfg, **kwargs):
    """Run the sparse branch with the configuration selected by the model controller."""
    out, _ = sol3_attn_with_route(q, k, v, profile="sana", config=cfg, **kwargs)
    return out


def activate(*, counters=None, fusion_modules=(), release_points=(), binding=None, target=None):
    """Use the installed backend and fused kernels during normal Sol³ generation."""
    return _binding.activate("sana", __package__ + ".sana_wm_fusions", release_points,
                             counters=counters, fusion_modules=fusion_modules, binding=binding, target=target)
