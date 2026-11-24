"""FastH3 joint attention with protected text and audio prefix tokens."""

from sol3_attn import sol3_attn_with_route
from . import _binding


def attention(q, k, v, *, config, prefix_tokens=0, dense=None):
    """Run the sparse branch with the configuration selected by the model controller."""
    return sol3_attn_with_route(q, k, v, profile="fast", config=config, prefix_tokens=prefix_tokens, dense=dense)


def activate(*, counters=None, fusion_modules=(), release_points=(), binding=None, target=None):
    """Use the installed backend and fused kernels during normal Sol³ generation."""
    return _binding.activate("fast", __package__ + ".fasth3_fusions", release_points,
                             counters=counters, fusion_modules=fusion_modules, binding=binding, target=target)
