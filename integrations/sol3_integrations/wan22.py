"""Wan2.2 high- and low-noise video self-attention."""

from contextlib import contextmanager
from sol3_attn import sol3_attn_with_route
from . import _binding


def attention(q, k, v, *, cfg, **kwargs):
    """Run the sparse branch with the configuration selected by the model controller."""
    out, _ = sol3_attn_with_route(q, k, v, profile="wan", config=cfg, **kwargs)
    return out


@contextmanager
def activate(*, counters=None, fusion_modules=(), release_points=(), binding=None, application=None, target=None):
    """Use the installed backend and fused kernels during normal Sol³ generation."""
    with _binding.activate("wan", __package__ + ".wan22_fusions", release_points,
                           counters=counters, fusion_modules=fusion_modules, binding=binding, target=target) as scope:
        if application is not None:
            scope.release_after(application if isinstance(application, type) else type(application), "warmup")
        yield scope
