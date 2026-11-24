"""H3-Base joint attention with dense prefix-query rows."""

from contextlib import contextmanager

from sol3_attn import sol3_attn_with_route
from . import _binding


def attention(q, k, v, *, cfg, **kwargs):
    """Run the sparse branch with the configuration selected by the model controller."""
    out, _ = sol3_attn_with_route(q, k, v, profile="h3", config=cfg, **kwargs)
    return out


@contextmanager
def activate(*, counters=None, fusion_modules=(), release_points=(), binding=None, target=None):
    """Use the installed backend and fused kernels during normal Sol³ generation."""
    from .h3_workspace_guard import existing_guard_scope

    with _binding.activate("h3", __package__ + ".h3_base_fusions", release_points,
                           counters=counters, fusion_modules=fusion_modules, binding=binding, target=target) as scope:
        with existing_guard_scope():
            yield scope
