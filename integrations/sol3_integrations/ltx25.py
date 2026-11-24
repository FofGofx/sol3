"""LTX-2.5 Stage 1 video attention after the native capacity check."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from importlib import import_module

from sol3_attn import sol3_attn_with_route
from . import _binding


def attention(q, k, v, *, cfg, **kwargs):
    """Run the sparse branch with the configuration selected by the model controller."""
    out, _ = sol3_attn_with_route(q, k, v, profile="ltx", config=cfg, **kwargs)
    return out


@contextmanager
def activate(*, counters=None, fusion_modules=(), release_points=(), stage1_class=None, binding=None, target=None):
    """Bind only the selected Stage 1 video call. The native capacity probe uses its own factory."""
    if target is not None:
        if stage1_class is not None:
            raise ValueError("Pass either a Stage 1 sparse-call target or stage1_class")
        with _binding.activate("ltx", __package__ + ".ltx25_fusions", release_points, target=target,
                               counters=counters, fusion_modules=fusion_modules, binding=binding) as scope:
            yield scope
        return
    if stage1_class is None:
        stage1_class = import_module("ltx_sol_attn").Stage1SolAttention
    in_stage1 = ContextVar("ltx_video_self_attention", default=False)
    with _binding.activate("ltx", __package__ + ".ltx25_fusions", release_points, enabled=in_stage1.get,
                          counters=counters, fusion_modules=fusion_modules, binding=binding) as scope:
        original = stage1_class._run_sol

        @wraps(original)
        def run_stage1(stage, *args, **kwargs):
            # The native Stage 1 controller can construct a new configured
            # kernel inside _sol_api. Intercept that leaf after configuration.
            token = in_stage1.set(True)
            try:
                return original(stage, *args, **kwargs)
            finally:
                in_stage1.reset(token)

        _binding._replace(scope.stack, stage1_class, "_run_sol", run_stage1)
        yield scope
