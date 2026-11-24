"""Run the FastH3 SM120 joint exact/representative kernel."""

import math

import torch
import triton

from ...profiles import PROFILES
from ..workspace import GIB, Workspace, available_memory, choose_plan, count_launch
from . import fused_attention as compact
from ._providers import sol3_compact, sol3_kv_covariance
from .routing import DEFAULT_COVARIANCE, make_route


_PROFILE = PROFILES["fast"]
_WORKSPACE = Workspace("fast", __package__ + ".")


def _head_slice(route, start, end, heads, blocks, dimension):
    selected = dict(route, b=1, h=end - start)
    for key in ("flags", "bits"):
        selected[key] = route[key].view(heads, blocks, blocks)[start:end]
    for key in ("kc", "vm"):
        selected[key] = route[key].view(heads, blocks, dimension)[start:end]
    return selected


def forward(q, k, v, *, counters=None, config=None, prefix_tokens=0, dense=None):
    q, k, v = (x.contiguous() for x in (q, k, v))
    batch, tokens, heads, dimension = q.shape
    if not 0 <= prefix_tokens < tokens or (prefix_tokens and dense is None):
        raise ValueError("A protected FastH3 prefix requires the native dense-query callback")
    cfg = dict(config or {})
    gamma = float(cfg.pop("cov_gamma", DEFAULT_COVARIANCE))
    if gamma not in (0.0, 0.25, 0.5, 1.0, 1.5):
        raise ValueError("Unsupported FastH3 covariance coefficient")

    with _WORKSPACE.call(q, counters) as state:
        plan = choose_plan(
            tokens, heads, dimension, available_memory(q),
            preferred_heads=_PROFILE.long_heads, use_score_cache=False, budget_limit=24 * GIB,
        )
        route = make_route(q, k, v, cfg, prefix_tokens)
        covariance = sol3_kv_covariance.global_covariance(k, v, prefix_tokens) if gamma else None
        blocks = triton.cdiv(tokens, 64)
        out = torch.empty_like(q)
        chunk = plan.head_chunk if plan.minimum_fits else 4
        bk, warps = _PROFILE.joint_launch if state.optimized else (128, 8)
        for start in range(0, heads, chunk):
            end = min(heads, start + chunk)
            part = _head_slice(route, start, end, heads, blocks, dimension)
            cov = covariance.view(heads, dimension, dimension)[start:end] if covariance is not None else None
            qs, ks, vs = (x[:, :, start:end] for x in (q, k, v))
            if not plan.minimum_fits:
                for name in ("flags", "bits", "kc", "vm"):
                    part[name] = part[name].contiguous()
                out[:, :, start:end] = sol3_compact.attention(
                    qs.contiguous(), ks.contiguous(), vs.contiguous(), part,
                    prefix_tokens=prefix_tokens, dense=dense, bk=128, warps=8, stages=1,
                    covariance=cov, cov_gamma=gamma,
                )
                count_launch("compact_fallback")
                continue
            ids, count = compact.prepare(qs, part)
            compact.packed_attention[(blocks, batch * (end - start))](
                qs, ks, vs, part["kc"], part["vm"], part["flags"], ids, count,
                out[:, :, start:end], qs, qs, cov if cov is not None else qs,
                tokens, blocks, end - start, dimension, prefix_tokens,
                dimension ** -0.5 * math.log2(math.e), bk, False, False,
                GAMMA=gamma, QST=qs.stride(1), KST=ks.stride(1),
                VST=vs.stride(1), OST=out.stride(1), num_warps=warps, num_stages=1,
            )
            count_launch("joint")
            count_launch(f"joint_bk{bk}_warps{warps}")
        if prefix_tokens and plan.minimum_fits:
            for start in range(0, heads, 4):
                end = min(heads, start + 4)
                out[:, :prefix_tokens, start:end] = dense(
                    q[:, :prefix_tokens, start:end], k[:, :, start:end], v[:, :, start:end],
                )
        return out, route


def release():
    return _WORKSPACE.release()
