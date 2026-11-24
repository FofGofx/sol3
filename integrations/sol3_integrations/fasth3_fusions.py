#!/usr/bin/env python3
"""Optional FastH3 operator fusions inspired by the SGLang KWL runtime.

The kernels fuse normalization, partial rotary embeddings, SwiGLU and
per-row modulation. Each fusion can be enabled independently and has
an eager fallback.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - environment without triton
    HAVE_TRITON = False


# --------------------------------------------------------------------------------------------
# 1. residual + gate[idx] * branch, then RMSNorm, then scale[idx]/shift[idx] modulation
# --------------------------------------------------------------------------------------------
# Eager runs this as thirteen passes over the activation: gather the gate row, fuse the residual,
# read it back to normalize, gather scale, gather shift, read all three to modulate. The three
# gathers each materialize a full (rows, hidden) tensor out of a table with a handful of rows,
# which is the specific waste MiniMax-H3 adds over LTX-2 — its modulation is per row, not per batch.
#
# One pass instead: the row's residual, branch output and gate row come in, `hidden` is written once
# because the next half-block needs it as its own residual, and the normalized-and-modulated value
# is written from the same registers.

if HAVE_TRITON:

    @triton.jit
    def _residual_gate_rmsnorm_modulate_kernel(
        hidden_out_ptr, normed_out_ptr,
        residual_ptr, branch_ptr, weight_ptr,
        gate_ptr, scale_ptr, shift_ptr, index_ptr,
        n_cols, n_index, eps,
        stride_row, stride_table_row,
        BLOCK: tl.constexpr,
        lora_ptr, HAS_LORA: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        offset = row * stride_row + cols

        # The modulation tables are `chunk(6, dim=-1)` views of one wide projection, so their row
        # stride is 6*hidden, not hidden. Assuming contiguity here reads a different table's row.
        # `adaln_indices` has one entry per *sequence* row and eager broadcasts it over the batch,
        # so a flattened (batch*seq) row has to wrap. Reading `index_ptr + row` directly walks off
        # the end of the index tensor for batch > 1 and gathers arbitrary table rows.
        table_row = tl.load(index_ptr + (row % n_index))
        table_offset = table_row * stride_table_row + cols

        residual = tl.load(residual_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        if HAS_LORA:
            delta = tl.load(lora_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            # Match the materialized PEFT sum before the existing fused math.
            branch = (branch + delta).to(branch_ptr.dtype.element_ty).to(tl.float32)
        gate = tl.load(gate_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)

        hidden = residual + gate * branch
        tl.store(hidden_out_ptr + offset, hidden.to(hidden_out_ptr.dtype.element_ty), mask=mask)

        # RMSNorm over the row, in the same registers that already hold it.
        variance = tl.sum(hidden * hidden, axis=0) / n_cols
        normed = hidden * tl.math.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = normed * weight

        scale = tl.load(scale_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        out = normed * (1.0 + scale) + shift
        tl.store(normed_out_ptr + offset, out.to(normed_out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_modulate_kernel(
        out_ptr, x_ptr, weight_ptr, scale_ptr, shift_ptr, index_ptr,
        n_cols, n_index, eps, stride_row, stride_table_row,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        offset = row * stride_row + cols
        # See the residual variant for both notes: non-contiguous chunk views, and an index vector
        # that is per sequence row and must wrap over the batch.
        table_offset = tl.load(index_ptr + (row % n_index)) * stride_table_row + cols

        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / n_cols
        normed = x * tl.math.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = normed * weight

        scale = tl.load(scale_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offset, (normed * (1.0 + scale) + shift).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_kernel(out_ptr, x_ptr, n_cols, stride_in_row, stride_out_row, BLOCK: tl.constexpr, lora_ptr, HAS_LORA: tl.constexpr):
        # A 15-second unsharded FFN exceeds 2**31 input elements.
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        value = tl.load(x_ptr + row * stride_in_row + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + row * stride_in_row + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_LORA:
            delta_value = tl.load(lora_ptr + row * stride_in_row + cols, mask=mask, other=0.0).to(tl.float32)
            delta_gate = tl.load(lora_ptr + row * stride_in_row + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
            value = (value + delta_value).to(x_ptr.dtype.element_ty).to(tl.float32)
            gate = (gate + delta_gate).to(x_ptr.dtype.element_ty).to(tl.float32)
        out = value * (gate * tl.sigmoid(gate))
        tl.store(out_ptr + row * stride_out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _row_addressable(table: torch.Tensor) -> torch.Tensor:
    """A table the kernel can address with one row stride.

    The three modulation tables come out of `chunk(6, dim=-1)`, so each is a strided view whose row
    stride is 6*hidden while its column stride is 1. That is exactly addressable as
    `row * stride(0) + col`, so no copy is needed. Anything with a column stride other than 1 is
    made contiguous instead of being silently mis-indexed.
    """
    return table if table.stride(-1) == 1 else table.contiguous()


def _warps_for(block: int) -> int:
    """A 5376-wide row held in registers needs the warps to spread it, or it spills to local."""
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    return 4


def fused_rmsnorm_modulate(x, weight, scale, shift, index, eps):
    """RMSNorm then `* (1 + scale[index]) + shift[index]`, one pass.

    `scale` and `shift` are chunk views with a row stride of 6*hidden, and `x` may itself be a
    context-parallel shard, so both strides are read rather than assumed.
    """
    cols = x.shape[-1]
    # `reshape` only promises the shape: it can hand back a view whose last-dim stride is not 1,
    # and the kernel uses one row stride for both the input and the freshly allocated output.
    flat = x.reshape(-1, cols).contiguous()
    rows = flat.shape[0]
    scale, shift = _row_addressable(scale), _row_addressable(shift)
    out = torch.empty_like(flat)
    _rmsnorm_modulate_kernel[(rows,)](
        out, flat, weight, scale, shift, index, cols, index.numel(), eps,
        flat.stride(0), scale.stride(0),
        BLOCK=_next_pow2(cols), num_warps=_warps_for(_next_pow2(cols)),
    )
    return out.view_as(x)


def fused_residual_gate_rmsnorm_modulate(residual, branch, gate, weight, scale, shift, index, eps, lora=None):
    """`hidden = residual + gate[index] * branch`, then RMSNorm+modulate. Returns both."""
    cols = residual.shape[-1]
    res_flat = residual.reshape(-1, cols).contiguous()
    br_flat = branch.reshape(-1, cols).contiguous()
    delta = _lora_input(branch, lora)
    if delta is not None:
        delta = delta.reshape(-1, cols).contiguous()
    rows = res_flat.shape[0]
    gate, scale, shift = _row_addressable(gate), _row_addressable(scale), _row_addressable(shift)
    hidden = torch.empty_like(res_flat)
    normed = torch.empty_like(res_flat)
    _residual_gate_rmsnorm_modulate_kernel[(rows,)](
        hidden, normed, res_flat, br_flat, weight, gate, scale, shift, index,
        cols, index.numel(), eps, res_flat.stride(0), gate.stride(0),
        lora_ptr=delta, HAS_LORA=delta is not None,
        BLOCK=_next_pow2(cols), num_warps=_warps_for(_next_pow2(cols)),
    )
    return hidden.view_as(residual), normed.view_as(residual)


def fused_swiglu(x, lora=None):
    """`value * silu(gate)` over a `(..., 2F)` tensor, one read of 2F and one write of F."""
    delta = _lora_input(x, lora)
    cols = x.shape[-1] // 2
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    if delta is not None:
        delta = delta.reshape_as(flat).contiguous()
    out = torch.empty(flat.shape[0], cols, dtype=x.dtype, device=x.device)
    _swiglu_kernel[(flat.shape[0],)](
        out, flat, cols, flat.stride(0), out.stride(0), BLOCK=_next_pow2(cols), num_warps=_warps_for(_next_pow2(cols)),
        lora_ptr=delta, HAS_LORA=delta is not None,
    )
    return out.view(*x.shape[:-1], cols)


# --------------------------------------------------------------------------------------------
# 2. qk-norm + partial rotary
# --------------------------------------------------------------------------------------------
# The reference's fused qknorm+rope rotates the whole head, which MiniMax-H3 does not: it rotates
# the leading `rotary_dim` channels and passes the rest through. Eager pays for that twice over —
# `cat((-x2, x1))` materializes the rotated half and `cat((rotary, pass))` materializes the whole
# head again, plus a `.contiguous()`. Six passes over q and the same over k, per block.

if HAVE_TRITON:

    @triton.jit
    def _qknorm_partial_rope_kernel(
        out_ptr, x_ptr, weight_ptr, cos_ptr, sin_ptr,
        head_dim, rotary_dim, half_dim, heads, seq, eps,
        BLOCK: tl.constexpr,
    ):
        # One program per (token, head): the RMS is over head_dim, and rotary pairs channel j with
        # j + half_dim inside the rotary span. cos/sin are shared across heads and indexed by the
        # position within the sequence, so the batch stride has to be divided out.
        pid = tl.program_id(0)
        token = (pid // heads) % seq
        cols = tl.arange(0, BLOCK)
        mask = cols < head_dim
        base = pid * head_dim

        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / head_dim
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = x * tl.math.rsqrt(variance + eps) * weight

        in_rotary = cols < rotary_dim
        in_first_half = cols < half_dim
        # partner channel: j+half within the first half, j-half within the second
        partner = tl.where(in_first_half, cols + half_dim, cols - half_dim)
        partner_value = tl.load(x_ptr + base + partner, mask=in_rotary, other=0.0).to(tl.float32)
        partner_variance = variance  # same row, so the same normalizer applies
        partner_weight = tl.load(weight_ptr + partner, mask=in_rotary, other=0.0).to(tl.float32)
        partner_normed = partner_value * tl.math.rsqrt(partner_variance + eps) * partner_weight
        # rotate_half: the first half pairs with -second, the second with +first
        rotated = tl.where(in_first_half, -partner_normed, partner_normed)

        cos = tl.load(cos_ptr + token * rotary_dim + cols, mask=in_rotary, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + token * rotary_dim + cols, mask=in_rotary, other=0.0).to(tl.float32)
        out = tl.where(in_rotary, normed * cos + rotated * sin, normed)
        tl.store(out_ptr + base + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def fused_qknorm_rope(x, weight, cos, sin, eps):
    """RMSNorm per head then partial rotary, for `(B, seq, heads, head_dim)`.

    Rotation tables retain their input precision. With FP32 tables and BF16 hidden
    states, rounding can differ from the eager implementation, which casts the tables to BF16.
    """
    batch, seq, heads, head_dim = x.shape
    rotary_dim = cos.shape[-1]
    # Guards for configurations MiniMax-H3 does not use but a future one might. Without the first,
    # the partner load is masked by `in_rotary` rather than by `cols < head_dim` and would read into
    # the next (token, head) row; without the second, `rotary_dim // 2` truncates and the pairing
    # silently desyncs from eager's `chunk(2, -1)`.
    if rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"fused qknorm+rope needs an even rotary_dim no wider than head_dim, "
            f"got rotary_dim={rotary_dim}, head_dim={head_dim}"
        )
    flat = x.reshape(-1, head_dim).contiguous()
    out = torch.empty_like(flat)
    _qknorm_partial_rope_kernel[(flat.shape[0],)](
        out, flat, weight, cos, sin,
        head_dim, rotary_dim, rotary_dim // 2, heads, seq, eps,
        BLOCK=_next_pow2(head_dim), num_warps=4,
    )
    return out.view(batch, seq, heads, head_dim)


def _lora_input(base, delta):
    if delta is not None and (base.shape != delta.shape or base.dtype != delta.dtype
                              or base.device != delta.device):
        raise ValueError("LoRA base and branch must share shape, dtype and device")
    return delta


if HAVE_TRITON:
    @triton.jit
    def _lora_gate_residual_kernel(out_ptr, residual_ptr, base_ptr, delta_ptr,
                                   gate_ptr, index_ptr, size, cols, n_index,
                                   stride_gate, BLOCK: tl.constexpr):
        offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offset < size
        row, col = offset // cols, offset % cols
        idx = tl.load(index_ptr + row % n_index, mask=mask, other=0)
        base = tl.load(base_ptr + offset, mask=mask, other=0).to(tl.float32)
        delta = tl.load(delta_ptr + offset, mask=mask, other=0).to(tl.float32)
        branch = (base + delta).to(base_ptr.dtype.element_ty).to(tl.float32)
        gate = tl.load(gate_ptr + idx * stride_gate + col, mask=mask, other=0).to(tl.float32)
        # Eager first rounds gate*branch to BF16, then adds the BF16 residual.
        scaled = (gate * branch).to(base_ptr.dtype.element_ty).to(tl.float32)
        residual = tl.load(residual_ptr + offset, mask=mask, other=0).to(tl.float32)
        tl.store(out_ptr + offset, residual + scaled, mask=mask)


def fused_lora_gate_residual(residual, base, delta, gate, index):
    """PEFT sum + indexed gate + residual, preserving both BF16 boundaries."""
    _lora_input(base, delta)
    if delta is None:
        raise ValueError("fused LoRA residual requires a branch")
    if base.dtype != torch.bfloat16 or gate.dtype != base.dtype or residual.dtype != base.dtype:
        return residual + gate.index_select(0, index) * (base + delta)
    if residual.shape != base.shape or not index.numel():
        raise ValueError("LoRA residual shape or index mismatch")
    residual, base, delta = (x.contiguous() for x in (residual, base, delta))
    gate = _row_addressable(gate)
    out = torch.empty_like(base)
    if out.numel():
        _lora_gate_residual_kernel[(triton.cdiv(out.numel(), 1024),)](
            out, residual, base, delta, gate, index, out.numel(), out.shape[-1],
            index.numel(), gate.stride(0), BLOCK=1024, num_warps=4,
            enable_fp_fusion=False,
        )
    return out

if HAVE_TRITON:
    @triton.jit
    def _gate_residual_kernel(O,R,X,G,I,SIZE:tl.constexpr,C:tl.constexpr,NI:tl.constexpr,SG:tl.constexpr,BLOCK:tl.constexpr):
        ix=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK);mask=ix<SIZE
        row,col=ix//C,ix%C
        idx=tl.load(I+row%NI,mask,0)
        x=tl.load(X+ix,mask,0).to(tl.float32)
        g=tl.load(G+idx*SG+col,mask,0).to(tl.float32)
        r=tl.load(R+ix,mask,0).to(tl.float32)
        # Match eager multiply's materialized BF16 result before the addition.
        product=(x*g).to(X.dtype.element_ty).to(tl.float32)
        tl.store(O+ix,r+product,mask)

def fused_gate_residual(residual,branch,gate,index):
    """Indexed gated residual with eager BF16 rounding, including batch broadcast."""
    if branch.dtype!=torch.bfloat16 or gate.dtype!=branch.dtype or residual.dtype!=branch.dtype:
        return residual+gate.index_select(0,index)*branch
    if residual.shape!=branch.shape or index.numel()!=branch.shape[-2]:
        raise ValueError("Indexed residual shape mismatch")
    residual,branch=(x.contiguous() for x in (residual,branch))
    gate=_row_addressable(gate);out=torch.empty_like(branch)
    if out.numel():
        _gate_residual_kernel[(triton.cdiv(out.numel(),1024),)](out,residual,branch,gate,index,out.numel(),out.shape[-1],index.numel(),gate.stride(0),1024,num_warps=4,enable_fp_fusion=False)
    return out


@triton.jit
def batched_rope(O,X,W,C,S,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,RD:tl.constexpr,HALF:tl.constexpr,EPS:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,ROWS:tl.constexpr,R:tl.constexpr,BD:tl.constexpr):
    rows=tl.program_id(0)*R+tl.arange(0,R);cols=tl.arange(0,BD)
    tok=(rows//H)%T;batch=rows//(H*T);head=rows%H
    base=batch.to(tl.int64)*S0+tok.to(tl.int64)*S1+head*S2
    mask=(rows<ROWS)[:,None]&(cols<D)[None,:]
    x=tl.load(X+base[:,None]+cols[None,:],mask,0).to(tl.float32)
    variance=tl.sum(x*x,1)/D;inv=tl.rsqrt(variance+EPS)
    w=tl.load(W+cols,cols<D,0).to(tl.float32)
    norm=x*inv[:,None]*w[None,:]
    partner=tl.where(cols<HALF,cols+HALF,cols-HALF)
    pr=tl.load(X+base[:,None]+partner[None,:],(rows<ROWS)[:,None]&(cols<RD)[None,:],0).to(tl.float32)
    pw=tl.load(W+partner,cols<RD,0).to(tl.float32)
    pn=pr*inv[:,None]*pw[None,:]
    rotated=tl.where((cols<HALF)[None,:],-pn,pn)
    c=tl.load(C+tok[:,None]*RD+cols[None,:],(rows<ROWS)[:,None]&(cols<RD)[None,:],1).to(tl.float32)
    s=tl.load(S+tok[:,None]*RD+cols[None,:],(rows<ROWS)[:,None]&(cols<RD)[None,:],0).to(tl.float32)
    out=tl.where((cols<RD)[None,:],norm*c+rotated*s,norm)
    tl.store(O+rows[:,None].to(tl.int64)*D+cols[None,:],out,mask)


def fused_qknorm_rope_strided(x,w,c,s,eps):
    r,warps=1,4
    b,t,h,d=x.shape
    if c.shape[-1]>d or c.shape[-1]%2:raise ValueError("Invalid rotary width")
    if x.stride(-1)!=1:x=x.contiguous()
    out=torch.empty(x.shape,device=x.device,dtype=x.dtype)
    batched_rope[(triton.cdiv(b*t*h,r),)](out,x,w,c,s,t,h,d,c.shape[-1],c.shape[-1]//2,eps,*x.stride()[:3],b*t*h,r,triton.next_power_of_2(d),num_warps=warps)
    return out

KERNELS = {'pointwise': ('_residual_gate_rmsnorm_modulate_kernel', '_rmsnorm_modulate_kernel', '_swiglu_kernel', '_qknorm_partial_rope_kernel', '_lora_gate_residual_kernel', '_gate_residual_kernel'), 'rotary': ('batched_rope',)}
