"""SANA Stage 2 video-block modulation, gating and split RoPE."""

import torch
import triton
import triton.language as tl


from ._pointwise import _gate, _mod, _rope


@triton.jit
def _split_rope(X,C,S,O,N,T,H,D:tl.constexpr,R:tl.constexpr,CB,CH,CT,CR,SB,SH,ST,SR,BLOCK:tl.constexpr):
 ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);ds=ix%D;hh=(ix//D)%H;tt=(ix//(D*H))%T;bb=ix//(D*H*T)
 partner=tl.where(ds<R,ds+R,ds-R)
 x=tl.load(X+ix,ix<N,0).to(tl.float32);y=tl.load(X+ix-ds+partner,ix<N,0).to(tl.float32)
 c=tl.load(C+bb*CB+hh*CH+tt*CT+(ds%R)*CR,ix<N,0)
 s=tl.load(S+bb*SB+hh*SH+tt*ST+(ds%R)*SR,ix<N,0)
 product=x*c;ss=tl.where(ds<R,-s,s)
 z=product+ss*y
 tl.store(O+ix,z,ix<N)

KERNELS = {'pointwise': ('_mod', '_gate', '_rope'), 'rotary': ('_split_rope',)}
