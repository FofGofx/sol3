"""HunyuanVideo input rotary and pointwise block fusions."""

import torch
import triton
import triton.language as tl


from ._pointwise import _gate, _mod, _rope


@triton.jit
def rope_kernel(X,C,S,O,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,ROWS:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,C0:tl.constexpr,C1:tl.constexpr,SS0:tl.constexpr,SS1:tl.constexpr,BLOCK:tl.constexpr=256):
 ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=ix<ROWS*D
 ds=ix%D;h=ix//D%H;ts=ix//(D*H)%T;b=ix//(D*H*T)
 partner=ds^1
 x=tl.load(X+b.to(tl.int64)*S0+ts.to(tl.int64)*S1+h*S2+ds*S3,valid,0).to(tl.float32)
 y=tl.load(X+b.to(tl.int64)*S0+ts.to(tl.int64)*S1+h*S2+partner*S3,valid,0).to(tl.float32)
 y=tl.where(ds%2==0,-y,y)
 c=tl.load(C+ts.to(tl.int64)*C0+ds*C1,valid,0)
 s=tl.load(S+ts.to(tl.int64)*SS0+ds*SS1,valid,0)
 out=x*c+y*s
 tl.store(O+ix,out,valid)

KERNELS = {'pointwise': ('_mod', '_gate', '_rope'), 'rotary': ('rope_kernel',)}
