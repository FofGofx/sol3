"""Indexed modulation, residual gating and rotary embedding kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mod(X,S,A,I,O,N,D:tl.constexpr,ROWS:tl.constexpr,INDEXED:tl.constexpr,ST,AT,BLOCK:tl.constexpr):
 ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);r=ix//D;c=ix%D
 sr=tl.load(I+r,ix<N,0) if INDEXED else r%ROWS
 x=tl.load(X+ix,ix<N,0).to(tl.float32);s=tl.load(S+sr.to(tl.int64)*ST+c,ix<N,0);a=tl.load(A+sr.to(tl.int64)*AT+c,ix<N,0)
 add=(1.+s.to(tl.float32)).to(S.dtype.element_ty).to(tl.float32)
 mul=(x*add).to(X.dtype.element_ty).to(tl.float32)
 tl.store(O+ix,mul+a.to(tl.float32),ix<N)


@triton.jit
def _gate(X,Y,G,I,O,N,D:tl.constexpr,ROWS:tl.constexpr,INDEXED:tl.constexpr,GT,BLOCK:tl.constexpr):
 ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);r=ix//D;c=ix%D
 gr=tl.load(I+r,ix<N,0) if INDEXED else r%ROWS
 x=tl.load(X+ix,ix<N,0).to(tl.float32);y=tl.load(Y+ix,ix<N,0).to(tl.float32);g=tl.load(G+gr.to(tl.int64)*GT+c,ix<N,0).to(tl.float32)
 prod=(y*g).to(Y.dtype.element_ty).to(tl.float32)
 tl.store(O+ix,x+prod,ix<N)


@triton.jit
def _rope(X,C,S,O,N,T,H,D:tl.constexpr,R:tl.constexpr,BLOCK:tl.constexpr):
 ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);ds=ix%D;token=(ix//(H*D))%T
 x=tl.load(X+ix,ix<N,0);partner=tl.where(ds<R//2,ds+R//2,ds-R//2)
 y=tl.load(X+ix-ds+partner,(ix<N)&(ds<R),0).to(tl.float32);y=tl.where(ds<R//2,-y,y)
 c=tl.load(C+token*R+ds,(ix<N)&(ds<R),0).to(X.dtype.element_ty).to(tl.float32)
 s=tl.load(S+token*R+ds,(ix<N)&(ds<R),0).to(X.dtype.element_ty).to(tl.float32)
 first=(x.to(tl.float32)*c).to(X.dtype.element_ty).to(tl.float32);second=(y*s).to(X.dtype.element_ty).to(tl.float32)
 tl.store(O+ix,tl.where(ds<R,first+second,x.to(tl.float32)),ix<N)


@triton.jit
def _mod_index32(X,S,A,I,O,N,D:tl.constexpr,ROWS:tl.constexpr,INDEXED:tl.constexpr,ST,AT,BLOCK:tl.constexpr):
 ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);r=ix//D;c=ix%D
 sr=tl.load(I+r,ix<N,0) if INDEXED else r%ROWS
 x=tl.load(X+ix,ix<N,0).to(tl.float32);s=tl.load(S+sr*ST+c,ix<N,0);a=tl.load(A+sr*AT+c,ix<N,0)
 add=(1.+s.to(tl.float32)).to(S.dtype.element_ty).to(tl.float32)
 mul=(x*add).to(X.dtype.element_ty).to(tl.float32)
 tl.store(O+ix,mul+a.to(tl.float32),ix<N)


@triton.jit
def _gate_index32(X,Y,G,I,O,N,D:tl.constexpr,ROWS:tl.constexpr,INDEXED:tl.constexpr,GT,BLOCK:tl.constexpr):
 ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);r=ix//D;c=ix%D
 gr=tl.load(I+r,ix<N,0) if INDEXED else r%ROWS
 x=tl.load(X+ix,ix<N,0).to(tl.float32);y=tl.load(Y+ix,ix<N,0).to(tl.float32);g=tl.load(G+gr*GT+c,ix<N,0).to(tl.float32)
 prod=(y*g).to(Y.dtype.element_ty).to(tl.float32)
 tl.store(O+ix,x+prod,ix<N)
